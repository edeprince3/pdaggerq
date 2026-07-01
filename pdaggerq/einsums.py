"""einsums (Einsums/Einsums) code-generation for pq_graph's ``to_strings("ir")``.

This is the einsums *dispatch*: it lowers the flat IR statement list

    { target:{name,indices,classes,is_intermediate}, is_assignment, coeff,
      operands:[{name,indices,classes,is_intermediate}, ...] }

into a sequence of einsums ``einsum(...)`` calls. Two things are einsums-specific
and handled here:

* ``einsum()`` is **binary** (one A, one B). An n-operand contraction is left-
  folded into n-1 pairwise einsums in pq_graph's (optimised) operand order,
  materialising one small intermediate per step.
* einsums only GEMM-dispatches a binary contraction when each operand is laid out
  ``[free…, K…]`` or ``[K…, free…]`` with the contracted block ``K`` **contiguous
  at one end** (same K order in both) and the output is ``[freeA…, freeB…]`` --
  otherwise it silently falls to a ~20x slower generic scalar loop (the fused
  4-index "ladder" intermediates hit this). We therefore emit **TTGT**: permute
  each operand into that layout (HPTT, skipped when already so), GEMM into the
  natural ``[fA, fB]`` order, and fold any output permute into the accumulating
  ``permute`` into the target. Permutes are emitted only when an operand isn't
  already GEMM-ready; the K order is chosen to avoid permuting an operand when
  possible.
* a single tensor cannot be indexed with a **repeated label** (a diagonal, e.g. a
  proton trace ``B["QOO"][Q,J,J]``); ``split_repeats`` rewrites the summed
  diagonal into distinct contracted labels, exact when the operand is symmetric
  in that pair (the DF ``B`` blocks and ``Id`` are).

Everything project-specific -- how an IR tensor name maps to a C++ expression,
how an orbital-class char maps to a dimension, the function signature/namespace --
is supplied by the caller via ``operand_cxx`` and ``dim_of``. The caller emits the
wrapper; :func:`lower` returns just the contraction body lines. See
``examples``/neocc for a driver.

Index/class vocabulary (NEO-species-aware, from ``Line::type()``):
    o -> electron occ, v -> electron vir, O -> proton occ, V -> proton vir,
    Q -> DF aux, L -> excited state.
"""

import json
import re

# fresh-label pool for split_repeats (any case, einsums index:: symbols a..Z)
_POOL = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_ir(ir_lines):
    """Parse the lines of ``g.to_strings("ir")`` into a list of statement dicts."""
    return [json.loads(l) for l in ir_lines if l.strip().startswith("{")]


def target_shape(statements, target):
    """``(rank, classes)`` of an output/intermediate ``target`` as it appears in
    the parsed IR -- e.g. the residual ``R``'s index classes -- so a consumer need
    not hard-code them. ``classes`` use the ``Line::type()`` vocabulary
    (o/v electron, O/V proton, Q aux, L excited). Raises if the target is absent.
    """
    for st in statements:
        if st["target"]["name"] == target:
            classes = list(st["target"]["classes"])
            return len(classes), classes
    raise ValueError(f"target {target!r} not found in the IR")


def dbl(x):
    """A C++ double literal (einsum requires both prefactors the same type)."""
    return repr(float(x))


def sanitize(name):
    """A C++-identifier-safe form of an IR tensor name."""
    return re.sub(r"[^0-9A-Za-z]", "_", name)


def indices(labels):
    """``["a","i"]`` -> ``"index::a, index::i"``."""
    return ", ".join(f"index::{l}" for l in labels)


def split_repeats(stmt):
    """Rewrite diagonal-indexed operands (a label repeated within one tensor and
    summed, e.g. ``B["QOO"][Q,J,J] * Id["OO"][J,J]``) into distinct contracted
    labels (``B[Q,J,K] * Id[J,K]``). einsums cannot index a single tensor with a
    repeated label; the split pairs are symmetric so this is exact. Only summed
    labels (absent from the target) are split.

    Returns ``(statement, label->class map)``; the statement is a shallow copy
    with rewritten operand index lists and the map is extended for fresh labels.
    """
    tgt = stmt["target"]
    ops = stmt["operands"]
    target_labels = set(tgt["indices"])

    lab2cls = {}
    for t in [tgt] + ops:
        for l, c in zip(t["indices"], t["classes"]):
            lab2cls[l] = c
    used = set(lab2cls)

    repeated = set()
    for o in ops:
        for l in set(o["indices"]):
            if o["indices"].count(l) > 1 and l not in target_labels:
                repeated.add(l)
    if not repeated:
        return stmt, lab2cls

    fresh_for = {}  # (orig_label, occurrence_rank) -> fresh label
    new_ops = []
    for o in ops:
        seen = {}
        new_idx = []
        for l in o["indices"]:
            if l in repeated:
                r = seen.get(l, 0)
                seen[l] = r + 1
                if r == 0:
                    new_idx.append(l)
                else:
                    key = (l, r)
                    if key not in fresh_for:
                        nl = next(c for c in _POOL if c not in used)
                        used.add(nl)
                        fresh_for[key] = nl
                        lab2cls[nl] = lab2cls[l]
                    new_idx.append(fresh_for[key])
            else:
                new_idx.append(l)
        no = dict(o)
        no["indices"] = new_idx
        new_ops.append(no)

    ns = dict(stmt)
    ns["operands"] = new_ops
    return ns, lab2cls


def default_temp_decl(name, dims):
    """Declare a zeroed einsums intermediate (einsums outputs are pre-allocated)."""
    return f'auto {name} = create_zero_tensor("{name}", {dims});'


def _kblock(idx, kset):
    """If the contracted indices ``kset`` form a contiguous block at one end of
    ``idx``, return that block's label order (BLAS contracts it as op(A) with an
    optional transpose); otherwise None."""
    kpos = [i for i, l in enumerate(idx) if l in kset]
    if not kpos:
        return None
    contiguous = kpos == list(range(kpos[0], kpos[0] + len(kpos)))
    at_end = kpos[0] == 0 or kpos[-1] == len(idx) - 1
    return [idx[i] for i in kpos] if (contiguous and at_end) else None


def lower(statements, operand_cxx, dim_of, temp_decl=default_temp_decl, indent="    "):
    """Lower an IR statement list to einsums C++ contraction-body lines (TTGT).

    statements : list of IR dicts (already JSON-parsed, in emission order)
    operand_cxx: callable ``name -> (kind, cxx_expr)`` where kind is one of
                 ``"input"`` / ``"temp"`` / ``"param"``; maps an IR tensor name
                 (``B["Qov"]``, ``tmps_["1_QVO"]``, ``R`` ...) to its C++ form.
    dim_of     : mapping ``class_char -> C++ size expression`` (e.g. {"o": "no"}).
    temp_decl  : callable ``(cxx_name, dims) -> declaration line``.
    indent     : leading whitespace for each emitted line.

    Returns a list of C++ source lines (the body only; the caller emits the
    function signature, includes and namespace).
    """
    out = []
    w = lambda s: out.append(indent + s)
    declared_temps = set()

    for si, st in enumerate(statements):
        st, lab2cls = split_repeats(st)

        def dims(labels):
            return ", ".join(dim_of[lab2cls[l]] for l in labels)

        tgt = st["target"]
        ops = st["operands"]
        coeff = st["coeff"]
        cpref = "0.0" if st["is_assignment"] else "1.0"

        tgt_kind, tgt_cxx = operand_cxx(tgt["name"])
        tgt_idx = tgt["indices"]

        if tgt_kind == "temp" and tgt_cxx not in declared_temps:
            assert st["is_assignment"], f"temp {tgt_cxx} accumulated before declared"
            w(temp_decl(tgt_cxx, dims(tgt_idx)))
            declared_temps.add(tgt_cxx)

        w(f"// [{si}] {'=' if st['is_assignment'] else '+='} {coeff} * "
          + " * ".join(o["name"] for o in ops))

        op_cxx = [operand_cxx(o["name"])[1] for o in ops]
        op_idx = [list(o["indices"]) for o in ops]
        n = len(ops)

        if n == 1:
            # a single-operand statement is a (possibly permuted, scaled) copy --
            # an antisymmetrizer permutation or a bare-integral assignment. One
            # accumulating HPTT permute covers all four cases (assignment/accumulate
            # x identity/permuted): C = cpref*C + coeff*perm(A). einsums' permute
            # handles the identity permutation, so no index-match special case.
            w(f"permute({cpref}, Indices{{{indices(tgt_idx)}}}, &{tgt_cxx}, "
              f"{dbl(coeff)}, Indices{{{indices(op_idx[0])}}}, {op_cxx[0]});")
            continue

        # left-fold the n operands into n-1 pairwise binary einsums, each emitted
        # as a TTGT (permute-align operands -> GEMM -> permute output if needed)
        acc_cxx, acc_idx = op_cxx[0], op_idx[0]
        for k in range(1, n):
            last = (k == n - 1)
            b_cxx, b_idx = op_cxx[k], op_idx[k]

            # output index set of this pairwise step
            if last:
                out_set = set(tgt_idx)
            else:
                downstream = set(tgt_idx)
                for o in op_idx[k + 1:]:
                    downstream |= set(o)
                out_set = (set(acc_idx) | set(b_idx)) & downstream

            common = set(acc_idx) & set(b_idx)
            kset = {l for l in common if l not in out_set}   # contracted
            batch = common & out_set                          # in A, B and output
            fa = [l for l in acc_idx if l in out_set]          # A's free (A order)
            fb = [l for l in b_idx if l in out_set]            # B's free (B order)
            gemm_order = fa + fb

            if batch:
                # a batch index (appears in A, B and the output): not a plain
                # GEMM/outer product -- keep the generic einsum (rare).
                if last:
                    o_ptr, o_idx, ocp, op = "&" + tgt_cxx, tgt_idx, cpref, dbl(coeff)
                else:
                    keep, seen = [], set()
                    for l in acc_idx + b_idx:
                        if l in out_set and l not in seen:
                            seen.add(l)
                            keep.append(l)
                    wname = f"w{si}_{k}"
                    w(temp_decl(wname, dims(keep)))
                    o_ptr, o_idx, ocp, op = "&" + wname, keep, "0.0", "1.0"
                w(f"einsum({ocp}, Indices{{{indices(o_idx)}}}, {o_ptr}, {op}, "
                  f"Indices{{{indices(acc_idx)}}}, {acc_cxx}, "
                  f"Indices{{{indices(b_idx)}}}, {b_cxx});")
                if not last:
                    acc_cxx, acc_idx = o_ptr[1:], o_idx
                continue

            if kset:
                # a genuine contraction: permute so the contracted block K is
                # contiguous at one end of each operand (choose the K order that
                # lets an operand stay put -- prefer B, often the larger amplitude).
                korder = _kblock(b_idx, kset) or _kblock(acc_idx, kset) \
                    or [l for l in acc_idx if l in kset]

                if _kblock(acc_idx, kset) == korder:
                    a_use, a_idx = acc_cxx, acc_idx             # already [free,K]/[K,free]
                else:
                    a_idx = fa + korder
                    a_use = f"a{si}_{k}"
                    w(temp_decl(a_use, dims(a_idx)))
                    w(f"permute(Indices{{{indices(a_idx)}}}, &{a_use}, "
                      f"Indices{{{indices(acc_idx)}}}, {acc_cxx});")

                if _kblock(b_idx, kset) == korder:
                    b_use_, b_use_idx = b_cxx, b_idx
                else:
                    b_use_idx = korder + fb
                    b_use_ = f"b{si}_{k}"
                    w(temp_decl(b_use_, dims(b_use_idx)))
                    w(f"permute(Indices{{{indices(b_use_idx)}}}, &{b_use_}, "
                      f"Indices{{{indices(b_idx)}}}, {b_cxx});")
            else:
                # pure outer product (no contracted index): operands go in as-is.
                # The einsum below writes the concatenated [fa, fb] order -- each
                # operand's indices contiguous in the output -- which is the ONLY
                # layout einsums recognizes as a real outer product (Dispatch.hpp
                # einsum_is_outer_product). The final permute reorders to the target.
                a_use, a_idx = acc_cxx, acc_idx
                b_use_, b_use_idx = b_cxx, b_idx

            if not last:
                # result stored in natural [fa, fb] order (GEMM / outer product);
                # becomes the accumulator for the next fold step
                wname = f"w{si}_{k}"
                w(temp_decl(wname, dims(gemm_order)))
                w(f"einsum(0.0, Indices{{{indices(gemm_order)}}}, &{wname}, 1.0, "
                  f"Indices{{{indices(a_idx)}}}, {a_use}, "
                  f"Indices{{{indices(b_use_idx)}}}, {b_use_});")
                acc_cxx, acc_idx = wname, gemm_order
            elif gemm_order == tgt_idx:
                # GEMM straight into the target (accumulating with cpref)
                w(f"einsum({cpref}, Indices{{{indices(tgt_idx)}}}, &{tgt_cxx}, {dbl(coeff)}, "
                  f"Indices{{{indices(a_idx)}}}, {a_use}, "
                  f"Indices{{{indices(b_use_idx)}}}, {b_use_});")
            else:
                # GEMM into natural order, then accumulate-permute into the target
                gname = f"g{si}_{k}"
                w(temp_decl(gname, dims(gemm_order)))
                w(f"einsum(0.0, Indices{{{indices(gemm_order)}}}, &{gname}, 1.0, "
                  f"Indices{{{indices(a_idx)}}}, {a_use}, "
                  f"Indices{{{indices(b_use_idx)}}}, {b_use_});")
                w(f"permute({cpref}, Indices{{{indices(tgt_idx)}}}, &{tgt_cxx}, {dbl(coeff)}, "
                  f"Indices{{{indices(gemm_order)}}}, {gname});")

    return out
