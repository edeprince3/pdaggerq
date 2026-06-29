"""einsums (Einsums/Einsums) code-generation for pq_graph's ``to_strings("ir")``.

This is the einsums *dispatch*: it lowers the flat IR statement list

    { target:{name,indices,classes,is_intermediate}, is_assignment, coeff,
      operands:[{name,indices,classes,is_intermediate}, ...] }

into a sequence of einsums ``einsum(...)`` calls. Two things are einsums-specific
and handled here:

* ``einsum()`` is **binary** (one A, one B). An n-operand contraction is left-
  folded into n-1 pairwise einsums in pq_graph's (optimised) operand order,
  materialising one small intermediate per step. Each binary contraction
  dispatches to GEMM.
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


def lower(statements, operand_cxx, dim_of, temp_decl=default_temp_decl, indent="    "):
    """Lower an IR statement list to einsums C++ contraction-body lines.

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
    def dims(labels, lab2cls):
        return ", ".join(dim_of[lab2cls[l]] for l in labels)

    out = []
    w = lambda s: out.append(indent + s)
    declared_temps = set()

    for si, st in enumerate(statements):
        st, lab2cls = split_repeats(st)
        tgt = st["target"]
        ops = st["operands"]
        coeff = st["coeff"]
        cpref = "0.0" if st["is_assignment"] else "1.0"

        tgt_kind, tgt_cxx = operand_cxx(tgt["name"])
        tgt_idx = tgt["indices"]

        if tgt_kind == "temp" and tgt_cxx not in declared_temps:
            assert st["is_assignment"], f"temp {tgt_cxx} accumulated before declared"
            w(temp_decl(tgt_cxx, dims(tgt_idx, lab2cls)))
            declared_temps.add(tgt_cxx)
        target_ptr = tgt_cxx

        w(f"// [{si}] {'=' if st['is_assignment'] else '+='} {coeff} * "
          + " * ".join(o["name"] for o in ops))

        op_cxx = [operand_cxx(o["name"])[1] for o in ops]
        op_idx = [o["indices"] for o in ops]
        n = len(ops)

        if n == 1:
            # einsums has no prefactored unary einsum; the axpy / permute-into-temp
            # path is unused by current methods -- add it when first needed.
            raise NotImplementedError(
                "single-operand IR statement (add the axpy/permute path when a "
                "method first emits one)")

        # left-fold the n operands into n-1 pairwise binary einsums
        acc_cxx, acc_idx = op_cxx[0], op_idx[0]
        for k in range(1, n):
            last = (k == n - 1)
            if last:
                out_ptr, out_idx, ocpref, opref = "&" + target_ptr, tgt_idx, cpref, dbl(coeff)
            else:
                downstream = set(tgt_idx)
                for o in op_idx[k + 1:]:
                    downstream |= set(o)
                seen, keep = set(), []
                for l in list(acc_idx) + list(op_idx[k]):
                    if l in downstream and l not in seen:
                        seen.add(l)
                        keep.append(l)
                wname = f"w{si}_{k}"
                w(temp_decl(wname, dims(keep, lab2cls)))
                out_ptr, out_idx, ocpref, opref = "&" + wname, keep, "0.0", "1.0"

            w(f"einsum({ocpref}, Indices{{{indices(out_idx)}}}, {out_ptr}, {opref}, "
              f"Indices{{{indices(acc_idx)}}}, {acc_cxx}, "
              f"Indices{{{indices(op_idx[k])}}}, {op_cxx[k]});")

            if not last:
                acc_cxx, acc_idx = out_ptr[1:], out_idx  # drop the '&'

    return out
