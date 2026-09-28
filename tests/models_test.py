"""Tests for the CC model library (pdaggerq.models).

Structural checks over every model are exhaustive and instant; generation is
exercised only on the cheap models (the high-rank residuals -- ccsdt/ccsdtq and
the full/hybrid NEO triples/quadruples -- are correct but slow to build, so they
are covered by the dedicated examples, not here). Run: pytest tests/models_test.py
"""

import math

from pdaggerq import einsums, models


def test_models_present_and_projected():
    expected = {
        "ccd", "ccsd", "ccsdt", "ccsdtq",
        "neo-ccd", "neo-ccsd", "neo-ccsdt", "neo-ccsdtq",
        "neo-ccd(ep)", "neo-ccsdt(eep)", "neo-ccsdtq(eeep)",
        "ccsd(t)", "neo-ccsd(t)",
    }
    assert expected <= set(models.MODELS), expected - set(models.MODELS)
    # every amplitude of every model -- iterated and perturbative -- has a projection
    for m in models.MODELS.values():
        for amp in m.T + m.T_pt:
            assert amp in models.PROJECTION, f"{m.name}: no projection for {amp}"
    # the hybrids drop the matching pure-electron excitation
    assert "t3" not in models.model("neo-ccsdt(eep)").T   # eep, no electron t3
    assert "t4" not in models.model("neo-ccsdtq(eeep)").T  # eeep, no electron t4
    assert models.model("neo-ccd(ep)").T == ("tep11",)     # minimal e-p model
    # NEO models are general in the proton count: proton doubles (tp2) and the
    # proton-proton fluctuation (vp) are present wherever the rank allows (they vanish
    # for a single proton). neo-ccd(ep) stays the minimal single-proton model.
    assert "tp2" in models.model("neo-ccsd").T
    assert "tp2" in models.model("neo-ccsdt(eep)").T
    assert models.model("neo-ccsd").H == ("f", "v", "fp", "gep", "vp")
    assert models.model("neo-ccd(ep)").H == ("f", "v", "fp", "gep")   # single-proton
    assert models.model("ccsd").H == ("f", "v")
    print("test_models_present_and_projected OK")


def test_single_proton_models():
    # every vp-model has an auto-derived "<name>-1p" single-proton counterpart: vp
    # stripped from H, all >=2-proton amplitudes dropped. Numerically bit-for-bit with
    # the full model at one proton (verified separately to ~1e-14/energy exact).
    assert models._proton_count("t2") == 0
    assert models._proton_count("tp1") == 1 and models._proton_count("tp4") == 4
    assert models._proton_count("tep11") == 1 and models._proton_count("tep21") == 1
    assert models._proton_count("tep12") == 2 and models._proton_count("tep13") == 3

    expect = {
        "neo-ccd-1p":          ("t2", "tep11"),
        "neo-ccsd-1p":         ("t1", "t2", "tp1", "tep11"),
        "neo-ccsdt-1p":        ("t1", "t2", "t3", "tp1", "tep11", "tep21"),
        "neo-ccsdtq-1p":       ("t1", "t2", "t3", "t4", "tp1", "tep11", "tep21", "tep31"),
        "neo-ccsdt(eep)-1p":   ("t1", "t2", "tp1", "tep11", "tep21"),
        "neo-ccsdtq(eeep)-1p": ("t1", "t2", "tp1", "tep11", "tep21", "tep31"),
    }
    for name, T in expect.items():
        m = models.model(name)
        assert m.T == T, (name, m.T)
        assert m.H == ("f", "v", "fp", "gep"), (name, m.H)   # H_NEO, no vp
        assert "vp" not in m.H and not any(models._proton_count(a) >= 2 for a in m.T)
    # models already at/below one proton gain nothing -> no -1p entry
    assert "neo-ccd(ep)-1p" not in models.MODELS
    assert "ccsd-1p" not in models.MODELS
    # lambda names follow the trimmed set; the reduced model still generates
    assert models.lambda_amps("neo-ccsd-1p") == ["l1", "l2", "lp1", "lep11"]
    assert any(l.strip().startswith("{") for l in models.energy_graph("neo-ccsd-1p").to_strings("ir"))
    assert any(l.strip().startswith("{")
               for l in models.residual_ir("neo-ccsd-1p", "tep11"))
    # The -1p models are NOT redundant with the full ones, and a consumer cannot reach
    # them by gating amplitudes on the particle count: the one- and many-proton
    # Hamiltonians differ. Deleting the >=2-proton amplitudes leaves every vp term
    # standing, and for a single proton those describe it interacting with itself.
    import re
    from pdaggerq._pdaggerq import pq_helper

    def resid(H, T, proj):
        pq = pq_helper("fermi")
        pq.set_left_operators([[proj]])
        for h in H:
            pq.add_st_operator(1.0, [h], list(T), True)
        pq.simplify()
        return [" ".join(t) for t in pq.strings()]

    full = models.model("neo-ccsd")
    gated_T = [a for a in full.T if models._proton_count(a) <= 1]   # what gating leaves
    one_p = models.model("neo-ccsd-1p")
    assert list(one_p.T) == gated_T, (list(one_p.T), gated_T)      # same cluster ...
    assert "vp" in full.H and "vp" not in one_p.H                  # ... different H

    proj = models.PROJECTION["tp1"]
    gated = resid(full.H, gated_T, proj)        # amplitudes gated, many-proton H kept
    true_1p = resid(one_p.H, gated_T, proj)     # the -1p model
    extra = [t for t in gated if t not in true_1p]
    assert extra, "gating amplitudes alone should leave vp terms behind"
    assert len(gated) > len(true_1p), (len(gated), len(true_1p))
    # every surviving term is pure proton-proton: an antisymmetrized integral over four
    # nuclear labels, which is exactly the self-interaction a lone proton must not have
    for t in extra:
        assert re.search(r"<n\w+,n\w+\|\|n\w+,n\w+>", t), t

    print("test_single_proton_models OK")


def test_cheap_models_generate():
    # cheap residuals must be non-empty (and, implicitly, generate without error)
    for name, amp in [("ccd", "t2"), ("neo-ccd(ep)", "tep11")]:
        ir = [l for l in models.residual_ir(name, amp) if l.strip().startswith("{")]
        assert ir, f"{name}/{amp} generated an empty residual"
    # energy too
    e = [l for l in models.energy_graph("ccd").to_strings("ir") if l.strip().startswith("{")]
    assert e, "ccd energy generated empty"
    print("test_cheap_models_generate OK")


def test_bad_lookups_raise():
    try:
        models.model("ccsdtqp")
        assert False, "expected KeyError"
    except KeyError:
        pass
    try:
        models.residual_graph("ccd", "t1")   # ccd has no singles
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("test_bad_lookups_raise OK")


def test_spin_axis():
    # case enumeration (electron alpha/beta; NEO high-spin vs full nuclear manifold)
    assert models.spin_cases("t2") == ["aaaa", "abab", "bbbb"]
    assert models.spin_cases("tep11", "high-spin") == ["aa_n", "bb_n"]
    assert models.spin_cases("tep11", "full") == ["aa_naa", "aa_nbb", "bb_naa", "bb_nbb"]
    # spin-orbital default still generates; a spin block also generates non-empty
    so = [l for l in models.residual_ir("neo-ccd(ep)", "tep11") if l.strip().startswith("{")]
    blk = [l for l in models.residual_ir("neo-ccd(ep)", "tep11", spin_case="aa_n")
           if l.strip().startswith("{")]
    assert so and blk, (len(so), len(blk))
    try:
        models.residual_ir("neo-ccd(ep)", "tep11", spin_case="zz")
        assert False, "expected ValueError for an unknown spin_case"
    except ValueError:
        pass
    print("test_spin_axis OK")


def test_lambda_and_gradient():
    # de-excitation naming (leading t -> l) and full excitation-operator coverage
    assert models.lambda_amps("neo-ccsd") == ["l1", "l2", "lp1", "lp2", "lep11"]
    assert models.lambda_amps("ccd") == ["l2"]
    for m in models.MODELS.values():
        for amp in m.T:
            assert amp in models.EXCITATION, f"{m.name}: no excitation for {amp}"
    # generation on the cheapest model: Lambda is rank-4, gradient rank-2 per species
    lam = einsums.parse_ir(models.lambda_ir("neo-ccd(ep)", "tep11"))
    assert lam and einsums.target_shape(lam, "R")[0] == 4, len(lam)
    gp = einsums.parse_ir(models.gradient_ir("neo-ccd(ep)", "proton"))
    assert gp and einsums.target_shape(gp, "R") == (2, ["V", "O"]), einsums.target_shape(gp, "R")
    # error paths
    try:
        models.gradient_graph("neo-ccd(ep)", "muon")
        assert False, "expected ValueError for a bad species"
    except ValueError:
        pass
    try:
        models.lambda_graph("ccd", "t1")     # ccd has no singles
        assert False, "expected ValueError for a missing amplitude"
    except ValueError:
        pass
    print("test_lambda_and_gradient OK")


def test_lambda_consistency():
    """The Lambda equations are LINEAR in the multipliers, and the true CC Lambda system
    is consistent: some lambda makes every residual block exactly zero. Assemble the full
    coupled system A.lambda + b from lambda_ir (evaluating its residual at lambda = 0 and at
    unit vectors), solve, and assert the residual at the solution is ~0. A dropped / mispaired
    / duplicated term makes the emitted system INCONSISTENT (no lambda gives R = 0) -- which
    is exactly what a non-converging Jacobi/DIIS solve on a linear system means. Also assert
    the optimized (opt6) emission reproduces the opt0 solution, so opt6 is not just internally
    consistent but computes the SAME system (a wrong-but-consistent variant is caught too)."""
    import itertools
    import numpy as np
    from collections import defaultdict

    DIM = {"o": 2, "v": 3, "O": 1, "V": 2, "Q": 5}
    VIR, OCC = {"v", "V"}, {"o", "O"}

    def antisym(a, cl):
        out = a.copy(); groups = defaultdict(list)
        for ax, c in enumerate(cl): groups[c].append(ax)
        for c, axes in groups.items():
            if len(axes) >= 2 and (c in VIR or c in OCC):
                perms = list(itertools.permutations(range(len(axes)))); acc = np.zeros_like(out)
                for p in perms:
                    par = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j]) & 1
                    src = list(range(out.ndim))
                    for k, ax in enumerate(axes): src[ax] = axes[p[k]]
                    acc += (-1 if par else 1) * np.transpose(out, src)
                out = acc / len(perms)
        return out

    def interp(ir, inp):
        st = {}
        val = lambda o: st[o["name"]] if o["name"] in st else inp[o["name"]]
        for s in ir:
            sub = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            c = s["coeff"] * np.einsum(sub + "->" + out, *[val(o) for o in s["operands"]], optimize=True)
            t = s["target"]["name"]; st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st["R"]

    def assemble(name, opt, base=None):
        amps = list(models.model(name).T)                 # one residual block per t-amplitude
        irs = {a: einsums.parse_ir(models.lambda_ir(name, a, opt_level=opt)) for a in amps}
        names = {}
        for ir in irs.values():
            produced = {s["target"]["name"] for s in ir}
            for s in ir:
                for o in s["operands"]:
                    if o["name"] not in produced:
                        names[o["name"]] = tuple(o["classes"])
        if base is None:                                  # integrals + t-amplitudes (fixed inputs)
            rng = np.random.default_rng(1); base = {}
            for n, c in sorted(names.items()):
                if n.startswith("l"): continue
                a = rng.standard_normal(tuple(DIM[x] for x in c))
                base[n] = antisym(a, c) if (n.startswith("t") and len(c) >= 4) else a
        lnames = sorted(n for n in names if n.startswith("l"))   # the multipliers = unknowns
        shp = {n: [DIM[x] for x in names[n]] for n in lnames}
        siz = {n: int(np.prod(shp[n])) for n in lnames}
        off = {}; tot = 0
        for n in lnames: off[n] = tot; tot += siz[n]

        def R(lvec):
            inp = dict(base)
            for n in lnames: inp[n] = lvec[off[n]:off[n] + siz[n]].reshape(shp[n])
            return np.concatenate([interp(irs[a], inp).flatten() for a in amps])

        b = R(np.zeros(tot)); A = np.zeros((b.size, tot))
        for k in range(tot):
            e = np.zeros(tot); e[k] = 1.0; A[:, k] = R(e) - b
        return A, b, base, R

    for name in ("ccsd", "neo-ccd(ep)", "neo-ccsd-1p"):
        A0, b0, base, R0 = assemble(name, 0)
        x0, *_ = np.linalg.lstsq(A0, -b0, rcond=None)
        assert np.abs(A0 @ x0 + b0).max() < 1e-9, (name, "opt0 Lambda system INCONSISTENT")
        _, _, _, R6 = assemble(name, 6, base=base)        # opt6 residual, same inputs
        assert np.abs(R6(x0)).max() < 1e-9, (name, "opt6 Lambda disagrees with opt0")
    print("test_lambda_consistency OK")


def test_rdm():
    import pdaggerq

    assert {"rdm_ir", "rdm_graph"} <= set(models.__all__)

    # blocks generate with the right rank; proton n-labels pick up O/V classes
    d_oo = einsums.parse_ir(models.rdm_ir("ccsd", "e1(i,j)"))
    assert d_oo and einsums.target_shape(d_oo, "D") == (2, ["o", "o"])
    d_vvoo = einsums.parse_ir(models.rdm_ir("ccsd", "e2(a,b,i,j)"))
    assert d_vvoo and einsums.target_shape(d_vvoo, "D") == (4, ["v", "v", "o", "o"])
    d_pOO = einsums.parse_ir(models.rdm_ir("neo-ccd(ep)", "e1(ni,nj)"))
    assert d_pOO and einsums.target_shape(d_pOO, "D") == (2, ["O", "O"])

    # regression: a nuclear index whose letter is a reserved sigma label (L/R/X/Y)
    # must classify as proton occ, not an excited-state line. e2(nI,nJ,nL,nK) was
    # rank-3 with a bogus 'L'-typed identity before the Line-ctor fix; it must now
    # match the lowercase block's structure and carry no sig ('L') class.
    up = einsums.parse_ir(models.rdm_ir("neo-ccd(ep)", "e2(nI,nJ,nL,nK)"))
    lo = einsums.parse_ir(models.rdm_ir("neo-ccd(ep)", "e2(ni,nj,nl,nk)"))
    up_ranks = sorted(len(st["target"]["indices"]) for st in up)
    up_cls = {c for st in up for o in st["operands"] for c in o["classes"]}
    assert 4 in up_ranks, up_ranks
    assert up_ranks == sorted(len(st["target"]["indices"]) for st in lo)
    assert "L" not in up_cls, up_cls

    # the construction matches examples/ccsd_d2.py for a 2-RDM block
    def strs(setup):
        pq = pdaggerq.pq_helper("fermi")
        setup(pq)
        pq.simplify()
        return sorted(" ".join(t) for t in pq.strings())

    def ref(pq):
        pq.set_left_operators([["1"], ["l1"], ["l2"]])
        pq.add_st_operator(1.0, ["e2(i,a,l,k)"], ["t1", "t2"])

    def mine(pq):
        pq.set_left_operators([["1"]] + [[l] for l in models.lambda_amps("ccsd")])
        pq.add_st_operator(1.0, ["e2(i,a,l,k)"], list(models.model("ccsd").T))

    assert strs(ref) == strs(mine)
    print("test_rdm OK")


def _slices(ref):
    d = ref["dims"]; ne = d["o"] + d["v"]; npr = d["O"] + d["V"]
    return {"o": slice(0, d["o"]), "v": slice(d["o"], ne),
            "O": slice(0, d["O"]), "V": slice(d["O"], npr)}


def _integral_block(op, ref):
    """The h/hp/g/gep block an energy_from_rdm_ir operand refers to."""
    SL = _slices(ref)
    base, blk = op["name"].split('["')[0], op["name"].split('"')[1]
    src = {"h": ref["h"], "hp": ref.get("hp"), "g": ref["g"], "gep": ref.get("gep")}[base]
    return src[tuple(SL[c] for c in blk)]


def _rdm_at_zero_amps(ir, target, classes, ref):
    """Evaluate an rdm_block_ir block at ZERO amplitudes: keep only the statements whose
    operands are all reference quantities (Id / already-built intermediates); any term
    touching an amplitude vanishes."""
    import numpy as np
    D = ref["dims"]
    store = {}
    for s in ir:
        arrs, skip = [], False
        for op in s["operands"]:
            nm = op["name"]
            if nm.startswith("Id["):
                arrs.append(np.eye(D[op["classes"][0]]))
            elif nm in store:
                arrs.append(store[nm])
            else:                        # an amplitude -> the whole term is zero at t=0
                skip = True
                break
        if skip:
            continue
        out = "".join(s["target"]["indices"])
        sub = ",".join("".join(o["indices"]) for o in s["operands"])
        c = s["coeff"] * np.einsum(sub + "->" + out, *arrs, optimize=True)
        t = s["target"]["name"]
        store[t] = c.copy() if s["is_assignment"] else store[t] + c
    return store.get(target, np.zeros(tuple(D[c] for c in classes)))


def test_energy_from_rdm():
    import itertools, json
    import numpy as np
    assert "energy_from_rdm_ir" in models.__all__
    # electronic: E is a scalar tracing the electron RDM blocks D1/D2 against integrals
    e = einsums.parse_ir(models.energy_from_rdm_ir("ccsd"))
    assert e and einsums.target_shape(e, "E") == (0, [])          # scalar energy
    bases = {o["name"].split('["')[0] for st in e for o in st["operands"]}
    assert {"h", "g", "D1", "D2"} <= bases, bases
    assert len(e) == 4 + 16                                       # all 1- and 2-body o/v blocks
    # NEO also traces the proton (D1_n) and mixed e-p (D2_ep) RDMs
    ep = {o["name"].split('["')[0]
          for st in einsums.parse_ir(models.energy_from_rdm_ir("neo-ccd(ep)")) for o in st["operands"]}
    assert {"D1", "D2", "hp", "D1_n", "gep", "D2_ep"} <= ep, ep

    # numeric: the block sum reproduces the full trace E = h.D1 + 1/2 g.D2 exactly
    no, nv = 2, 3
    nmo = no + nv
    SL = {"o": slice(0, no), "v": slice(no, nmo)}
    rng = np.random.default_rng(0)
    full = {n: rng.standard_normal((nmo,) * r) for n, r in (("h", 2), ("g", 4), ("D1", 2), ("D2", 4))}
    E = 0.0
    for st in e:
        arrs = []
        for op in st["operands"]:
            nm, blk = op["name"].split('["')
            arrs.append(full[nm][tuple(SL[c] for c in blk.rstrip('"]'))])
        subs = ",".join("".join(op["indices"]) for op in st["operands"])
        E += st["coeff"] * np.einsum(f"{subs}->", *arrs, optimize=True)
    # the g.D2 pairing is ("D2", [0,1,3,2]): g's last two slots meet D2's last two swapped
    ref = np.einsum("pq,pq->", full["h"], full["D1"]) + 0.5 * np.einsum("abcd,abdc->", full["g"], full["D2"])
    assert abs(E - ref) < 1e-10, (E, ref)

    # ---- FULL-ENERGY IDENTITY (the end-to-end guard) --------------------------------
    # models.rdm_energy_reference IS the published consumer contract (its docstring
    # together with energy_from_rdm_ir's): it builds h/hp/g/gep from raw randoms per
    # the documented recipe (h = f - mf_ee so the e-p mean field stays in h; hp = fp
    # dressed and unchanged; g plain physicist; gep the equations' own signed tensor),
    # evaluates every rdm_block_ir block plus the energy_from_rdm_ir trace, and
    # independently evaluates the raw <(1+L)H> Lagrangian from pq.strings. Asserting
    # their agreement here pins the contract; a wrong g.D2 slot pairing, a wrong gep
    # sign, or a pq_graph-corrupted D1_n all show up here and nowhere else.
    for mdl in ("ccsd", "neo-ccd(ep)"):
        ref = models.rdm_energy_reference(mdl)
        # (a) ALGEBRAIC identity: the RDM trace == the raw-H Lagrangian
        assert abs(ref["E_lagrangian"] - ref["E_rdm"]) < 1e-9, \
            (mdl, ref["E_lagrangian"], ref["E_rdm"])
        assert np.allclose(ref["h"], ref["f"] - ref["mf_ee"])
        if "hp" in ref:
            assert np.allclose(ref["hp"], ref["fp"])
            assert any(b == "D2_ep" for b, _ in ref["rdm"]), sorted(ref["rdm"])

        # (b) PHYSICAL identity -- the check the algebraic one CANNOT make. (a) compares
        # the RDM trace against the raw-H Lagrangian built from the SAME one-body inputs,
        # so any mis-dressing of h/hp cancels on both sides and is invisible. It hid a
        # real error: the contract used to say h = f - mf_ee (keeping the e-p mean field
        # in h) and hp = fp, which counts the e-p mean field THREE times -- E(t=0) came
        # out high by exactly 2*sum_iI gep(i,I,i,I). The one-body operators must be the
        # BARE cores. Assert the RDM energy at ZERO amplitudes reproduces the (NEO-)HF
        # reference energy built independently from those bare cores.
        E0 = 0.0
        for line in models.energy_from_rdm_ir(mdl):
            st = json.loads(line)
            arrs, skip = [], False
            for op in st["operands"]:
                base, blk = op["name"].split('["')[0], op["name"].split('"')[1]
                if base in ("h", "hp", "g", "gep"):
                    arrs.append(None)          # integral, filled below
                else:                          # RDM block at t=0: reference part only
                    ir = einsums.parse_ir(models.rdm_block_ir(mdl, base, blk))
                    arrs.append(_rdm_at_zero_amps(ir, f'{base}["{blk}"]', op["classes"], ref))
            arrs = [a if a is not None else _integral_block(op, ref)
                    for a, op in zip(arrs, st["operands"])]
            sub = ",".join("".join(op["indices"]) for op in st["operands"])
            E0 += st["coeff"] * float(np.einsum(sub + "->", *arrs, optimize=True))
        assert abs(E0 - ref["E_hf"]) < 1e-9, (
            mdl, "E_rdm(t=0) != HF reference -- are h/hp the BARE cores?", E0, ref["E_hf"])

    print("test_energy_from_rdm OK")


# interpret block-named IR (D1["ov"], g["vovo"], Id["oo"], ...) on random full tensors
def _interp_block(ir, full, no, nv):
    import numpy as np
    SL = {"o": slice(0, no), "v": slice(no, no + nv)}
    DIM = {"o": no, "v": nv}
    store = {}
    for st in ir:
        t = st["target"]
        if st["is_assignment"] or t["name"] not in store:
            store[t["name"]] = np.zeros(tuple(DIM[c] for c in t["classes"]))
        arrs = []
        for op in st["operands"]:
            base, blk = op["name"].split('["')
            arrs.append(full[base][tuple(SL[c] for c in blk.rstrip('"]'))])
        subs = ",".join("".join(op["indices"]) for op in st["operands"])
        store[t["name"]] = store[t["name"]] + st["coeff"] * np.einsum(
            f"{subs}->{''.join(t['indices'])}", *arrs, optimize=True)
    return store


def test_orbital_gradient_hessian():
    import numpy as np
    assert {"orbital_gradient_ir", "orbital_hessian_ir"} <= set(models.__all__)
    g = einsums.parse_ir(models.orbital_gradient_ir("ccsd", "electron"))
    assert einsums.target_shape(g, "grad") == (2, ["v", "o"])       # vir-occ gradient block
    bases = {o["name"].split('["')[0] for st in g for o in st["operands"]}
    assert {"h", "g", "D1", "D2"} <= bases
    # the previously-truncated occ blocks are now present (the fix)
    assert any(o["name"] == 'D1["oo"]' for st in g for o in st["operands"])
    # any well-formed cross term (none here) must keep the canonical e/p slot pattern
    for o in (o for st in g for o in st["operands"]):
        if o["name"].split('["')[0] in ("gep", "D2_ep"):
            assert ["p" if c in "OV" else "e" for c in o["classes"]] in (["e", "p", "e", "p"], ["p", "e", "e", "p"])
    # NEO electron gradient: the hand-derived e-p (gep/D2_ep) terms are well-formed
    # 2e2p and reproduce the finite-diff-validated formula g = T1 - T2 - T3 + T4.
    gneo = einsums.parse_ir(models.orbital_gradient_ir("neo-ccd(ep)", "electron"))
    gep_st = [st for st in gneo
              if any(o["name"].split('["')[0] in ("gep", "D2_ep") for o in st["operands"])]
    assert gep_st
    for st in gep_st:
        for o in st["operands"]:
            if o["name"].split('["')[0] in ("gep", "D2_ep"):
                assert ["p" if c in "OV" else "e" for c in o["classes"]] in (["e", "p", "e", "p"], ["p", "e", "e", "p"])
    no, nv, nO, nV = 2, 3, 1, 2
    ne, npp = no + nv, nO + nV
    SL = {"o": slice(0, no), "v": slice(no, ne), "O": slice(0, nO), "V": slice(nO, npp)}
    rng = np.random.default_rng(1)
    full = {"gep": rng.standard_normal((ne, npp, ne, npp)), "D2_ep": rng.standard_normal((npp, ne, ne, npp))}
    gg = np.zeros((nv, no))
    for st in gep_st:
        arrs = [full[o["name"].split('["')[0]][tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                for o in st["operands"]]
        subs = ",".join("".join(o["indices"]) for o in st["operands"])
        gg += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
    G, D = full["gep"], full["D2_ep"]
    form = (np.einsum("ePXQ,PeYQ->XY", G, D) - np.einsum("YPeQ,PXeQ->XY", G, D)
            - np.einsum("ePYQ,PeXQ->XY", G, D) + np.einsum("XPeQ,PYeQ->XY", G, D))
    assert np.max(np.abs(gg - form[no:, :no])) < 1e-10

    # electron Hessian: rank-4 vir-vir-occ-occ block
    hee = einsums.parse_ir(models.orbital_hessian_ir("ccsd"))
    assert einsums.target_shape(hee, "H") == (4, ["v", "v", "o", "o"])

    # NEO electron Hessian: hand-derived gep part (8 cross + 8 delta terms) is well-formed
    # 2e2p and reproduces the closed form d^2 E_ep/dkappa^2 (finite-diff-validated).
    hgep = [st for st in einsums.parse_ir(models.orbital_hessian_ir("neo-ccd(ep)", "electron"))
            if any(o["name"].split('["')[0] in ("gep", "D2_ep") for o in st["operands"])]
    assert hgep
    for st in hgep:
        for o in st["operands"]:
            if o["name"].split('["')[0] in ("gep", "D2_ep"):
                assert ["p" if c in "OV" else "e" for c in o["classes"]] in (["e", "p", "e", "p"], ["p", "e", "e", "p"])
    full["Id"] = np.eye(ne)
    HH = np.zeros((nv, nv, no, no))
    for st in hgep:
        arrs = [full[o["name"].split('["')[0]][tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                for o in st["operands"]]
        subs = ",".join("".join(o["indices"]) for o in st["operands"])
        HH += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
    o, v = slice(0, no), slice(no, ne)
    ee = np.einsum
    ref = (ee("aPbQ,PijQ->abij", G[v, :, v, :], D[:, o, o, :]) - ee("aPjQ,PibQ->abij", G[v, :, o, :], D[:, o, v, :])
           - ee("iPbQ,PajQ->abij", G[o, :, v, :], D[:, v, o, :]) + ee("iPjQ,PabQ->abij", G[o, :, o, :], D[:, v, v, :])
           + ee("bPaQ,PjiQ->abij", G[v, :, v, :], D[:, o, o, :]) - ee("bPiQ,PjaQ->abij", G[v, :, o, :], D[:, o, v, :])
           - ee("jPaQ,PbiQ->abij", G[o, :, v, :], D[:, v, o, :]) + ee("jPiQ,PbaQ->abij", G[o, :, o, :], D[:, v, v, :]))
    eo, ev = np.eye(no), np.eye(nv)
    for M in (ee("aPpQ,PbpQ->ab", G[v, :, :, :], D[:, v, :, :]), ee("bPpQ,PapQ->ab", G[v, :, :, :], D[:, v, :, :]),
              ee("pPaQ,PpbQ->ab", G[:, :, v, :], D[:, :, v, :]), ee("pPbQ,PpaQ->ab", G[:, :, v, :], D[:, :, v, :])):
        ref -= 0.5 * ee("ab,ij->abij", M, eo)
    for M in (ee("iPpQ,PjpQ->ij", G[o, :, :, :], D[:, o, :, :]), ee("jPpQ,PipQ->ij", G[o, :, :, :], D[:, o, :, :]),
              ee("pPiQ,PpjQ->ij", G[:, :, o, :], D[:, :, o, :]), ee("pPjQ,PpiQ->ij", G[:, :, o, :], D[:, :, o, :])):
        ref -= 0.5 * ee("ij,ab->abij", M, ev)
    assert np.max(np.abs(HH - ref)) < 1e-10

    # NEO proton-row gradient & Hessian (bare proton core relabel + gep proton terms):
    # well-formed, and the gep gradient part reproduces its finite-diff-validated form.
    pg = einsums.parse_ir(models.orbital_gradient_ir("neo-ccd(ep)", "proton"))
    ph = einsums.parse_ir(models.orbital_hessian_ir("neo-ccd(ep)", "proton"))
    assert einsums.target_shape(pg, "grad") == (2, ["V", "O"])
    assert einsums.target_shape(ph, "H") == (4, ["V", "V", "O", "O"])
    for ir in (pg, ph):
        for st in ir:
            for op in st["operands"]:
                b = op["name"].split('["')[0]
                if b in ("gep", "D2_ep"):
                    assert ["p" if c in "OV" else "e" for c in op["classes"]] in (["e", "p", "e", "p"], ["p", "e", "e", "p"])
                if b in ("hp", "D1_n"):
                    assert all(c in "OV" for c in op["classes"])
    SLp = {"o": slice(0, no), "v": slice(no, ne), "O": slice(0, nO), "V": slice(nO, npp)}
    pgg = np.zeros((nV, nO))
    for st in (s for s in pg if any(op["name"].split('["')[0] in ("gep", "D2_ep") for op in s["operands"])):
        arrs = [full[op["name"].split('["')[0]][tuple(SLp[c] for c in op["name"].split('["')[1].rstrip('"]'))]
                for op in st["operands"]]
        subs = ",".join("".join(op["indices"]) for op in st["operands"])
        pgg += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
    pform = (np.einsum("EPFX,PEFY->XY", G, D) - np.einsum("EYFQ,XEFQ->XY", G, D)
             - np.einsum("EPFY,PEFX->XY", G, D) + np.einsum("EXFQ,YEFQ->XY", G, D))
    assert np.max(np.abs(pgg - pform[nO:, :nO])) < 1e-10

    # NEO e-p CROSS Hessian block H_ai,nbNj (gep only, no delta terms): well-formed
    # and reproduces the finite-diff-validated closed form (16 terms).
    ch = einsums.parse_ir(models.orbital_hessian_ir("neo-ccd(ep)", "electron", "proton"))
    assert einsums.target_shape(ch, "H") == (4, ["v", "V", "o", "O"])
    Hc = np.zeros((nv, nV, no, nO))
    for st in ch:
        for op in st["operands"]:
            if op["name"].split('["')[0] in ("gep", "D2_ep"):
                assert ["p" if c in "OV" else "e" for c in op["classes"]] in (["e", "p", "e", "p"], ["p", "e", "e", "p"])
        arrs = [full[op["name"].split('["')[0]][tuple(SLp[c] for c in op["name"].split('["')[1].rstrip('"]'))]
                for op in st["operands"]]
        subs = ",".join("".join(op["indices"]) for op in st["operands"])
        Hc += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
    vv, VV, oo, OO = SLp["v"], SLp["V"], SLp["o"], SLp["O"]
    cf = np.zeros((nv, no, nV, nO))     # closed form, output (a,i,b,j)
    cf += (ee("pbaQ,jpiQ->aibj", G[:, VV, vv, :], D[OO, :, oo, :]) - ee("pjaQ,bpiQ->aibj", G[:, OO, vv, :], D[VV, :, oo, :])
           + ee("pPab,Ppij->aibj", G[:, :, vv, VV], D[:, :, oo, OO]) - ee("pPaj,Ppib->aibj", G[:, :, vv, OO], D[:, :, oo, VV])
           - ee("ibqQ,jaqQ->aibj", G[oo, VV, :, :], D[OO, vv, :, :]) + ee("ijqQ,baqQ->aibj", G[oo, OO, :, :], D[VV, vv, :, :])
           - ee("iPqb,Paqj->aibj", G[oo, :, :, VV], D[:, vv, :, OO]) + ee("iPqj,Paqb->aibj", G[oo, :, :, OO], D[:, vv, :, VV])
           - ee("pbiQ,jpaQ->aibj", G[:, VV, oo, :], D[OO, :, vv, :]) + ee("pjiQ,bpaQ->aibj", G[:, OO, oo, :], D[VV, :, vv, :])
           - ee("pPib,Ppaj->aibj", G[:, :, oo, VV], D[:, :, vv, OO]) + ee("pPij,Ppab->aibj", G[:, :, oo, OO], D[:, :, vv, VV])
           + ee("abqQ,jiqQ->aibj", G[vv, VV, :, :], D[OO, oo, :, :]) - ee("ajqQ,biqQ->aibj", G[vv, OO, :, :], D[VV, oo, :, :])
           + ee("aPqb,Piqj->aibj", G[vv, :, :, VV], D[:, oo, :, OO]) - ee("aPqj,Piqb->aibj", G[vv, :, :, OO], D[:, oo, :, VV]))
    assert np.max(np.abs(Hc - cf.transpose(0, 2, 1, 3))) < 1e-10

    # unsupported paths raise cleanly
    for bad, exc in ((lambda: models.orbital_gradient_ir("ccsd", "muon"), ValueError),
                     (lambda: models.orbital_gradient_ir("ccsd", "proton"), ValueError),
                     (lambda: models.orbital_hessian_ir("neo-ccd(ep)", "proton", "electron"), NotImplementedError),
                     (lambda: models.orbital_hessian_ir("ccsd", "electron", "proton"), ValueError)):
        try:
            bad(); assert False, "expected an error"
        except exc:
            pass
    print("test_orbital_gradient_hessian OK")


def test_hessian_ir_matches_orbital_hessian():
    """The AMPLITUDE-contracted (T,Lambda) orbital Hessian must equal the fixed-RDM one
    elementwise: they are two contraction paths to the same number (the RDMs are built from
    the same t/Lambda). The fixed-RDM route is finite-difference-verified, so this pins the
    T,Lambda route exactly, with no finite differences needed.

    Why the T,Lambda route exists: the fixed-RDM Hessian/diag/sigma contract D2[vvvv], a
    WAVEFUNCTION quantity that density fitting cannot factorize (DF factorizes integrals).
    At v=400 spin-orbitals that block is ~191 GB and the route is unavailable. This one never
    forms D2 at all, so it is DF-native -- asserted below: B factors, zero g[vvvv], zero
    D2[vvvv]. Its H is (v,o,v,o) = o^2 v^2 (t2-sized), from which a consumer gets the Hessian
    diagonal and sigma = H.kappa directly.

    NB the double commutator is expanded into operator PRODUCTS fed to add_st_operator, NOT
    built with pq_helper's add_double_commutator, whose two-body piece is wrong."""
    import numpy as np

    name = "neo-ccd(ep)"
    ref = models.rdm_energy_reference(name, seed=17)
    d = ref["dims"]; no, nv, nO, nV = d["o"], d["v"], d["O"], d["V"]
    ne, npr = no + nv, nO + nV
    SL = {"o": slice(0, no), "v": slice(no, ne), "O": slice(0, nO), "V": slice(nO, npr)}
    D = {"o": no, "v": nv, "O": nO, "V": nV}

    def assemble(base, shape):
        A = np.zeros(shape)
        for (b, blk), arr in ref["rdm"].items():
            if b == base:
                A[tuple(SL[c] for c in blk)] = arr
        return A
    RDM = {"h": ref["h"], "g": ref["g"], "hp": ref["hp"], "gep": ref["gep"],
           "D1": assemble("D1", (ne, ne)), "D2": assemble("D2", (ne,) * 4),
           "D1_n": assemble("D1_n", (npr, npr)),
           "D2_ep": assemble("D2_ep", (npr, ne, ne, npr)), "Id": np.eye(max(ne, npr))}

    def ev_rdm(ir):
        by = {}
        for st in ir:
            by.setdefault(st["target"]["name"], []).append(st)
        (_, sts), = by.items()
        acc = np.zeros(tuple(SL[c].stop - SL[c].start
                             for c in sts[0]["target"]["classes"]))
        for st in sts:
            arrs = [RDM[o["name"].split('["')[0]][
                        tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                    for o in st["operands"]]
            subs = ",".join("".join(o["indices"]) for o in st["operands"])
            acc += st["coeff"] * np.einsum(
                f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
        return acc

    def ev_amp(ir):
        # base names alone are ambiguous: f(ov) is the electron Fock, f(OV) the proton one
        def named(base, classes):
            proton = all(c in "OV" for c in classes)
            if base == "f":          return ref["fp"] if proton else ref["f"]
            if base in ("eri", "v"): return ref["eri"]
            if base in ("g", "gep"): return ref["gep"]
            return None
        store = {}
        def val(o):
            nm = o["name"]
            if nm in store: return store[nm]
            if nm.startswith("Id["): return np.eye(D[o["classes"][0]])
            arr = named(nm.split("[")[0], o["classes"])
            if arr is not None:
                return arr[tuple(SL[c] for c in o["classes"])]
            return ref["amps"][nm]
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            c = s["coeff"] * np.einsum(subs + "->" + out,
                                       *[val(o) for o in s["operands"]], optimize=True)
            t = s["target"]["name"]
            store[t] = c.copy() if s["is_assignment"] else store[t] + c
        return store["H"]

    def no_repeated_target(ir, tag):
        # A repeated index on a TARGET is not expressible: T[i,j,I,I] = ... defines T on
        # its nuclear diagonal only, and a consumer that allocates from the index classes
        # writes the diagonal and silently leaves the rest zero. (Found downstream: a
        # non-injective nuclear label->subscript map -- case-folding "ni"/"nI" onto one 'I'
        # -- collapsed two distinct nuclear indices; fixed by the per-statement subscript
        # pool in Line::assign_subscripts, with an emission-side assert in ir_emit.) Free to
        # check here -- the IR is already parsed -- and this is the exact opt6 build where
        # the bug lived (tmps_["20_ooOO"]).
        for st in ir:
            idx = st["target"]["indices"]
            assert len(idx) == len(set(idx)), (
                f"{tag}: repeated target index in {st['target']['name']}{idx}")

    for rs, cs in (("electron", "electron"), ("proton", "proton"), ("electron", "proton")):
        # Emitted target index orders DIFFER and must be read from the IR:
        #   same-species: both routes emit [a,b,i,j]
        #   cross:  orbital_hessian_ir -> [a,nb,i,nj]   hessian_ir -> [a,i,nb,nj]
        # They are shape-compatible when nv == no, so a wrong transpose mis-compares
        # SILENTLY. Bring both to [a,i,b,j].
        H_rdm = ev_rdm(einsums.parse_ir(
            models.orbital_hessian_ir(name, rs, cs))).transpose(0, 2, 1, 3)
        amp_ir = einsums.parse_ir(models.hessian_ir(name, rs, cs, df=False, opt_level=0))
        no_repeated_target(amp_ir, f"hessian_ir[{rs},{cs}] opt0 nodf")
        H_amp = ev_amp(amp_ir)
        if rs == cs:
            H_amp = H_amp.transpose(0, 2, 1, 3)
        err = float(np.max(np.abs(H_amp - H_rdm))) / max(float(np.max(np.abs(H_rdm))), 1e-30)
        assert err < 1e-9, (rs, cs, err)

    # DF-native and free of every v^4 object -- the whole point of this route
    ir = einsums.parse_ir(models.hessian_ir(name, "electron", "electron"))   # df=True
    # opt6 df electron-electron: the exact statement (tmps_["20_ooOO"]) that collided.
    no_repeated_target(ir, "hessian_ir[electron,electron] opt6 df")
    names = [o["name"] for s in ir for o in s["operands"]]
    assert not any(n == 'g["vvvv"]' for n in names), "T,Lambda Hessian must not touch g[vvvv]"
    assert not any(n.split("[")[0] in ("D1", "D2", "D2_ep", "D1_n") for n in names), \
        "T,Lambda Hessian must not contract any RDM (D2[vvvv] is the v^4 wall)"
    assert any(n.split("[")[0] == "B" for n in names), "df=True must give B factors"
    print("test_hessian_ir_matches_orbital_hessian OK")


def test_gradient_ir_matches_orbital_gradient():
    """The two orbital-gradient routes must give the SAME gradient:
      * orbital_gradient_ir : fixed-RDM  <[H, E-]> contracted with D1/D2/D1_n/D2_ep
                              (FD-verified by test_orbital_gradient_finite_difference)
      * gradient_ir         : AMPLITUDE form <(1+L) e^-T [H_N, E-] e^T>, contracted with
                              t/Lambda directly -- no RDMs materialised
    H_N = H - E_ref and E_ref is a number, so [H_N, E-] == [H, E-]: same operator, and on
    consistent inputs the two must agree exactly. They do, for ELECTRONIC models.

    Regression: _optimized used to blanket-apply remove_gep_reference_traces() to the
    gradient too. Term-dropping does not commute with taking a commutator (removing
    trace-carrying terms FROM <[H,E-]> is not <[H-T,E-]>), so the NEO gradient came out
    wrong (electron rel 0.41, proton rel 0.73) while the electron-only models were
    unaffected -- a consumer saw a gradient failing its FD check that flipped with gep's
    charge sign. gradient_graph now passes gep_traces=False; both routes then agree to
    ~5e-16 for BOTH species. NB the ST cross-species commutator itself is FINE."""
    import numpy as np

    for name in ("ccsd", "neo-ccd(ep)"):
        ref = models.rdm_energy_reference(name, seed=17)
        d = ref["dims"]; no, nv, nO, nV = d["o"], d["v"], d["O"], d["V"]
        ne, npr = no + nv, nO + nV
        SL = {"o": slice(0, no), "v": slice(no, ne),
              "O": slice(0, nO), "V": slice(nO, npr)}
        D = {"o": no, "v": nv, "O": nO, "V": nV}
        is_neo = "gep" in ref

        def assemble(base, shape):
            A = np.zeros(shape)
            for (b, blk), arr in ref["rdm"].items():
                if b == base:
                    A[tuple(SL[c] for c in blk)] = arr
            return A
        FULL = {"h": ref["h"], "g": ref["g"], "D1": assemble("D1", (ne, ne)),
                "D2": assemble("D2", (ne,) * 4), "Id": np.eye(max(ne, npr))}
        if is_neo:
            FULL.update({"hp": ref["hp"], "gep": ref["gep"],
                         "D1_n": assemble("D1_n", (npr, npr)),
                         "D2_ep": assemble("D2_ep", (npr, ne, ne, npr))})

        # The emitted IR names tensors by BASE + block, and the base alone is ambiguous:
        # f(ov) is the electron Fock while f(OV) is the PROTON Fock (fp) -- same name,
        # distinguished only by the index classes. eri(..) is the antisymmetrized electron
        # ERI; g(..) is the mixed e-p gep. Dispatch on the classes, not the name.
        def named(base, classes):
            proton = all(c in "OV" for c in classes)
            if base == "f":   return ref["fp"] if proton else ref["f"]
            if base in ("eri", "v"): return ref["eri"]
            if base in ("g", "gep"): return ref["gep"]
            return None

        for species in (("electron",) if not is_neo else ("electron", "proton")):
            (_, G_rdm), = _interp_blocks(
                einsums.parse_ir(models.orbital_gradient_ir(name, species)), FULL, SL).items()

            store = {}
            def val(o):
                nm = o["name"]
                if nm in store: return store[nm]
                if nm.startswith("Id["): return np.eye(D[o["classes"][0]])
                arr = named(nm.split("[")[0], o["classes"])
                if arr is not None:
                    return arr[tuple(SL[c] for c in o["classes"])]
                return ref["amps"][nm]
            for s in einsums.parse_ir(
                    models.gradient_ir(name, species, df=False, opt_level=0)):
                subs = ",".join("".join(o["indices"]) for o in s["operands"])
                out = "".join(s["target"]["indices"])
                c = s["coeff"] * np.einsum(subs + "->" + out,
                                           *[val(o) for o in s["operands"]], optimize=True)
                t = s["target"]["name"]
                store[t] = c.copy() if s["is_assignment"] else store[t] + c
            G_amp = store["R"]

            err = float(np.max(np.abs(G_amp - G_rdm))) / max(float(np.max(np.abs(G_rdm))), 1e-30)
            assert err < 1e-10, (name, species, err, G_amp, G_rdm)
    print("test_gradient_ir_matches_orbital_gradient OK")


def test_orbital_gradient_finite_difference():
    """The orbital GRADIENT, HESSIAN, its DIAGONAL and the SIGMA product must be the exact
    first/second derivatives of the fixed-RDM energy that energy_from_rdm_ir traces -- as an
    algebraic identity in the integrals, hence for BOTH signs of gep (a consumer chooses
    gep's charge sign). Verified against finite differences with the RDMs held FIXED and the
    integrals rotated by exp(kappa).

    Run in TWO regimes, and the second one is not optional:

      * ONE quantum proton  -- the physical case for a single-proton NEO model.
      * TWO quantum protons -- the ONLY regime in which the proton 2-RDM D2_n, and with it
        every proton-proton two-body (vp) term, is nonzero. A 2-RDM needs two particles, so
        at one proton the vp terms vanish from the energy AND from all of its derivatives:
        analytic and FD agree perfectly whether or not vp is present. That blindness is
        exactly how vp went missing from the OO quantities (reported by neocc) -- the bug
        and the fix are BOTH unverifiable without this arm. Do not reduce this to one proton.

    History: this test is what caught (a) the electron two-body gradient contracting D2 in
    pq_helper's slot order instead of the consumer's -- an exact sign flip, since D2 is
    antisymmetric in those slots -- and (b) the Hessian coming from pq_helper's DOUBLE
    commutator, whose two-body piece is wrong and which was emitted unsymmetrized. The
    block-structure tests below cannot see either: they only check block-sum ==
    full-contraction, against a reference built from the same source the emitter used."""
    import numpy as np

    def expm(A):                                     # A is tiny & antisymmetric
        R = np.eye(A.shape[0]); T = np.eye(A.shape[0])
        for k in range(1, 18):
            T = T @ A / k
            R = R + T
        return R

    for name, nO in (("neo-ccd(ep)", 1), ("neo-ccsd", 2)):
        # distinct extents: a mis-ordered index becomes a SHAPE error rather than a
        # silent mis-compare (see the note on rdm_energy_reference)
        ref = models.rdm_energy_reference(name, seed=17, no=2, nv=3, nO=nO, nV=4)
        d = ref["dims"]; no, nv, nV = d["o"], d["v"], d["V"]
        ne, npr = no + nv, nO + nV
        SL = {"o": slice(0, no), "v": slice(no, ne),
              "O": slice(0, nO), "V": slice(nO, npr)}
        has_vp = "vp" in ref

        def assemble(base, shape):
            A = np.zeros(shape)
            for (b, blk), arr in ref["rdm"].items():
                if b == base:
                    A[tuple(SL[c] for c in blk)] = arr
            return A

        D1 = assemble("D1", (ne, ne)); D2 = assemble("D2", (ne,) * 4)
        D1n = assemble("D1_n", (npr, npr)); D2ep = assemble("D2_ep", (npr, ne, ne, npr))
        D2n = assemble("D2_n", (npr,) * 4)
        h0, g0, hp0 = ref["h"], ref["g"], ref["hp"]
        vp0 = ref["vp"] if has_vp else np.zeros((npr,) * 4)

        # the vp terms are only exercised if the proton 2-RDM is actually nonzero
        if has_vp:
            assert np.max(np.abs(D2n)) > 1e-6, (name, "D2_n is zero -- vp would be untested")

        def energy(h, g, hp, gep, vp):               # the energy_from_rdm contract
            E = (np.einsum("pq,pq->", h, D1)
                 + 0.5 * np.einsum("abcd,abdc->", g, D2)
                 + np.einsum("PQ,PQ->", hp, D1n)
                 + np.einsum("ePfQ,PefQ->", gep, D2ep))
            if has_vp:
                E = E + 0.5 * np.einsum("ABCD,ABDC->", vp, D2n)
            return float(E)

        # the assembled trace must reproduce the library's own energy (pins the pairings)
        assert abs(energy(h0, g0, hp0, ref["gep"], vp0) - ref["E_rdm"]) < 1e-9, name

        def rot(ke, kp, gep):
            Ue, Up = expm(ke), expm(kp)
            return (Ue.T @ h0 @ Ue,
                    np.einsum("pqrs,pA,qB,rC,sD->ABCD", g0, Ue, Ue, Ue, Ue),
                    Up.T @ hp0 @ Up,
                    np.einsum("ePfQ,eA,fB,PC,QD->ACBD", gep, Ue, Ue, Up, Up),
                    np.einsum("PQRS,PA,QB,RC,SD->ABCD", vp0, Up, Up, Up, Up))

        def kmat(sp, a, i, t):
            n, nocc = (npr, nO) if sp == "p" else (ne, no)
            A = np.zeros((n, n))
            A[nocc + a, i] += t; A[i, nocc + a] -= t
            return A

        ZE, ZP = np.zeros((ne, ne)), np.zeros((npr, npr))
        eps = 1e-4

        def fd_grad(sp, gep):
            nvv, noo = (nV, nO) if sp == "p" else (nv, no)
            G = np.zeros((nvv, noo))
            for a in range(nvv):
                for i in range(noo):
                    def E(t):
                        return energy(*rot(kmat("e", a, i, t) if sp == "e" else ZE,
                                           kmat("p", a, i, t) if sp == "p" else ZP, gep))
                    G[a, i] = (E(eps) - E(-eps)) / (2 * eps)
            return G

        def fd_hess(rs, cs, gep):
            rnv, rno = (nV, nO) if rs == "p" else (nv, no)
            cnv, cno = (nV, nO) if cs == "p" else (nv, no)
            H = np.zeros((rnv, rno, cnv, cno))
            for a in range(rnv):
                for i in range(rno):
                    for b in range(cnv):
                        for j in range(cno):
                            def E2(ta, tb):
                                ke = (kmat("e", a, i, ta) if rs == "e" else ZE) + \
                                     (kmat("e", b, j, tb) if cs == "e" else ZE)
                                kp = (kmat("p", a, i, ta) if rs == "p" else ZP) + \
                                     (kmat("p", b, j, tb) if cs == "p" else ZP)
                                return energy(*rot(ke, kp, gep))
                            H[a, i, b, j] = (E2(eps, eps) - E2(eps, -eps)
                                             - E2(-eps, eps) + E2(-eps, -eps)) / (4 * eps * eps)
            return H

        def ev(ir, gep, extra=None):
            F = {"h": h0, "g": g0, "hp": hp0, "gep": gep, "vp": vp0, "D1": D1, "D2": D2,
                 "D1_n": D1n, "D2_ep": D2ep, "D2_n": D2n, "Id": np.eye(max(ne, npr))}
            F.update(extra or {})
            by = {}
            for st in ir:
                by.setdefault(st["target"]["name"], []).append(st)
            (_, sts), = by.items()
            acc = np.zeros(tuple(SL[c].stop - SL[c].start
                                 for c in sts[0]["target"]["classes"]))
            for st in sts:
                arrs = [F[o["name"].split('["')[0]][
                            tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                        if '["' in o["name"] else F[o["name"]] for o in st["operands"]]
                subs = ",".join("".join(o["indices"]) for o in st["operands"])
                acc += st["coeff"] * np.einsum(
                    f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
            return acc

        def close(got, fd, *tag):
            err = float(np.max(np.abs(got - fd))) / max(float(np.max(np.abs(fd))), 1e-30)
            assert err < 1e-5, (name, *tag, err)

        # ---- GRADIENT: both species, and BOTH SIGNS of gep (algebraic in the integrals)
        for sign in (+1.0, -1.0):
            gep = sign * ref["gep"]
            for sp, s in (("electron", "e"), ("proton", "p")):
                close(ev(einsums.parse_ir(models.orbital_gradient_ir(name, sp)), gep),
                      fd_grad(s, gep), "gradient", sp, "gep sign", sign)

        # ---- HESSIAN (all three blocks), its DIAGONAL, and SIGMA
        gep = ref["gep"]
        H_ee, H_pp, H_ep = (fd_hess("e", "e", gep), fd_hess("p", "p", gep),
                            fd_hess("e", "p", gep))
        for tag, rs, cs, FD in (("ee", "electron", "electron", H_ee),
                                ("pp", "proton", "proton", H_pp),
                                ("ep", "electron", "proton", H_ep)):
            M = ev(einsums.parse_ir(models.orbital_hessian_ir(
                name, row_species=rs, col_species=cs)), gep).transpose(0, 2, 1, 3)
            close(M, FD, "hessian", tag)
            if rs == cs:      # the true fixed-RDM Hessian is exactly symmetric
                assert np.max(np.abs(M - M.transpose(2, 3, 0, 1))) / \
                       max(float(np.max(np.abs(M))), 1e-30) < 1e-10, (name, "asym", tag)

        for sp, FD in (("electron", H_ee), ("proton", H_pp)):
            close(ev(einsums.parse_ir(models.orbital_hessian_diag_ir(name, sp)), gep),
                  np.einsum("aiai->ai", FD), "hessian diag", sp)

        rng2 = np.random.default_rng(9)
        tr_e = rng2.standard_normal((nv, no)); tr_p = rng2.standard_normal((nV, nO))
        Ke = np.zeros((ne, ne)); Ke[no:, :no] = tr_e
        Kp = np.zeros((npr, npr)); Kp[nO:, :nO] = tr_p
        # sigma carries the CROSS coupling: s^e = H^ee.k^e + H^ep.k^p (and transposed for p)
        for sp, FD in (("electron", np.einsum("aibj,bj->ai", H_ee, tr_e)
                                    + np.einsum("aiBJ,BJ->ai", H_ep, tr_p)),
                       ("proton",   np.einsum("AIBJ,BJ->AI", H_pp, tr_p)
                                    + np.einsum("aiAI,ai->AI", H_ep, tr_e))):
            close(ev(einsums.parse_ir(models.orbital_sigma_ir(name, sp)), gep,
                     {"kappa": Ke, "kappa_n": Kp}), FD, "sigma", sp)

    print("test_orbital_gradient_finite_difference OK")


def test_orbital_hessian_diag():
    import numpy as np
    assert "orbital_hessian_diag_ir" in models.__all__
    diag = einsums.parse_ir(models.orbital_hessian_diag_ir("ccsd", "electron"))
    assert einsums.target_shape(diag, "hdiag") == (2, ["v", "o"])        # rank-2 diagonal
    assert not any(l in ("b", "j")                                       # no leftover column labels
                   for st in diag for v in [st["target"], *st["operands"]] for l in v["indices"])

    # numeric (self-contained): the relabel-diagonal == diag of the block Hessian
    no, nv = 3, 4
    nmo = no + nv
    rng = np.random.default_rng(0)
    full = {n: rng.standard_normal((nmo,) * r) for n, r in (("h", 2), ("g", 4), ("D1", 2), ("D2", 4))}
    full["Id"] = np.eye(nmo)
    H4 = _interp_block(einsums.parse_ir(models.orbital_hessian_ir("ccsd")), full, no, nv)["H"]
    h = _interp_block(diag, full, no, nv)["hdiag"]
    assert float(np.max(np.abs(h - np.einsum("aaii->ai", H4)))) < 1e-10
    print("test_orbital_hessian_diag OK")


def test_orbital_sigma():
    import numpy as np
    assert "orbital_sigma_ir" in models.__all__
    no, nv, nO, nV = 2, 3, 1, 2
    ne, npp = no + nv, nO + nV
    SL = {"o": slice(0, no), "v": slice(no, ne), "O": slice(0, nO), "V": slice(nO, npp)}
    rng = np.random.default_rng(3)
    full = {n: rng.standard_normal((ne,) * r) for n, r in (("h", 2), ("g", 4), ("D1", 2), ("D2", 4))}
    full.update({"gep": rng.standard_normal((ne, npp, ne, npp)), "D2_ep": rng.standard_normal((npp, ne, ne, npp)),
                 "hp": rng.standard_normal((npp, npp)), "D1_n": rng.standard_normal((npp, npp)),
                 "kappa": rng.standard_normal((ne, ne)), "kappa_n": rng.standard_normal((npp, npp))})

    def interp(ir, shape):
        out = np.zeros(shape)
        for st in ir:
            arrs = []
            for op in st["operands"]:
                base, blk = op["name"].split('["'); blk = blk.rstrip('"]')
                arrs.append(np.eye(ne if blk.islower() else npp)[tuple(SL[c] for c in blk)] if base == "Id"
                            else full[base][tuple(SL[c] for c in blk)])
            subs = ",".join("".join(op["indices"]) for op in st["operands"])
            out = out + st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
        return out

    kap, kapn = full["kappa"][SL["v"], SL["o"]], full["kappa_n"][SL["V"], SL["O"]]
    Hee = interp(einsums.parse_ir(models.orbital_hessian_ir("neo-ccd(ep)", "electron")), (nv, nv, no, no))
    Hep = interp(einsums.parse_ir(models.orbital_hessian_ir("neo-ccd(ep)", "electron", "proton")), (nv, nV, no, nO))
    Hpp = interp(einsums.parse_ir(models.orbital_hessian_ir("neo-ccd(ep)", "proton")), (nV, nV, nO, nO))
    se_ir = einsums.parse_ir(models.orbital_sigma_ir("neo-ccd(ep)", "electron"))
    sp_ir = einsums.parse_ir(models.orbital_sigma_ir("neo-ccd(ep)", "proton"))
    assert einsums.target_shape(se_ir, "sigma") == (2, ["v", "o"])
    assert einsums.target_shape(sp_ir, "sigma") == (2, ["V", "O"])
    # sigma^e = H^ee.kappa^e + H^ep.kappa^p ;  sigma^p = H^pp.kappa^p + (H^ep)^T.kappa^e
    assert np.max(np.abs(interp(se_ir, (nv, no))
                         - np.einsum("abij,bj->ai", Hee, kap) - np.einsum("abij,bj->ai", Hep, kapn))) < 1e-10
    assert np.max(np.abs(interp(sp_ir, (nV, nO))
                         - np.einsum("abij,bj->ai", Hpp, kapn) - np.einsum("abij,ai->bj", Hep, kap))) < 1e-10
    # electronic (non-NEO): sigma = H^ee.kappa, no cross term
    assert einsums.target_shape(einsums.parse_ir(models.orbital_sigma_ir("ccsd")), "sigma") == (2, ["v", "o"])
    try:
        models.orbital_sigma_ir("ccsd", "proton"); assert False
    except ValueError:
        pass
    print("test_orbital_sigma OK")


def test_orbital_gradient_active_space():
    import re
    import numpy as np
    # default is unchanged (single active-active block, bare target)
    assert einsums.target_shape(einsums.parse_ir(models.orbital_gradient_ir("ccsd")), "grad") == (2, ["v", "o"])

    nc, no, nv, nx = 1, 2, 2, 4
    nmo = nc + no + nv + nx
    SL = {"c": slice(0, nc), "o": slice(nc, nc + no), "v": slice(nc + no, nc + no + nv), "x": slice(nc + no + nv, nmo)}
    asl = slice(nc, nc + no + nv); na = no + nv
    rng = np.random.default_rng(0)
    h = rng.standard_normal((nmo, nmo)); h = h + h.T
    grw = rng.standard_normal((nmo,) * 4); g = grw + grw.transpose(1, 0, 3, 2)
    D1 = np.zeros((nmo, nmo)); D1[asl, asl] = rng.standard_normal((na, na))       # active-only
    d2 = np.zeros((nmo,) * 4); d2[asl, asl, asl, asl] = rng.standard_normal((na,) * 4)
    D2 = d2 - d2.transpose(1, 0, 2, 3); D2 = D2 - D2.transpose(0, 1, 3, 2)
    FULL = {"h": h, "g": g, "D1": D1, "D2": D2}

    # reference generalized-Fock gradient (full sums, active-only RDMs) from the
    # FD-verified term source -- the first rotation-derivative of the fixed-RDM energy.
    # It used to be rebuilt here from pq_helper's commutator, i.e. the same source the
    # emitter used, so this test could not see a wrong emission. The physics is guarded by
    # test_orbital_gradient_finite_difference; this only checks block-sum == full-contraction.
    Gref = _interp_terms(models._rot_deriv(models._energy_bare_terms("ccsd"), "a", "i", "e"),
                         ["a", "i"], (nmo, nmo), FULL)

    ir = einsums.parse_ir(models.orbital_gradient_ir("ccsd", "electron", rotation_classes=("c", "o", "v", "x")))
    # every non-redundant block except the all-inactive x-c (which vanishes on the active RDM)
    assert {st["target"]["name"] for st in ir} == {'grad["vo"]', 'grad["xo"]', 'grad["xv"]', 'grad["oc"]', 'grad["vc"]'}
    # no inactive index ever lands in an RDM -> one free inactive-virtual index, J/K-shaped
    assert not any(o["name"].split('["')[0] in ("D1", "D2") and any(c in "xcXC" for c in o["classes"])
                   for st in ir for o in st["operands"])
    by = {}
    for st in ir:
        by.setdefault(st["target"]["name"], []).append(st)
    for name, sts in by.items():
        rc, cc = re.search(r'\["(\w)(\w)"\]', name).groups()
        out = np.zeros((SL[rc].stop - SL[rc].start, SL[cc].stop - SL[cc].start))
        for st in sts:
            arrs = [FULL[o["name"].split('["')[0]][tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                    for o in st["operands"]]
            subs = ",".join("".join(o["indices"]) for o in st["operands"])
            out += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
        assert np.max(np.abs(out - Gref[SL[rc], SL[cc]])) < 1e-10, name
    print("test_orbital_gradient_active_space OK")


def _active_space_H4(FULL, nmo, name="ccsd"):
    """Reference full electron Hessian H4[a,b,i,j] over nmo with active-only RDMs.

    Built from models._electron_hessian_terms -- the fixed-RDM second rotation-derivative
    of the energy, which is finite-difference-verified by
    test_orbital_gradient_finite_difference. It replaces pq_helper's DOUBLE commutator,
    whose two-body piece is wrong and which emitted the unsymmetrized <[[H,A],B]>. (This
    reference used to be built from that same broken commutator, so these active-space
    tests were validating the emitter against its own source -- which is exactly why the
    defect survived. The physics is now guarded by the FD test; these check only that the
    block decomposition reproduces the full contraction.)"""
    import numpy as np
    H4 = np.zeros((nmo,) * 4)
    for t in models._electron_hessian_terms(name):
        coeff, ts = models._parse_rdm_term(t)
        arrs = [FULL["Id"] if nm == "d" else FULL[nm] for nm, _ in ts]
        H4 += coeff * np.einsum(",".join("".join(i) for _, i in ts) + "->abij", *arrs, optimize=True)
    return H4


def _active_space_tensors(seed):
    import numpy as np
    nc, no, nv, nx = 1, 2, 2, 3
    nmo = nc + no + nv + nx
    SL = {"c": slice(0, nc), "o": slice(nc, nc + no), "v": slice(nc + no, nc + no + nv), "x": slice(nc + no + nv, nmo)}
    asl = slice(nc, nc + no + nv); na = no + nv
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((nmo, nmo)); h = h + h.T
    grw = rng.standard_normal((nmo,) * 4); g = grw + grw.transpose(1, 0, 3, 2)
    D1 = np.zeros((nmo, nmo)); D1[asl, asl] = rng.standard_normal((na, na))
    d2 = np.zeros((nmo,) * 4); d2[asl, asl, asl, asl] = rng.standard_normal((na,) * 4)
    D2 = d2 - d2.transpose(1, 0, 2, 3); D2 = D2 - D2.transpose(0, 1, 3, 2)
    FULL = {"h": h, "g": g, "D1": D1, "D2": D2, "Id": np.eye(nmo)}
    RC = ("c", "o", "v", "x")
    blocks = [(hi, lo) for hi in RC for lo in RC if models._CLASS_LEVEL[hi] > models._CLASS_LEVEL[lo]]
    return nmo, SL, FULL, RC, blocks


def _interp_blocks(ir, FULL, SL):
    import numpy as np
    by = {}
    for st in ir:
        by.setdefault(st["target"]["name"], []).append(st)
    res = {}
    for name, sts in by.items():
        shp = tuple(SL[c].stop - SL[c].start for c in sts[0]["target"]["classes"])
        out = np.zeros(shp)
        for st in sts:
            arrs = [FULL[o["name"].split('["')[0]][tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                    for o in st["operands"]]
            subs = ",".join("".join(o["indices"]) for o in st["operands"])
            out += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
        res[name] = out
    return res


def test_orbital_diag_active_space():
    import numpy as np
    assert einsums.target_shape(einsums.parse_ir(models.orbital_hessian_diag_ir("ccsd")), "hdiag") == (2, ["v", "o"])
    nmo, SL, FULL, RC, blocks = _active_space_tensors(2)
    H4 = _active_space_H4(FULL, nmo)
    got = _interp_blocks(einsums.parse_ir(models.orbital_hessian_diag_ir("ccsd", "electron", rotation_classes=RC)), FULL, SL)
    for hi, lo in blocks:
        name = f'hdiag["{hi}{lo}"]'
        ref = np.einsum("aaii->ai", H4[SL[hi], SL[hi], SL[lo], SL[lo]])
        if name in got:
            assert np.max(np.abs(got[name] - ref)) < 1e-10, name
        else:                                              # not emitted => zero active-RDM contribution
            assert np.max(np.abs(ref)) < 1e-12, name
    print("test_orbital_diag_active_space OK")


def test_orbital_sigma_active_space():
    import re
    import numpy as np
    import pdaggerq
    # default unchanged
    assert einsums.target_shape(einsums.parse_ir(models.orbital_sigma_ir("ccsd")), "sigma") == (2, ["v", "o"])

    nc, no, nv, nx = 1, 2, 2, 3
    nmo = nc + no + nv + nx
    SL = {"c": slice(0, nc), "o": slice(nc, nc + no), "v": slice(nc + no, nc + no + nv), "x": slice(nc + no + nv, nmo)}
    asl = slice(nc, nc + no + nv); na = no + nv
    rng = np.random.default_rng(1)
    h = rng.standard_normal((nmo, nmo)); h = h + h.T
    grw = rng.standard_normal((nmo,) * 4); g = grw + grw.transpose(1, 0, 3, 2)
    D1 = np.zeros((nmo, nmo)); D1[asl, asl] = rng.standard_normal((na, na))
    d2 = np.zeros((nmo,) * 4); d2[asl, asl, asl, asl] = rng.standard_normal((na,) * 4)
    D2 = d2 - d2.transpose(1, 0, 2, 3); D2 = D2 - D2.transpose(0, 1, 3, 2)
    kap = rng.standard_normal((nmo, nmo))
    FULL = {"h": h, "g": g, "D1": D1, "D2": D2, "Id": np.eye(nmo), "kappa": kap}
    RC = ("c", "o", "v", "x")
    blocks = [(hi, lo) for hi in RC for lo in RC if models._CLASS_LEVEL[hi] > models._CLASS_LEVEL[lo]]

    # reference full Hessian H4 over nmo (active-only RDMs), from the FD-verified term
    # source -- NOT pq_helper's double commutator, which this used to rebuild inline
    # (i.e. it validated the emitter against its own broken source). See _active_space_H4.
    H4 = _active_space_H4(FULL, nmo)

    ir = einsums.parse_ir(models.orbital_sigma_ir("ccsd", "electron", rotation_classes=RC))
    assert all(sum(c in "xX" for c in st["target"]["classes"]) <= 1 for st in ir)   # one free x per term
    by = {}
    for st in ir:
        by.setdefault(st["target"]["name"], []).append(st)
    for rhi, rlo in blocks:
        name = f'sigma["{rhi}{rlo}"]'
        if name not in by:
            continue
        out = np.zeros((SL[rhi].stop - SL[rhi].start, SL[rlo].stop - SL[rlo].start))
        for st in by[name]:
            arrs = [FULL[o["name"].split('["')[0]][tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                    for o in st["operands"]]
            subs = ",".join("".join(o["indices"]) for o in st["operands"])
            out += st["coeff"] * np.einsum(f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
        ref = sum(np.einsum("abij,bj->ai", H4[SL[rhi], SL[chi], SL[rlo], SL[clo]], kap[SL[chi], SL[clo]])
                  for chi, clo in blocks)
        assert np.max(np.abs(out - ref)) < 1e-10, name
    print("test_orbital_sigma_active_space OK")


def _interp_terms(terms, out_letters, shape, F):
    import numpy as np
    G = np.zeros(shape)
    for term in terms:
        coeff, tensors = models._parse_rdm_term(term)
        lab = {}
        for _, idx in tensors:
            for l in idx:
                lab.setdefault(l, chr(ord("A") + len(lab)))
        subs = ",".join("".join(lab[l] for l in idx) for _, idx in tensors)
        G += coeff * np.einsum(f"{subs}->{''.join(lab[l] for l in out_letters)}",
                               *[F[n] for n, _ in tensors], optimize=True)
    return G


def test_orbital_proton_gradient_active_space():
    import re
    import numpy as np
    ne = 4                                                 # electron all active (o=0:2, v=2:4)
    nC, nO, nV, nX = 1, 1, 2, 2
    npp = nC + nO + nV + nX
    SL = {"o": slice(0, 2), "v": slice(2, 4), "C": slice(0, nC), "O": slice(nC, nC + nO),
          "V": slice(nC + nO, nC + nO + nV), "X": slice(nC + nO + nV, npp)}
    pasl = slice(nC, nC + nO + nV)
    rng = np.random.default_rng(4)
    hp = rng.standard_normal((npp, npp)); hp = hp + hp.T
    D1n = np.zeros((npp, npp)); D1n[pasl, pasl] = rng.standard_normal((nO + nV, nO + nV))
    gw = rng.standard_normal((ne, npp, ne, npp)); gep = gw + gw.transpose(2, 3, 0, 1)
    D2 = rng.standard_normal((npp, ne, ne, npp)); D2ep = np.zeros_like(D2); D2ep[pasl, :, :, pasl] = D2[pasl, :, :, pasl]
    # the derived terms use the GENERIC names h/g/D1/D2 (which _block_resolve renames to
    # hp/gep/D1_n/D2_ep -- or vp/D2_n -- from the index species)
    F = {"g": gep, "h": hp, "D1": D1n, "D2": D2ep,
         "gep": gep, "hp": hp, "D1_n": D1n, "D2_ep": D2ep}
    # from the FD-verified source (was the hand-derived _HP_GRAD_TERMS +
    # _GEP_PROTON_GRAD_TERMS, which the rotation-derivative derivation supersedes)
    Gp = _interp_terms(models._rot_deriv(models._energy_bare_terms("neo-ccd(ep)"),
                                         "na", "ni", "p"),
                       ["na", "ni"], (npp, npp), F)
    g = einsums.parse_ir(models.orbital_gradient_ir("neo-ccd(ep)", "proton", rotation_classes=("c", "o", "v", "x")))
    assert {st["target"]["name"] for st in g} == {'grad["VO"]', 'grad["XO"]', 'grad["XV"]', 'grad["OC"]', 'grad["VC"]'}
    assert not any(o["name"].split('["')[0] in ("D1_n", "D2_ep") and any(c in "CX" for c in o["classes"])
                   for st in g for o in st["operands"])
    for name, out in _interp_blocks(g, F, SL).items():
        hi, lo = re.search(r'\["(\w)(\w)"\]', name).groups()
        assert np.max(np.abs(out - Gp[SL[hi], SL[lo]])) < 1e-10, name
    print("test_orbital_proton_gradient_active_space OK")


def test_orbital_cross_sigma_active_space():
    import re
    import numpy as np
    nc, no, nv, nx = 1, 2, 1, 2
    nC, nO, nV, nX = 1, 1, 2, 1
    nme, nmp = nc + no + nv + nx, nC + nO + nV + nX
    SL = {"c": slice(0, nc), "o": slice(nc, nc + no), "v": slice(nc + no, nc + no + nv), "x": slice(nc + no + nv, nme),
          "C": slice(0, nC), "O": slice(nC, nC + nO), "V": slice(nC + nO, nC + nO + nV), "X": slice(nC + nO + nV, nmp)}
    ae, ap = slice(nc, nc + no + nv), slice(nC, nC + nO + nV)
    rng = np.random.default_rng(7)
    gw = rng.standard_normal((nme, nmp, nme, nmp)); gep = gw + gw.transpose(2, 3, 0, 1)
    D2 = rng.standard_normal((nmp, nme, nme, nmp)); D2ep = np.zeros_like(D2); D2ep[ap, ae, ae, ap] = D2[ap, ae, ae, ap]
    kp = rng.standard_normal((nmp, nmp))
    # the derived terms carry the GENERIC tensor names g/D2 (which _block_resolve renames
    # to gep/D2_ep from the index species); the old hand-written lists spelled gep/D2_ep out
    F = {"g": gep, "gep": gep, "D2": D2ep, "D2_ep": D2ep, "kappa_n": kp}
    # reference from the FD-verified term source (was models._GEP_CROSS_HESS_TERMS, the
    # hand-derived list the rotation-derivative derivation supersedes)
    Hep = _interp_terms(models._cross_hessian_terms("neo-ccd(ep)"), ["a", "nb", "i", "nj"],
                        (nme, nmp, nme, nmp), F)
    ir = einsums.parse_ir(models.orbital_sigma_ir("neo-ccd(ep)", "electron", rotation_classes=("c", "o", "v", "x")))
    cross = [st for st in ir if any(o["name"].split('["')[0] == "kappa_n" for o in st["operands"])]
    assert cross and all(sum(c in "xX" for c in st["target"]["classes"]) <= 1 for st in cross)
    pblk = [(hi, lo) for hi in "COVX" for lo in "COVX" if models._CLASS_LEVEL[hi] > models._CLASS_LEVEL[lo]]
    for name, out in _interp_blocks(cross, F, SL).items():
        ehi, elo = re.search(r'\["(\w)(\w)"\]', name).groups()
        ref = sum(np.einsum("abij,bj->ai", Hep[SL[ehi], SL[phi], SL[elo], SL[plo]], kp[SL[phi], SL[plo]])
                  for phi, plo in pblk)
        assert np.max(np.abs(out - ref)) < 1e-10, name
    print("test_orbital_cross_sigma_active_space OK")


def test_neo_gep_normal_ordered():
    import re
    import pdaggerq
    OCC = set("ijklmno")
    # every cluster-amplitude head in pq_helper's naming: electron t<n>, proton t<n>_n
    # (tp<n>), and mixed t<n>_ep (tep<..>). A term is amplitude-free iff it contains none.
    amps = ("t1(", "t2(", "t3(", "t4(",
            "t1_n(", "t2_n(", "t3_n(", "t4_n(",
            "t2_ep(", "t3_ep(", "t4_ep(")

    def has_gep_trace(line):                               # gep integral with a repeated occupied label
        for mm in re.finditer(r"g\(([^)]+)\)", line):
            idx = mm.group(1).split(",")
            if not any(i.startswith("n") for i in idx):
                continue
            for i in idx:
                core = i[1:] if i.startswith("n") else i
                if idx.count(i) >= 2 and core in OCC:
                    return True
        return False

    def residual(name, amp):                               # generated exactly as _optimized does it
        pq = pdaggerq.pq_helper("fermi")
        pq.set_left_operators([[models.PROJECTION[amp]]])
        m = models.model(name)
        for h in m.H:
            pq.add_st_operator(1.0, [h], list(m.T))
        pq.simplify()
        pq.remove_gep_reference_traces()
        return [" ".join(t) for t in pq.strings()]

    # cheap representative amplitudes (the strip is integral-structure-based, not
    # amplitude-rank-specific): electron/proton singles, the mixed double, both hybrids
    checks = {"neo-ccsd": ["t1", "tp1", "tep11"], "neo-ccd(ep)": ["tep11"]}
    for name, to_check in checks.items():
        for amp in to_check:
            terms = residual(name, amp)
            assert not any(has_gep_trace(t) for t in terms), (name, amp)   # no gep self-trace survives
            if amp in ("t1", "tp1"):                       # singles at t=0 reduce to the Fock block only
                af = [t for t in terms if not any(a in t for a in amps)]
                assert len(af) == 1 and "f(" in af[0] and "g(" not in af[0], (name, amp, af)
    # the strip removes only reference traces, never the genuine two-body driver
    assert any("g(a,na,i,ni)" in t for t in residual("neo-ccd(ep)", "tep11"))
    print("test_neo_gep_normal_ordered OK")


def test_opt_level_safe_default():
    """pq_graph opt_level=6 (intermediate *fusion*) was nondeterministic and emitted
    IR that consumers misread (fusion-created constant-scalar vertices as index-less
    operands). Both are fixed on this fork (canonical fusion ordering in fusion.cc +
    constant folding in ir_emit) and generation defaults to full opt6 again. Verify
    (1) the resolver default, (2) the default residual matches opt0, and (3) the
    fusion fix holds on the equation that used to break: opt6 is deterministic and
    correct (the reverse of the old tripwire; if this fails, re-cap to 5)."""
    import re, itertools
    import numpy as np
    from collections import defaultdict

    # (1) resolver: None -> the default (full opt6, fusion fixed) for every model;
    # explicit wins
    for m in ("ccsd", "neo-ccsd", "neo-ccd(ep)", "ccsdt", "ccsdtq",
              "neo-ccsdt(eep)", "neo-ccsdtq(eeep)"):
        assert models._opt_level_for(m, None) == 6, m
    assert models._opt_level_for("ccsd", 5) == 5                # explicit override wins
    assert models._opt_level_for("ccsd", 0) == 0

    # (2) numerical: default (resolved opt5) == opt0; opt6 is the bug the default avoids
    DIM = {"o": 3, "v": 4, "O": 1, "V": 4, "Q": 6}
    VIR, OCC = {"v", "V"}, {"o", "O"}
    is_amp = lambda nm: re.fullmatch(r"t\d+(_n|_ep)?", nm) is not None

    def antisym(a, cl):                        # CC antisymmetry over same-class vir/occ axes
        out = a.copy()
        groups = defaultdict(list)
        for ax, c in enumerate(cl):
            groups[c].append(ax)
        for c, axes in groups.items():
            if len(axes) >= 2 and (c in VIR or c in OCC):
                perms = list(itertools.permutations(range(len(axes))))
                acc = np.zeros_like(out)
                for p in perms:
                    par = sum(1 for i in range(len(p)) for j in range(i + 1, len(p))
                              if p[i] > p[j]) & 1
                    src = list(range(out.ndim))
                    for k, ax in enumerate(axes):
                        src[ax] = axes[p[k]]
                    acc += (-1 if par else 1) * np.transpose(out, src)
                out = acc / len(perms)
        return out

    def interp(ir, inp):                       # evaluate a residual IR statement list
        st = {}
        val = lambda o: st[o["name"]] if o["name"] in st else inp[o["name"]]
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            c = s["coeff"] * np.einsum(subs + "->" + out,
                                       *[val(o) for o in s["operands"]], optimize=True)
            t = s["target"]["name"]
            st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st["R"]

    ir0 = einsums.parse_ir(models.residual_graph("neo-ccsdt(eep)", "tep11",
                                                 opt_level=0).to_strings("ir"))
    produced = {s["target"]["name"] for s in ir0}
    names = {o["name"]: tuple(o["classes"]) for s in ir0 for o in s["operands"]
             if o["name"] not in produced}
    rng = np.random.default_rng(7)
    inp = {}
    for nm, cl in sorted(names.items()):
        a = rng.standard_normal(tuple(DIM[c] for c in cl))
        inp[nm] = antisym(a, cl) if is_amp(nm) else a
    truth = interp(ir0, inp)
    default = interp(einsums.parse_ir(models.residual_ir("neo-ccsdt(eep)", "tep11")), inp)
    assert np.max(np.abs(default - truth)) < 1e-9, np.max(np.abs(default - truth))
    # (3) opt_level 6 must be BYTE-DETERMINISTIC and numerically correct (matches opt0) on
    # the equation that used to break. The pq_graph optimizer's substitution/fusion passes
    # race on the lazy caches of shared Linkage objects (ThreadSanitizer flags consolidate.cc
    # / substitute.cc / linkage.cc), so at >1 thread the *chosen* substitutions -- hence the
    # emitted text -- would vary run to run. models._optimized therefore pins the optimizer to
    # a single thread (nthreads=1), which makes codegen byte-reproducible at any OMP_NUM_THREADS
    # (neocc relies on this for frozen codegen). This assertion exercises that pin: run under
    # OMP_NUM_THREADS>1 (CI does) so >1 thread is *available* and the pin is what forces
    # determinism. If it trips, either the pin was dropped or the race otherwise resurfaced
    # (edeprince3/pdaggerq#114).
    texts, worst = set(), 0.0
    for _ in range(3):
        ir6 = models.residual_ir("neo-ccsdt(eep)", "tep11", opt_level=6)
        texts.add("\n".join(l for l in ir6 if l.strip().startswith("{")))
        worst = max(worst, float(np.max(np.abs(interp(einsums.parse_ir(ir6), inp) - truth))))
    assert len(texts) == 1, (f"opt6 emission nondeterministic ({len(texts)} variants) -- "
                             "codegen single-thread pin lost or race resurfaced (#114)")
    assert worst < 1e-9, f"opt6 numerically wrong again (err {worst})"
    print("test_opt_level_safe_default OK")


def test_rdm_block_ir():
    """rdm_block_ir emits each RDM block with the TARGET named + index-ordered exactly
    as energy_from_rdm_ir / orbital_*_ir consume it. Checks (1) every block the energy
    trace references is reproduced with matching name/classes; (2) single-species blocks
    evaluate to the genuine rdm_graph block (permuted to consumer order); (3) a block the
    model cannot populate returns []."""
    import numpy as np
    assert "rdm_block_ir" in models.__all__

    # (1) structural: cover every RDM block energy_from_rdm_ir(neo-ccsd) references,
    # including the mixed D2_ep (whose consumer layout is (P,E,E',P'), e.g. "OovV").
    refs = {}
    for st in einsums.parse_ir(models.energy_from_rdm_ir("neo-ccsd")):
        for o in st["operands"]:
            base = o["name"].split('["')[0]
            if base in ("D1", "D2", "D1_n", "D2_ep"):
                refs[o["name"]] = (base, o["name"].split('["')[1].rstrip('"]'), o["classes"])
    assert any(b == "D2_ep" for _, (b, _, _) in refs.items())        # mixed blocks present
    for full_name, (base, block, classes) in refs.items():
        ir = einsums.parse_ir(models.rdm_block_ir("neo-ccsd", base, block))
        assert ir, (full_name, "unexpectedly empty")
        tgt = next(s["target"] for s in reversed(ir) if s["target"]["name"] == full_name)
        assert tgt["classes"] == classes, (full_name, tgt["classes"], classes)   # consumer order

    # (2) numeric: single-species blocks == the genuine rdm_graph block, consumer order.
    DIM = {"o": 3, "v": 4, "O": 2, "V": 3}
    rng = np.random.default_rng(11)

    def interp(ir, inp):
        st = {}
        val = lambda o: st[o["name"]] if o["name"] in st else inp[o["name"]]
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            c = s["coeff"] * np.einsum(subs + "->" + out, *[val(o) for o in s["operands"]],
                                       optimize=True)
            t = s["target"]["name"]
            st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st

    # Only PURE-ELECTRON blocks may be cross-checked against rdm_graph: pq_graph collapses a
    # block's internal proton indices onto its open ones, so its D1_n / D2_n / D2_ep are wrong
    # (that is exactly why rdm_block_ir emits every block from pq.strings instead). The nuclear
    # blocks are covered numerically by the full-energy identity in test_energy_from_rdm.
    blocks = [("D1", "oo"), ("D1", "ov"), ("D1", "vv"),
              ("D2", "oovv"), ("D2", "ovvo"), ("D2", "vvoo")]
    # opt_level=0 for the numeric comparison: both calls must lower the SAME graph, and
    # pq_graph's optimized (>=5) intermediate ordering is not guaranteed identical across
    # two independent invocations -- opt0 is deterministic and always correct.
    bir = {(t, b): einsums.parse_ir(models.rdm_block_ir("neo-ccsd", t, b, opt_level=0))
           for t, b in blocks}
    dir_ = {(t, b): einsums.parse_ir(
                models.rdm_ir("neo-ccsd", models._rdm_block_spec(t, b)[0], opt_level=0))
            for t, b in blocks}
    inp = {}
    for ir in list(bir.values()) + list(dir_.values()):
        produced = {s["target"]["name"] for s in ir}
        for s in ir:
            for o in s["operands"]:
                n = o["name"]
                if n in produced or n in inp:
                    continue
                a = rng.standard_normal(tuple(DIM[c] for c in o["classes"]))
                if n in ("t2", "l2") and len(o["classes"]) == 4:     # electron doubles: antisym
                    a = a - a.transpose(1, 0, 2, 3)
                    a = a - a.transpose(0, 1, 3, 2)
                inp[n] = a
    for t, b in blocks:
        op, consumer = models._rdm_block_spec(t, b)
        blk = interp(bir[(t, b)], inp)[f'{t}["{b}"]']
        native = interp(dir_[(t, b)], inp)["D"]
        final = next(s["target"] for s in reversed(dir_[(t, b)]) if s["target"]["name"] == "D")
        # _rdm_block_spec returns pq labels (proton ones carry the nuclear 'n' prefix);
        # rdm_graph's emitted IR strips it.
        perm = [list(final["indices"]).index(L.lstrip("n")) for L, _ in consumer]
        assert np.max(np.abs(blk - np.transpose(native, perm))) < 1e-10, (t, b)

    # (4) numeric end-to-end (mixed e-p): the RDM chain reproduces the e-p energy.
    #     sum_blocks gep.D2_ep  ==  <(1+L) e^-T gep e^T>.  Guards both the D2_ep emission
    #     (pq_graph collapses a mixed density's internal proton indices onto the open ones;
    #     the string emitter bypasses it) and the sign of the gep.D2_ep energy term. The
    #     identity holds at ARBITRARY amplitudes, so random (unconverged) inputs suffice.
    import itertools as _it
    import pdaggerq
    ne, npr = DIM["o"] + DIM["v"], DIM["O"] + DIM["V"]
    erg = {"o": list(range(DIM["o"])), "v": list(range(DIM["o"], ne))}
    prg = {"O": list(range(DIM["O"])), "V": list(range(DIM["O"], npr))}
    NV = {"o": "O", "v": "V"}

    def _sp(l):
        nuc = len(l) > 1 and l[0] == "n"
        base = l[1] if nuc else l[0]
        occ = base in "ijklmno"
        return ("O" if occ else "V") if nuc else ("o" if occ else "v")

    def _antisym(a, classes):
        """Antisymmetrize each run of equal index classes -- i.e. make the tensor a
        PHYSICAL amplitude. The identity being checked here holds term by term only for
        antisymmetric amplitudes: the two expansions consolidate a mutually-cancelling
        pair of terms differently (one side cancels it, the other carries both halves),
        so unsymmetrized random input makes the two sides differ by something that is
        identically zero for any real amplitude."""
        g, i = [], 0
        while i < len(classes):                       # maximal runs of one class
            j = i
            while j + 1 < len(classes) and classes[j + 1] == classes[i]:
                j += 1
            g.append(list(range(i, j + 1))); i = j + 1
        for grp in g:
            if len(grp) < 2:
                continue
            out = np.zeros_like(a)
            for perm in _it.permutations(range(len(grp))):
                sgn = 1
                pl = list(perm)                       # parity by counting inversions
                for x in range(len(pl)):
                    for y in range(x + 1, len(pl)):
                        if pl[x] > pl[y]:
                            sgn = -sgn
                axes = list(range(a.ndim))
                for k, src in enumerate(perm):
                    axes[grp[k]] = grp[src]
                out += sgn * np.transpose(a, axes)
            a = out
        return a

    for mdl in ("neo-ccd(ep)", "neo-ccsd", "neo-ccsdt(eep)"):
        rng2 = np.random.default_rng(7)
        gep = rng2.standard_normal((ne, npr, ne, npr))
        amp = {}                                          # random tensor per amplitude name

        def getamp(name, classes):
            classes = tuple(classes)
            if name not in amp:
                amp[name] = _antisym(rng2.standard_normal(tuple(DIM[c] for c in classes)),
                                     classes)
            assert amp[name].shape == tuple(DIM[c] for c in classes), name
            return amp[name]

        pq = pdaggerq.pq_helper("fermi")
        pq.set_left_operators([["1"]] + [[l] for l in models.lambda_amps(mdl)])
        pq.add_st_operator(1.0, ["gep"], list(models.model(mdl).T))
        pq.simplify()

        def gblk(idx):
            return gep[np.ix_(*[(erg if _sp(i) in "ov" else prg)[_sp(i)] for i in idx])]

        E_dir = 0.0                                       # <(1+L) e^-T gep e^T>
        for term in pq.strings():
            c = float(term[0]); ops = []; subs = []; lts = {}
            for tok in term[1:]:
                nm = tok[:tok.index("(")]; idx = tok[tok.index("(") + 1:-1].split(",")
                ops.append(gblk(idx) if nm == "g" else getamp(nm, [_sp(i) for i in idx]))
                subs.append("".join(lts.setdefault(i, chr(65 + len(lts))) for i in idx))
            E_dir += c * (float(np.einsum(",".join(subs) + "->", *ops, optimize=True)) if ops else 1.0)

        E_rdm = 0.0                                        # -sum gep.D2_ep via rdm_block_ir
        for c4 in _it.product("ov", repeat=4):
            dcls = "".join([NV[c4[1]], c4[0], c4[2], NV[c4[3]]])
            ir = einsums.parse_ir(models.rdm_block_ir(mdl, "D2_ep", dcls))
            if not ir:
                continue
            minp = {}
            for s in ir:
                for o in s["operands"]:
                    if o["name"].startswith("Id["):
                        minp[o["name"]] = np.eye(DIM[o["classes"][0]])
                    else:
                        minp[o["name"]] = getamp(o["name"], o["classes"])
            D = interp(ir, minp)[f'D2_ep["{dcls}"]']
            g = gep[np.ix_(erg[c4[0]], prg[NV[c4[1]]], erg[c4[2]], prg[NV[c4[3]]])]
            E_rdm += float(np.einsum("ijcd,jicd->", g, D, optimize=True))    # +gep.D2_ep
        assert abs(E_dir - E_rdm) < 1e-9, (mdl, E_dir, E_rdm)

    # (3) unpopulatable block -> empty (consumer zero-fills)
    assert models.rdm_block_ir("neo-ccd(ep)", "D1", "ov") == []
    print("test_rdm_block_ir OK")


def test_ir_pairing():
    """Every emitted >=3-operand IR statement must carry a `pairing` field -- the
    optimal binary contraction tree (subset DP in ir_emit, costed with the active
    scaling metric) -- and following that plan must never form an avoidable outer
    product (each binary step's sides share an index unless one side is a scalar).
    Regression: fusion's LinkTracker canonicalises (name-sorts) rebuilt operand
    lists, so the left fold of the emitted operand order can start with an outer
    product (e.g. the combined neo-ccd(ep) t2_ep.t2_ep.tmps quadratics blew up an
    o^2 v^2 O^2 V^2 intermediate); the DP-derived pairing is order-independent.
    Also checks the plan is a valid tree (einsums._fold_steps accepts it) and that
    evaluating VIA the plan reproduces the whole-statement einsum numerically."""
    import numpy as np

    DIM = {"o": 3, "v": 4, "O": 1, "V": 4, "Q": 6}
    rng = np.random.default_rng(3)

    def check(stmts, tag):
        n_multi = 0
        for s in stmts:
            ops = s["operands"]
            if len(ops) < 3:
                continue
            n_multi += 1
            assert "pairing" in s, (tag, s["target"]["name"], "missing pairing")
            steps = einsums._fold_steps(s["pairing"], len(ops))
            assert len(steps) == len(ops) - 1, (tag, "invalid tree", s["pairing"])

            names = [o["indices"] for o in ops]

            # the leaf operands under a step's subtree (to decide kept indices)
            def leaves(ref, out):
                kind, i = ref
                if kind == "op":
                    out.add(i)
                else:
                    leaves(steps[i][0], out)
                    leaves(steps[i][1], out)
                return out

            # (1) no avoidable outer product: each step's sides share an index
            #     unless one side is a scalar (no indices)
            step_sets = []
            def idx_of(ref):
                kind, i = ref
                return set(names[i]) if kind == "op" else step_sets[i]
            for li, ri in steps:
                a, b = idx_of(li), idx_of(ri)
                assert (a & b) or not a or not b, \
                    (tag, s["target"]["name"], "outer-product step", s["pairing"])
                step_sets.append(a | b)

            # (2) numeric: evaluating via the plan == whole-statement einsum
            vals = [rng.standard_normal(tuple(DIM[c] for c in o["classes"]))
                    for o in ops]
            whole = np.einsum(
                ",".join("".join(o["indices"]) for o in ops)
                + "->" + "".join(s["target"]["indices"]), *vals, optimize=True)
            sv, si = [], []
            def val_of(ref):
                kind, i = ref
                return (vals[i], names[i]) if kind == "op" else (sv[i], si[i])
            for li, ri in steps:
                (av, ai), (bv, bi) = val_of(li), val_of(ri)
                sub = leaves(li, set()) | leaves(ri, set())
                later = set(s["target"]["indices"])
                for j in range(len(ops)):
                    if j not in sub:
                        later |= set(names[j])
                out = [l for l in dict.fromkeys(list(ai) + list(bi)) if l in later]
                sv.append(np.einsum(f'{"".join(ai)},{"".join(bi)}->{"".join(out)}',
                                    av, bv))
                si.append(out)
            via_plan = np.einsum(f'{"".join(si[-1])}->{"".join(s["target"]["indices"])}',
                                 sv[-1])
            err = float(np.max(np.abs(via_plan - whole)))
            assert err < 1e-10, (tag, s["target"]["name"], err)
        return n_multi

    n1 = check(einsums.parse_ir(models.equations_ir("neo-ccd(ep)")), "neo-ccd(ep)")
    n2 = check(einsums.parse_ir(models.residual_ir("ccd", "t2")), "ccd/t2")
    assert n1 >= 10 and n2 >= 1, (n1, n2)   # the check must actually exercise plans
    print(f"test_ir_pairing OK ({n1}+{n2} multi-operand plans verified)")


def _check_pt_contract(model_names, dims):
    """The consumer contract for a perturbative correction, checked per block.

    Shared by the (T) and (Q) tests: the checks are identical, only the models and the
    orbital-space sizes differ. ``dims`` is (NO, NV, NPO, NPV); a rank-n antisymmetric
    block of one species is identically zero below n orbitals of that species, so the
    quadruples caller has to pass at least 4 of each for pppp/eeee to be non-vacuous."""
    import itertools, math
    import numpy as np
    from collections import defaultdict
    from pdaggerq._pdaggerq import pq_helper

    # >=3 occupied protons: a rank-3 antisymmetric proton block (ppp) is identically
    # zero below that, and the checks below would be vacuous
    NO, NV, NPO, NPV = dims
    DIM = {"o": NO, "v": NV, "O": NPO, "V": NPV}
    NE, NP = NO + NV, NPO + NPV
    SL = {"o": slice(0, NO), "v": slice(NO, NE), "O": slice(0, NPO), "V": slice(NPO, NP)}
    VIR, OCC = {"v", "V"}, {"o", "O"}

    def antisym(a, cl):
        out = a.copy()
        groups = defaultdict(list)
        for ax, c in enumerate(cl):
            groups[c].append(ax)
        for c, axes in groups.items():
            if len(axes) >= 2 and (c in VIR or c in OCC):
                perms = list(itertools.permutations(range(len(axes))))
                acc = np.zeros_like(out)
                for p in perms:
                    par = sum(1 for i in range(len(p)) for j in range(i + 1, len(p))
                              if p[i] > p[j]) & 1
                    src = list(range(out.ndim))
                    for k, ax in enumerate(axes):
                        src[ax] = axes[p[k]]
                    acc += (-1 if par else 1) * np.transpose(out, src)
                out = acc / len(perms)
        return out

    def interp(ir, inp, target):
        st = {}
        # inputs are keyed by (name, classes). Mixed blocks from rank 3 up now carry
        # distinct names (l3_ep21 vs l3_ep12), so the name alone would do; keying by both
        # is kept because it is what a consumer should do and it catches any regression
        # that reintroduces a shared name
        val = lambda o: (st[o["name"]] if o["name"] in st
                         else inp[(o["name"], tuple(o["classes"]))])
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            c = s["coeff"] * np.einsum(subs + "->" + out,
                                       *[val(o) for o in s["operands"]], optimize=True)
            t = s["target"]["name"]
            st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st[target]

    def externals(ir_list):
        names = {}
        for ir in ir_list:
            produced = {s["target"]["name"] for s in ir}
            for s in ir:
                for o in s["operands"]:
                    if o["name"] not in produced:
                        names[(o["name"], tuple(o["classes"]))] = tuple(o["classes"])
        return names

    def align(a, from_cls, to_cls):
        """reorder axes by index class (within a class the choice is a global sign)"""
        free = defaultdict(list)
        for ax, c in enumerate(from_cls):
            free[c].append(ax)
        return np.transpose(a, [free[c].pop(0) for c in to_cls])

    def chemist(rng, n, m=None):
        """(pq|rs) with the full permutational symmetry of a real ERI"""
        m = n if m is None else m
        A = rng.standard_normal((n * n, m * m))
        if m == n:
            A = A + A.T                                    # (pq) <-> (rs)
        C = A.reshape(n, n, m, m)
        C = C + C.transpose(1, 0, 2, 3)                     # p <-> q
        return C + C.transpose(0, 1, 3, 2)                  # r <-> s

    def integrals(seed):
        rng = np.random.default_rng(seed)
        asym = lambda c: np.einsum("prqs->pqrs", c) - np.einsum("psqr->pqrs", c)
        return {"eri": asym(chemist(rng, NE)),              # <pq||rs>, physicist order
                "vp": asym(chemist(rng, NP)),
                # gep(p,P,q,Q) = (pq|PQ): distinguishable species, no exchange
                "g": np.einsum("pqPQ->pPqQ", chemist(rng, NE, NP))}

    def fill(names, seed, full=None, eps=None):
        rng = np.random.default_rng(seed)
        inp = {}
        for key, cl in sorted(names.items()):
            nm = key[0]
            if '["' in nm:                                  # an integral block
                base, blk = nm.split('["')[0], nm.split('"')[1]
                assert blk == "".join(cl), (nm, cl)
                if base in ("f", "fp"):                     # canonical: diagonal Fock
                    inp[key] = (np.diag(eps[cl[0]]) if cl[0] == cl[1]
                                else np.zeros(tuple(DIM[c] for c in cl)))
                else:
                    inp[key] = full[base][tuple(SL[c] for c in cl)]
            else:
                inp[key] = antisym(rng.standard_normal(tuple(DIM[c] for c in cl)), cl)
        return inp

    def conjugate(inp, names, keep_rank):
        """Lambda = T^dagger: index order reversed (l2_ep(i,I,A,a) = t2_ep(a,A,I,i)).

        Multipliers below ``keep_rank`` are zeroed. A perturbative block of rank n pairs
        directly with L_(n-1); the lower-rank multipliers are the extra terms that make
        the correction ``(T)``/``(Q)`` rather than ``[T]``/``[Q]``, and the pairing
        identity below is the bracketed part. So (T) keeps L2 and drops L1, and (Q)
        keeps L3 and drops L1 and L2."""
        for key, cl in names.items():
            ln = key[0]
            if not ln.startswith("l"):
                continue
            if len(cl) // 2 < keep_rank:
                inp[key] = np.zeros(tuple(DIM[c] for c in cl))
            else:
                tkey = ("t" + ln[1:], tuple(reversed(cl)))
                assert tkey in names, (key, tkey)
                inp[key] = np.transpose(inp[tkey], list(range(len(cl)))[::-1])

    # --- numeric: the consumer contract ---------------------------------------------
    for name in model_names:
        m = models.model(name)
        for amp in m.T_pt:
            num_ir = einsums.parse_ir(models.pt_amplitude_ir(name, amp, df=False))
            e_ir = einsums.parse_ir(models.pt_energy_ir(name, amp, df=False))
            cls = einsums.target_shape(num_ir, "R")[1]
            names = externals([num_ir, e_ir])
            for (ln, _c), cl in list(names.items()):            # every l needs its t
                if ln.startswith("l") and len(cl) >= 4:
                    names.setdefault(("t" + ln[1:], tuple(reversed(cl))), tuple(reversed(cl)))
            w = math.prod(math.factorial(cls.count(c)) for c in set(cls))
            # the weight is the product of the index-group factorials, equivalently
            # (n_e!)^2 (n_p!)^2 for a block with n_e electron and n_p proton excitations:
            # 36 eee/ppp and 4 eep/epp at rank 3; 576 eeee/pppp, 36 eeep/eppp, 16 eepp at 4
            n_p = models._proton_count(amp)
            n_e = len(cls) // 2 - n_p
            assert w == math.factorial(n_e) ** 2 * math.factorial(n_p) ** 2, (name, amp, w)

            # <proj| [F, T_pt] |0>: the zeroth-order coupling that sets the denominator
            pq = pq_helper("fermi")
            pq.set_left_operators([[models.PROJECTION[amp]]])
            for h in m.H:
                if h in ("f", "fp"):
                    pq.add_commutator(1.0, [h], [amp])
            pq.simplify()
            f_ir = einsums.parse_ir(models._optimized(pq, "R", False, 0).to_strings("ir"))
            f_names = externals([f_ir])
            t_key = next(k for k in f_names if not k[0].startswith(("f", "fp")))

            for seed in (1, 2):
                rng = np.random.default_rng(1000 + seed)
                eps = {c: rng.standard_normal(DIM[c]) + (2.0 if c in VIR else -2.0)
                       for c in DIM}
                D = sum((1 if c in VIR else -1) *
                        eps[c].reshape([DIM[c] if k == ax else 1
                                        for k in range(len(cls))])
                        for ax, c in enumerate(cls))

                # (1) Fock coupling: R_F = -D * t  ->  t = numerator / D
                f_inp = fill(f_names, seed, None, eps)
                r_f = interp(f_ir, f_inp, "R")
                t_f = align(f_inp[t_key], list(f_names[t_key]), cls)
                # <proj|[F, T_pt]|0> = s * D * t, with s the PROJECTION reversal parity
                # (-1)^sum_species floor(rank_s / 2): every rank-3 block gives s = -1, but
                # at rank 4 the blocks with an even excitation rank in each species --
                # eeee, eepp, pppp -- flip to s = +1, so those consumers form t = -R/D.
                s_par = (-1) ** (n_e // 2 + n_p // 2)
                assert np.abs(r_f - s_par * D * t_f).max() < 1e-10 * np.abs(r_f).max(), \
                    (name, amp, "Fock coupling is not s*D*t", s_par)

                # (2) pairing: E_[T] = -(1/w) sum(numerator * t)
                inp = fill(names, seed, integrals(seed), eps)
                conjugate(inp, names, len(cls) // 2 - 1)
                num = interp(num_ir, inp, "R")
                # the perturbative amplitude in the energy equation: t3/t3_ep/t3_n at
                # rank 3, t4/t4_ep/t4_n at rank 4
                tgt = [o for st in e_ir for o in st["operands"]
                       if o["name"].startswith("t%d" % (len(cls) // 2))][0]
                tkey_e = (tgt["name"], tuple(tgt["classes"]))
                e = float(interp(e_ir, inp, "energy"))
                dot = float(np.tensordot(align(num, cls, tgt["classes"]),
                                         inp[tkey_e], axes=len(cls)))
                # E = s * (1/w) sum(R * t), with the same reversal parity s as the
                # Fock coupling -- so (T) reads E = -(1/w) sum(R*t) and the even-rank
                # (Q) blocks read E = +(1/w) sum(R*t)
                assert abs(e - s_par * dot / w) < 1e-8 * max(1.0, abs(e)), \
                    (name, amp, "numerator/energy pairing", e, s_par * dot / w)

                # (3) with t = numerator / D the correction is strictly negative
                # the physical amplitude solves R + s*D*t = 0, i.e. t = -R/(s*D);
                # combined with the pairing that gives E = -(1/w) sum(R^2/D) < 0
                pt = dict(inp)
                pt[tkey_e] = align(-num / (s_par * D), cls, list(names[tkey_e]))
                e_pt = float(interp(e_ir, pt, "energy"))
                assert e_pt < 0.0, (name, amp, "(T) correction is not negative", e_pt)

        # (4) the blocks are independent: the fused energy is exactly their sum, so an
        # on-the-fly consumer loses nothing by generating (and looping) them separately
        left = [[l] for l in models.lambda_amps(name)]
        fused = sorted(models._pt_pq(m, left, m.T_pt).strings())
        parts = sorted(s for a in m.T_pt
                       for s in models._pt_pq(m, left, [a]).strings())
        assert fused == parts, (name, len(fused), len(parts))


def test_perturbative_gradient():
    """The (T) GRADIENT quantities: the Lambda source terms and the explicit integral
    derivatives, each validated by finite difference against the object it claims to
    differentiate -- not against another pdaggerq expression.

    Three independent identities, all on a random but physically symmetric electronic
    system (hermitian, properly antisymmetrized <pq||rs>; the identities below are false
    for unsymmetric junk):

    1. LAMBDA SOURCE. With Lambda arbitrary (NOT converged -- the identity is algebraic),

           lambda_ir(cluster) + pt_lambda_source_ir(cluster)  ==  (w/rsign) dL/dt_cluster

       for L = <0|(1+Lambda) e^-T H e^T|0> + E_pt. Both sides carry pq_helper's w/rsign
       weighting for the block, which is exactly why a consumer adds them with coefficient
       one; the test checks that shared normalization rather than assuming it.

    2. dE_pt/df -> D1. Perturbing the orbital energies and rebuilding the perturbative
       amplitude reproduces the emitted D1 blocks. This is the denominator response, and
       it is the piece that has no counterpart in the wavefunction density.

    3. dE_pt/dW -> D2. Two checks: Euler's theorem (E_pt is quadratic in W, so contracting
       the emitted density back with the integral must return 2 E_pt exactly), and a
       directional finite difference in the two-electron integrals.

    NO=NV=4: a rank-3 antisymmetric block over 3 occupied orbitals is proportional to the
    Levi-Civita symbol, which makes spurious identities hold and the checks vacuous."""
    import itertools, json
    from collections import defaultdict
    import numpy as np
    from pdaggerq._pdaggerq import pq_helper

    assert {"pt_lambda_numerator_graph", "pt_lambda_numerator_ir", "pt_lambda_source_ir",
            "pt_gradient_rdm_block_ir"} <= set(models.__all__)

    PT, CC, AMP, S, W = "ccsd(t)", "ccsd", "t3", -1, 36.0
    NO = NV = 4
    NE = NO + NV
    DIM = {"o": NO, "v": NV}
    SL = {"o": slice(0, NO), "v": slice(NO, NE)}

    def antisym(a, cl):
        out = a.copy()
        groups = defaultdict(list)
        for ax, c in enumerate(cl):
            groups[c].append(ax)
        for c, axes in groups.items():
            if len(axes) < 2:
                continue
            acc = np.zeros_like(out)
            for p in itertools.permutations(range(len(axes))):
                par = sum(1 for i in range(len(p)) for j in range(i + 1, len(p))
                          if p[i] > p[j]) & 1
                src = list(range(out.ndim))
                for k, ax in enumerate(axes):
                    src[ax] = axes[p[k]]
                acc += (-1 if par else 1) * np.transpose(out, src)
            out = acc / math.factorial(len(axes))
        return out

    def externals(irs):
        names = {}
        for ir in irs:
            made = {s["target"]["name"] for s in ir}
            for s in ir:
                for o in s["operands"]:
                    if o["name"] not in made:
                        names[(o["name"], tuple(o["classes"]))] = tuple(o["classes"])
        return names

    def align(a, frm, to):
        free = defaultdict(list)
        for ax, c in enumerate(frm):
            free[c].append(ax)
        return np.transpose(a, [free[c].pop(0) for c in to])

    def interp(ir, inp, target):
        st = {}
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            val = lambda o: st[o["name"]] if o["name"] in st else inp[(o["name"], tuple(o["classes"]))]
            c = s["coeff"] * np.einsum(subs + "->" + out, *[val(o) for o in s["operands"]],
                                       optimize=True)
            t = s["target"]["name"]
            st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st[target]

    rng = np.random.default_rng(2024)
    A = rng.standard_normal((NE * NE, NE * NE))
    A = A + A.T
    C = A.reshape(NE, NE, NE, NE)
    C = C + C.transpose(1, 0, 2, 3)
    C = C + C.transpose(0, 1, 3, 2)
    ERI = np.einsum("prqs->pqrs", C) - np.einsum("psqr->pqrs", C)      # <pq||rs>
    EPS = {c: rng.standard_normal(DIM[c]) + (2.0 if c == "v" else -2.0) for c in DIM}
    t1 = rng.standard_normal((NV, NO)) * 0.1
    t2 = antisym(rng.standard_normal((NV, NV, NO, NO)), "vvoo") * 0.1
    L1 = rng.standard_normal((NO, NV)) * 0.1
    L2 = antisym(rng.standard_normal((NO, NO, NV, NV)), "oovv") * 0.1
    rev = lambda a: np.transpose(a, list(range(a.ndim))[::-1])

    def fill(names, amps, eri, eps):
        inp = {}
        for key, cl in names.items():
            nm = key[0]
            if nm.startswith("Id"):
                inp[key] = np.eye(DIM[cl[0]])
            elif nm.startswith("eri"):
                inp[key] = eri[tuple(SL[c] for c in cl)]
            elif nm.startswith(("f", "fp")):
                inp[key] = (np.diag(eps[cl[0]]) if cl[0] == cl[1]
                            else np.zeros(tuple(DIM[c] for c in cl)))
            else:
                inp[key] = align(amps[nm], amps[nm + "_cls"], list(cl))
        return inp

    parse = einsums.parse_ir
    NUM = parse(models.pt_amplitude_ir(PT, AMP, df=False))
    LNUM = parse(models.pt_lambda_numerator_ir(PT, AMP, df=False))
    EIR = parse(models.pt_energy_ir(PT, AMP, df=False))
    RC = einsums.target_shape(NUM, "R")[1]
    LC = einsums.target_shape(LNUM, "R")[1]

    def denom(eps):
        return sum((1 if c == "v" else -1) *
                   eps[c].reshape([DIM[c] if k == ax else 1 for k in range(len(RC))])
                   for ax, c in enumerate(RC))

    def triples(a, b, eri, eps):
        amps = {"t1": a, "t1_cls": "vo", "t2": b, "t2_cls": "vvoo"}
        D = denom(eps)
        R = interp(NUM, fill(externals([NUM]), amps, eri, eps), "R")
        Rl = align(interp(LNUM, fill(externals([LNUM]), amps, eri, eps), "R"), LC, RC)
        return -R / (S * D), -Rl / (S * D)

    def e_pt(a, b, eri, eps):
        t3, _ = triples(a, b, eri, eps)
        amps = {"t1": a, "t1_cls": "vo", "t2": b, "t2_cls": "vvoo",
                "l1": a.T, "l1_cls": "ov", "l2": rev(b), "l2_cls": "oovv",
                "t3": t3, "t3_cls": RC}
        return float(interp(EIR, fill(externals([EIR]), amps, eri, eps), "energy"))

    # the pairing identity the whole derivation rests on: E_pt = rsign/w sum(R_lam * t_pt)
    t3, l3lam = triples(t1, t2, ERI, EPS)
    E = e_pt(t1, t2, ERI, EPS)
    Rl = align(interp(LNUM, fill(externals([LNUM]),
               {"t1": t1, "t1_cls": "vo", "t2": t2, "t2_cls": "vvoo"}, ERI, EPS), "R"), LC, RC)
    assert abs(E - S * float((Rl * t3).sum()) / W) < 1e-12 * abs(E), "R_lam/energy pairing"

    base = {"t1": t1, "t1_cls": "vo", "t2": t2, "t2_cls": "vvoo",
            "l1": L1, "l1_cls": "ov", "l2": L2, "l2_cls": "oovv",
            "t3": t3, "t3_cls": RC, "l3": rev(t3), "l3_cls": list(RC)[::-1],
            "l3_lam": rev(l3lam), "l3_lam_cls": list(RC)[::-1]}
    # the integral derivatives below take the (T) PRESCRIPTION multipliers l = t^dagger,
    # the ones E_pt is defined with -- NOT the solved Lambda that the Lagrangian part
    # above uses. Same operand names, different arrays; feeding the solved Lambda here is
    # the one way to get this wrong silently, so the test uses both and keeps them apart.
    base_pt = dict(base)
    base_pt.update({"l1": t1.T, "l1_cls": "ov", "l2": rev(t2), "l2_cls": "oovv"})

    # --- 1. the augmented Lambda equations --------------------------------------------
    pq = pq_helper("fermi")
    pq.set_left_operators([["1"]] + [[l] for l in models.lambda_amps(CC)])
    for h in models.model(CC).H:
        pq.add_st_operator(1.0, [h], list(models.model(CC).T))
    pq.simplify()
    LAG = parse(models._optimized(pq, "energy", False, 0).to_strings("ir"))

    def lagrangian(a, b):
        amps = dict(base); amps["t1"] = a; amps["t2"] = b
        return float(interp(LAG, fill(externals([LAG]), amps, ERI, EPS), "energy")) \
            + e_pt(a, b, ERI, EPS)

    for cluster, cls in (("t1", "vo"), ("t2", "vvoo")):
        lam = parse(models.lambda_ir(CC, cluster, df=False, opt_level=0))
        src = [json.loads(l) for l in models.pt_lambda_source_ir(PT, AMP, cluster)]
        assert not any(o["is_intermediate"] for s in src for o in s["operands"]), \
            "source terms must carry no intermediates"
        assert max(sum(1 for o in s["operands"] if o["name"].startswith("l3"))
                   for s in src) == 1, "one perturbative operand per statement"
        lc = einsums.target_shape(lam, "R")[1]
        sc = einsums.target_shape(src, "R")[1]
        tot = (align(interp(lam, fill(externals([lam]), base, ERI, EPS), "R"), lc, cls)
               + align(interp(src, fill(externals([src]), base, ERI, EPS), "R"), sc, cls))
        n_e = len(cls) // 2
        wos = math.factorial(n_e) ** 2 / (-1) ** (n_e // 2)
        d = (rng.standard_normal(t1.shape) if cluster == "t1"
             else antisym(rng.standard_normal(t2.shape), "vvoo"))
        h = 1e-5
        fd = (lagrangian(t1 + h * d if cluster == "t1" else t1,
                         t2 + h * d if cluster == "t2" else t2)
              - lagrangian(t1 - h * d if cluster == "t1" else t1,
                           t2 - h * d if cluster == "t2" else t2)) / (2 * h)
        ana = float((tot * d).sum())
        assert abs(ana - wos * fd) < 1e-6 * max(1.0, abs(wos * fd)), \
            (cluster, "augmented Lambda residual", ana, wos * fd)

    # --- 2. dE_pt/df -> D1 -------------------------------------------------------------
    D1 = {}
    for b in ("oo", "ov", "vo", "vv"):
        ir = [json.loads(l) for l in models.pt_gradient_rdm_block_ir(PT, AMP, "D1", b)]
        if ir:
            assert max(sum(1 for o in s["operands"] if o["name"].startswith(("t3", "l3")))
                       for s in ir) == 2, "D1 pairs the amplitude with the multiplier"
            D1[b] = interp(ir, fill(externals([ir]), base_pt, ERI, EPS), f'D1["{b}"]')
    assert set(D1) == {"oo", "vv"}, sorted(D1)
    de = {c: rng.standard_normal(DIM[c]) for c in DIM}
    h = 1e-4
    ep = {c: EPS[c] + h * de[c] for c in DIM}
    em = {c: EPS[c] - h * de[c] for c in DIM}
    fd = (e_pt(t1, t2, ERI, ep) - e_pt(t1, t2, ERI, em)) / (2 * h)
    ana = sum(float((np.diag(D1[c + c]) * de[c]).sum()) for c in DIM)
    assert abs(ana - fd) < 1e-7 * max(1.0, abs(fd)), ("D1 denominator response", ana, fd)

    # --- 3. dE_pt/dW -> D2 -------------------------------------------------------------
    D2 = {}
    for b in ("".join(p) for p in itertools.product("ov", repeat=4)):
        ir = [json.loads(l) for l in models.pt_gradient_rdm_block_ir(PT, AMP, "D2", b)]
        if ir:
            assert max(sum(1 for o in s["operands"] if o["name"].startswith(("t3", "l3")))
                       for s in ir) == 1, "D2 carries one perturbative operand per statement"
            D2[b] = interp(ir, fill(externals([ir]), base_pt, ERI, EPS), f'D2["{b}"]')
    pair = lambda X: sum(0.25 * float(np.einsum("pqsr,pqrs->", D2[b],
                         X[SL[b[0]], SL[b[1]], SL[b[3]], SL[b[2]]])) for b in D2)
    assert abs(pair(ERI) - 2 * E) < 1e-10 * abs(E), ("Euler on W", pair(ERI), 2 * E)
    B = rng.standard_normal((NE * NE, NE * NE)); B = B + B.T
    C2 = B.reshape(NE, NE, NE, NE)
    C2 = C2 + C2.transpose(1, 0, 2, 3)
    C2 = C2 + C2.transpose(0, 1, 3, 2)
    dW = np.einsum("prqs->pqrs", C2) - np.einsum("psqr->pqrs", C2)
    h = 1e-4
    fd = (e_pt(t1, t2, ERI + h * dW, EPS) - e_pt(t1, t2, ERI - h * dW, EPS)) / (2 * h)
    assert abs(pair(dW) - fd) < 1e-7 * max(1.0, abs(fd)), ("D2 vs finite difference",
                                                           pair(dW), fd)
    # --- 4. NEO: the source lands on lambda_ir's target slot for slot -----------------
    # A mixed block is where this can go wrong: tep11's excitation operator reads
    # e2(a,nA,i,nI) but pq_graph groups the target by species, so lambda_ir's target is
    # (a,i,A,I). A transposed source would still have the right shape and the right block,
    # so nothing but this check would notice.
    neo = "neo-ccsd(t)-1p"
    for cluster in models.model(neo).T:
        lam = einsums.parse_ir(models.lambda_ir(neo, cluster, df=False, opt_level=0))
        src = [json.loads(l) for l in models.pt_lambda_source_ir(neo, "tep21", cluster)]
        if not src:
            continue
        assert einsums.target_shape(lam, "R")[1] == einsums.target_shape(src, "R")[1], \
            (neo, cluster, "source target classes differ from lambda_ir's")
        assert lam[0]["target"]["indices"] == src[0]["target"]["indices"], \
            (neo, cluster, "source target index order differs from lambda_ir's")

    print("test_perturbative_gradient OK")


def test_perturbative_gradient_neo():
    """The (T) gradient density for NEO, where the two-body blocks come in three
    conventions and each carries its own sign -- none of which is guessable.

        D2 / D2_n   same species, antisymmetrized <pq||rs>, the 1/4 of W's definition,
                    and the density operator's last two arguments swapped
        D2_ep       gep multiplies a PLAIN product of one-body operators, one per
                    species; electron and proton operators commute, so no swap, no 1/4,
                    and the opposite sign
        D1 / D1_n   the denominator response, one per species

    Also guards the mixed-block operand naming: the perturbative multiplier is fed to
    pq_helper as ``lep21`` but PRINTS as ``l3_ep21``, so the two halves must be told apart
    by a rule on the emitted name, not by the name that went in. When that was keyed on the
    input name the two halves silently shared one operand and the density was wrong with no
    other symptom.

    tep12 (one electron, two proton excitations) is the block that reaches the
    proton-proton integral; tep21 does not, and its D2_n is correctly empty."""
    import itertools, json
    from collections import defaultdict
    import numpy as np

    NAME = "neo-ccsd(t)"
    NO, NV, NPO, NPV = 3, 3, 2, 2
    NE, NP = NO + NV, NPO + NPV
    DIM = {"o": NO, "v": NV, "O": NPO, "V": NPV}
    SL = {"o": slice(0, NO), "v": slice(NO, NE), "O": slice(0, NPO), "V": slice(NPO, NP)}
    VIR = {"v", "V"}

    def antisym(a, cl):
        out = a.copy()
        groups = defaultdict(list)
        for ax, c in enumerate(cl):
            groups[c].append(ax)
        for c, axes in groups.items():
            if len(axes) < 2:
                continue
            acc = np.zeros_like(out)
            for p in itertools.permutations(range(len(axes))):
                par = sum(1 for i in range(len(p)) for j in range(i + 1, len(p))
                          if p[i] > p[j]) & 1
                src = list(range(out.ndim))
                for k, ax in enumerate(axes):
                    src[ax] = axes[p[k]]
                acc += (-1 if par else 1) * np.transpose(out, src)
            out = acc / math.factorial(len(axes))
        return out

    def externals(irs):
        names = {}
        for ir in irs:
            made = {s["target"]["name"] for s in ir}
            for s in ir:
                for o in s["operands"]:
                    if o["name"] not in made:
                        names[(o["name"], tuple(o["classes"]))] = tuple(o["classes"])
        return names

    def align(a, frm, to):
        free = defaultdict(list)
        for ax, c in enumerate(frm):
            free[c].append(ax)
        return np.transpose(a, [free[c].pop(0) for c in to])

    def interp(ir, inp, target):
        st = {}
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            val = lambda o: st[o["name"]] if o["name"] in st else inp[(o["name"], tuple(o["classes"]))]
            c = s["coeff"] * np.einsum(subs + "->" + out, *[val(o) for o in s["operands"]],
                                       optimize=True)
            t = s["target"]["name"]
            st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st[target]

    def chem(rng, n, m=None):
        m = n if m is None else m
        A = rng.standard_normal((n * n, m * m))
        if m == n:
            A = A + A.T
        C = A.reshape(n, n, m, m)
        C = C + C.transpose(1, 0, 2, 3)
        return C + C.transpose(0, 1, 3, 2)
    asym = lambda c: np.einsum("prqs->pqrs", c) - np.einsum("psqr->pqrs", c)

    rng = np.random.default_rng(4242)
    ERI, VP = asym(chem(rng, NE)), asym(chem(rng, NP))
    G = np.einsum("pqPQ->pPqQ", chem(rng, NE, NP))
    EPS = {c: rng.standard_normal(DIM[c]) + (2.0 if c in VIR else -2.0) for c in DIM}
    rev = lambda a: np.transpose(a, list(range(a.ndim))[::-1])
    AMPS = {}
    for nm, cls in (("t1", "vo"), ("t2", "vvoo"), ("t1_n", "VO"), ("t2_n", "VVOO"),
                    ("t2_ep", "vVoO")):
        AMPS[nm] = antisym(rng.standard_normal(tuple(DIM[c] for c in cls)), cls) * 0.1
        AMPS[nm + "_cls"] = cls
        ln = "l" + nm[1:]
        AMPS[ln], AMPS[ln + "_cls"] = rev(AMPS[nm]), cls[::-1]

    def fill(names, amps, eri, vp, g, eps):
        inp = {}
        for key, cl in names.items():
            nm = key[0]
            if nm.startswith("Id"):
                inp[key] = np.eye(DIM[cl[0]])
            elif nm.startswith("eri"):
                inp[key] = (eri if cl[0] in "ov" else vp)[tuple(SL[c] for c in cl)]
            elif nm.startswith("vp"):
                inp[key] = vp[tuple(SL[c] for c in cl)]
            elif nm.startswith("g"):
                inp[key] = g[tuple(SL[c] for c in cl)]
            elif nm.startswith(("f", "fp")):
                inp[key] = (np.diag(eps[cl[0]]) if cl[0] == cl[1]
                            else np.zeros(tuple(DIM[c] for c in cl)))
            else:
                inp[key] = align(amps[nm], amps[nm + "_cls"], list(cl))
        return inp

    for AMP in ("tep21", "tep12"):
        NUM = einsums.parse_ir(models.pt_amplitude_ir(NAME, AMP, df=False))
        LNUM = einsums.parse_ir(models.pt_lambda_numerator_ir(NAME, AMP, df=False))
        EIR = einsums.parse_ir(models.pt_energy_ir(NAME, AMP, df=False))
        RC = einsums.target_shape(NUM, "R")[1]
        LC = einsums.target_shape(LNUM, "R")[1]
        n_p = models._proton_count(AMP)
        n_e = len(RC) // 2 - n_p
        S = (-1) ** (n_e // 2 + n_p // 2)
        W = math.factorial(n_e) ** 2 * math.factorial(n_p) ** 2

        def denom(eps):
            return sum((1 if c in VIR else -1) *
                       eps[c].reshape([DIM[c] if k == ax else 1 for k in range(len(RC))])
                       for ax, c in enumerate(RC))

        def triples(eri, vp, g, eps):
            D = denom(eps)
            R = interp(NUM, fill(externals([NUM]), AMPS, eri, vp, g, eps), "R")
            Rl = align(interp(LNUM, fill(externals([LNUM]), AMPS, eri, vp, g, eps), "R"), LC, RC)
            return -R / (S * D), -Rl / (S * D)

        TGT = [o["name"] for s in EIR for o in s["operands"] if o["name"].startswith("t3")][0]

        def e_pt(eri, vp, g, eps):
            t, _ = triples(eri, vp, g, eps)
            a = dict(AMPS); a[TGT] = t; a[TGT + "_cls"] = "".join(RC)
            return float(interp(EIR, fill(externals([EIR]), a, eri, vp, g, eps), "energy"))

        t_pt, l_pt = triples(ERI, VP, G, EPS)
        E = e_pt(ERI, VP, G, EPS)
        Rl = align(interp(LNUM, fill(externals([LNUM]), AMPS, ERI, VP, G, EPS), "R"), LC, RC)
        assert abs(E - S * float((Rl * t_pt).sum()) / W) < 1e-10 * abs(E), (AMP, "pairing")

        base = dict(AMPS)
        base[TGT], base[TGT + "_cls"] = t_pt, "".join(RC)
        ln = "l" + TGT[1:]
        base[ln], base[ln + "_cls"] = rev(t_pt), "".join(RC)[::-1]
        base[ln + "_lam"], base[ln + "_lam_cls"] = rev(l_pt), "".join(RC)[::-1]

        # the two halves must be distinguishable on a MIXED block. Which cluster blocks a
        # perturbative block couples to varies (tep12 has no source against the pure
        # electron doubles at all), so look across all of them.
        seen = {o["name"] for cl in models.model(NAME).T
                for l in models.pt_lambda_source_ir(NAME, AMP, cl)
                for o in json.loads(l)["operands"]}
        assert {ln, ln + "_lam"} <= seen, \
            (AMP, "the two source halves share an operand name", sorted(seen))

        def blocks(tensor, letters, rank):
            out = {}
            for b in ("".join(p) for p in itertools.product(*([letters] * rank))):
                ir = [json.loads(l) for l in
                      models.pt_gradient_rdm_block_ir(NAME, AMP, tensor, b)]
                if ir:
                    out[b] = interp(ir, fill(externals([ir]), base, ERI, VP, G, EPS),
                                    f'{tensor}["{b}"]')
            return out

        # 1. D1 / D1_n -- the denominator response, one species at a time
        for tensor, letters in (("D1", "ov"), ("D1_n", "OV")):
            D1 = blocks(tensor, letters, 2)
            assert set(D1) == {letters[0] * 2, letters[1] * 2}, (AMP, tensor, sorted(D1))
            de = {c: rng.standard_normal(DIM[c]) for c in letters}
            h = 1e-4
            ep = {c: EPS[c] + (h * de[c] if c in letters else 0) for c in DIM}
            em = {c: EPS[c] - (h * de[c] if c in letters else 0) for c in DIM}
            fd = (e_pt(ERI, VP, G, ep) - e_pt(ERI, VP, G, em)) / (2 * h)
            ana = sum(float((np.diag(D1[c + c]) * de[c]).sum()) for c in letters)
            assert abs(ana - fd) < 1e-6 * max(1.0, abs(fd)), (AMP, tensor, ana, fd)

        # 2. D2_ep -- consumer pairing gep[e,p,e',p'] * D2_ep[p,e,e',p'], coefficient 1
        Dep = {}
        for b in ("".join(p) for p in itertools.product("OV", "ov", "ov", "OV")):
            ir = [json.loads(l) for l in models.pt_gradient_rdm_block_ir(NAME, AMP, "D2_ep", b)]
            if ir:
                Dep[b] = interp(ir, fill(externals([ir]), base, ERI, VP, G, EPS),
                                f'D2_ep["{b}"]')
        assert Dep, (AMP, "D2_ep empty")
        dg = np.einsum("pqPQ->pPqQ", chem(rng, NE, NP))
        h = 1e-4
        fd = (e_pt(ERI, VP, G + h * dg, EPS) - e_pt(ERI, VP, G - h * dg, EPS)) / (2 * h)
        ana = sum(float(np.einsum("pefq,epfq->", Dep[b],
                  dg[SL[b[1]], SL[b[0]], SL[b[2]], SL[b[3]]])) for b in Dep)
        assert abs(ana - fd) < 1e-6 * max(1.0, abs(fd)), (AMP, "D2_ep", ana, fd)

        # 3. D2_n -- same-species, so the electronic convention: 1/4 vp[pqrs] D2_n[pqsr].
        # tep21 cannot reach the proton-proton integral at all; its block is empty and so
        # is the derivative, which is itself worth asserting.
        Dn = blocks("D2_n", "OV", 4)
        dvp = asym(chem(rng, NP))
        fd = (e_pt(ERI, VP + h * dvp, G, EPS) - e_pt(ERI, VP - h * dvp, G, EPS)) / (2 * h)
        if Dn:
            ana = sum(0.25 * float(np.einsum("pqsr,pqrs->", Dn[b],
                      dvp[SL[b[0]], SL[b[1]], SL[b[3]], SL[b[2]]])) for b in Dn)
            assert abs(ana - fd) < 1e-6 * max(1.0, abs(fd)), (AMP, "D2_n", ana, fd)
        else:
            assert abs(fd) < 1e-10, (AMP, "D2_n empty but dE/dvp is not", fd)
    print("test_perturbative_gradient_neo OK")

def test_dims_cost_model():
    """The optimizer ranks contraction candidates by numeric flops at the dimensions it
    is given, so it removes the LARGE indices first -- and the second species' basis is
    typically far smaller than the electronic one (markedly so for muons and positrons).

    Checks that (a) the dimension-aware metric beats the dimension-blind one at realistic
    NEO sizes, and (b) ``dims=`` reaches the optimizer, so a consumer can supply its own
    basis sizes rather than the representative defaults."""
    TRUE = {"o": 10.0, "v": 40.0, "O": 1.0, "V": 4.0, "L": 1.0, "Q": 120.0}

    def flops(lines):
        """exact flop count of the binary contractions (the product of the extents of a
        binary einsum's index union); multi-operand statements are skipped rather than
        charged their un-factorized nested-loop cost, which would badly overestimate"""
        tot = 0.0
        for st in einsums.parse_ir(lines):
            if len(st["operands"]) != 2:
                continue
            idx = {}
            for o in st["operands"] + [st["target"]]:
                for i, c in zip(o["indices"], o["classes"]):
                    idx[i] = TRUE[c]
            p = 1.0
            for d in idx.values():
                p *= d
            tot += p
        return tot

    m = models.model("neo-ccsd")
    pq = lambda: models._projected_pq(m, [models.PROJECTION["tep11"]])
    blind = flops(models._optimized(pq(), "R", True, 6, None).to_strings("ir"))
    aware = flops(models._optimized(pq(), "R", True, 6, models._dims_for("neo-ccsd")).to_strings("ir"))
    matched = flops(models.residual_ir("neo-ccsd", "tep11", dims={"V": 4.0}))

    assert aware < blind, (aware, blind)          # knowing the sizes at all helps
    assert matched < aware, (matched, aware)      # knowing the RIGHT sizes helps more

    # dims reaches the electron-only models too (they are dimension-blind by default)
    assert models._dims_for("ccsd") is None
    assert models._dims_for("ccsd", {"v": 400.0})["v"] == 400.0
    # partial dicts merge over the defaults rather than replacing them
    t = models._dims_for("neo-ccsd", {"V": 4.0})
    # O defaults to the model's max proton rank (neo-ccsd carries tp2, so 2), which is
    # the smallest proton count at which every one of its blocks is nonzero
    assert t["V"] == 4.0 and t["v"] == models.DIMS["v"] and t["O"] == 2.0
    assert models._dims_for("neo-ccsd", {"O": 1.0})["O"] == 1.0    # and is overridable

    for bad in (lambda: models._dims_for("neo-ccsd", {"X": 1.0}),
                lambda: models.residual_ir("neo-ccsd", "tp1", dims={"nope": 2.0})):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass
    print("test_dims_cost_model OK")


def test_hamiltonian_split():
    """``operators=`` splits H exactly: a subset plus its complement reproduces the full
    equation, and each part is self-contained.

    This is what lets a consumer drop the proton-proton (``vp``) terms at load time. vp
    is part of H, not of the cluster, so it survives every amplitude gate -- and in the
    emitted code it is not identifiable, since density-fitted vp is proton-B x proton-B
    where gep is electron-B x proton-B and the optimizer folds both into intermediates.
    Generating the parts separately sidesteps that entirely.

    The identity holds for ARBITRARY inputs: both sides are the same polynomial in the
    same tensors, so no physical symmetry is needed here."""
    import numpy as np

    DIM = {"o": 2, "v": 3, "O": 2, "V": 3, "Q": 4}

    def run(ir_lines, inp, rng):
        ir = einsums.parse_ir(ir_lines)
        st = {}
        for s_ in ir:
            ops = []
            for o in s_["operands"]:
                if o.get("is_intermediate"):
                    ops.append(st[o["name"]]); continue
                key = (o["name"], tuple(o["classes"]))
                if key not in inp:
                    inp[key] = rng.standard_normal(tuple(DIM[c] for c in o["classes"]))
                ops.append(inp[key])
            subs = ",".join("".join(o["indices"]) for o in s_["operands"])
            out = "".join(s_["target"]["indices"])
            val = s_["coeff"] * np.einsum(subs + "->" + out, *ops, optimize=True)
            t = s_["target"]["name"]
            st[t] = val.copy() if s_["is_assignment"] else st[t] + val
        return st["R"]

    m = models.model("neo-ccsd")
    assert "vp" in m.H
    rest = tuple(h for h in m.H if h != "vp")

    for amp in ("tp1", "tep11"):
        full = models.residual_ir("neo-ccsd", amp, df=False)
        base = models.residual_ir("neo-ccsd", amp, df=False, operators=rest)
        vp = models.residual_ir("neo-ccsd", amp, df=False, operators=("vp",))

        inp = {}                                   # shared, so all three see one input set
        rng = np.random.default_rng(2026)
        r_full, r_base, r_vp = (run(x, inp, rng) for x in (full, base, vp))
        assert np.max(np.abs(r_full - (r_base + r_vp))) < 1e-11 * max(
            1.0, float(np.max(np.abs(r_full)))), amp

        # the vp part touches only proton-proton integrals -- no electron eri, no gep
        for st_ in einsums.parse_ir(vp):
            for o in st_["operands"]:
                if o.get("is_intermediate") or o["name"].startswith(("t", "l")):
                    continue
                if o["name"].startswith("eri"):
                    assert set(o["classes"]) <= {"O", "V"}, (amp, o["name"], o["classes"])
                assert not o["name"].startswith("g"), (amp, o["name"])   # gep is not vp

    # the split is exact for the energy too
    e_full = models.energy_graph("neo-ccsd", df=False).to_strings("ir")
    e_rest = models.energy_graph("neo-ccsd", df=False, operators=rest).to_strings("ir")
    e_vp = models.energy_graph("neo-ccsd", df=False, operators=("vp",)).to_strings("ir")
    assert len([l for l in e_vp if l.strip().startswith("{")]) > 0, "vp energy is empty"

    for bad in (lambda: models.residual_ir("neo-ccsd", "tp1", operators=("nope",)),
                lambda: models.residual_ir("neo-ccsd", "tp1", operators=("vp", "zz"))):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass
    print("test_hamiltonian_split OK")


def test_perturbative_rdm():
    """The perturbative contribution to the RDMs, validated against an INDEPENDENT
    construction: it must equal the CCSDT density's terms at the orders it keeps.

    ``ccsd(t)``'s perturbative density is, by construction, the terms of the full CCSDT
    density carrying the perturbative amplitude and/or its multiplier at first order --
    one ``t3`` and no ``l3``, one ``l3`` and no ``t3``, or one of each. Filtering the
    CCSDT density that way must give the same arrays, which pins the construction without
    relying on any energy/trace convention.

    Also pins the property a slice-wise consumer depends on: the emitted statements carry
    NO intermediates, so no intermediate can be shared between statements that pin
    different axes of it."""
    import itertools
    import numpy as np
    from collections import defaultdict
    from pdaggerq._pdaggerq import pq_helper

    NO, NV = 3, 4
    DIM = {"o": NO, "v": NV}
    rng = np.random.default_rng(9)
    amps = {}

    def antisym(a, cl):
        out = a.copy()
        g = defaultdict(list)
        for ax, c in enumerate(cl):
            g[c].append(ax)
        for c, axes in g.items():
            if len(axes) < 2:
                continue
            perms = list(itertools.permutations(range(len(axes))))
            acc = np.zeros_like(out)
            for perm in perms:
                sgn, pl = 1, list(perm)
                for x in range(len(pl)):
                    for y in range(x + 1, len(pl)):
                        if pl[x] > pl[y]:
                            sgn = -sgn
                src = list(range(out.ndim))
                for k, ax in enumerate(axes):
                    src[ax] = axes[perm[k]]
                acc += sgn * np.transpose(out, src)
            out = acc / len(perms)
        return out

    def amp(nm, cl):
        if nm not in amps:
            amps[nm] = antisym(rng.standard_normal(tuple(DIM[c] for c in cl)), cl)
        return amps[nm]

    spc = lambda i: "o" if i[0] in "ijklmn" else "v"

    def ev(strings, consumer):
        openlab = [L for L, _ in consumer]
        D = np.zeros([DIM[c] for _, c in consumer])
        for term in strings:
            c = float(term[0])
            perms, facs = [], []
            for tok in term[1:]:
                if tok.startswith("P("):
                    perms.append(tok[2:-1].split(","))
                elif "(" in tok:
                    facs.append(tok)
            lts = {x: chr(97 + i) for i, x in enumerate(openlab)}
            nxt = [len(openlab)]
            ops, subs = [], []
            for tok in facs:
                nm = tok[:tok.index("(")]
                idx = tok[tok.index("(") + 1:-1].split(",")
                ss = ""
                for i in idx:
                    if i not in lts:
                        lts[i] = chr(97 + nxt[0]); nxt[0] += 1
                    ss += lts[i]
                subs.append(ss)
                cl = [spc(i) for i in idx]
                ops.append(np.eye(DIM[cl[0]]) if nm == "d" else amp(nm, cl))
            out = "".join(chr(97 + i) for i in range(len(openlab)))
            base = c * np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)
            variants = [(1.0, list(range(len(openlab))))]
            for a, b in perms:
                ia, ib = openlab.index(a), openlab.index(b)
                nv = []
                for sgn, ax in variants:
                    nv.append((sgn, ax))
                    sw = list(ax); sw[ia], sw[ib] = sw[ib], sw[ia]
                    nv.append((-sgn, sw))
                variants = nv
            for sgn, ax in variants:
                D += sgn * np.transpose(base, np.argsort(ax))
        return D

    cnt = lambda t, nm: sum(1 for x in t[1:] if x.startswith(nm + "("))

    worst = 0.0
    for tensor, blocks in (("D1", ["oo", "ov", "vo", "vv"]),
                           ("D2", ["oooo", "ooov", "oovo", "oovv", "ovoo", "ovov",
                                   "ovvo", "ovvv", "vooo", "voov", "vovo", "vovv",
                                   "vvoo", "vvov", "vvvo", "vvvv"])):
        for b in blocks:
            op, consumer = models._rdm_block_spec(tensor, b)
            q = pq_helper("fermi")
            q.set_left_operators([["1"]] + [[l] for l in models.lambda_amps("ccsdt")])
            q.add_st_operator(1.0, [op], ["t1", "t2", "t3"])
            q.simplify()
            ref = [t for t in q.strings()
                   if 0 < cnt(t, "t3") + cnt(t, "l3")
                   and cnt(t, "t3") <= 1 and cnt(t, "l3") <= 1]
            mine = models._pt_rdm_pq("ccsd(t)", op, "t3").strings()
            a1, a2 = ev(ref, consumer), ev(mine, consumer)
            err = float(np.max(np.abs(a1 - a2)))
            worst = max(worst, err / max(1.0, float(np.max(np.abs(a1)))))
    assert worst < 1e-12, worst

    # the diagonal blocks come only from the L_pt.T_pt cross term; a construction that
    # drops it leaves them empty, which would pass every other check here
    for b in ("oo", "vv"):
        assert models.pt_rdm_block_ir("ccsd(t)", "t3", "D1", b), b

    # no intermediates -> a slice-wise consumer can never hit an intermediate shared
    # between statements that pin different axes of it
    for name, amp_ in (("ccsd(t)", "t3"), ("neo-ccsd(t)", "tep21"), ("neo-ccsd(t)", "tp3")):
        for tensor, b in (("D1", "oo"), ("D2", "oovv")):
            for st in einsums.parse_ir(models.pt_rdm_block_ir(name, amp_, tensor, b)):
                assert not st["target"].get("is_intermediate"), (name, tensor, b)
                for o in st["operands"]:
                    assert not o.get("is_intermediate"), (name, tensor, b)

    # the perturbative amplitude and its multiplier are ordinary operands, under the
    # rank-disambiguated names, so a slice-wise driver binds them directly
    ir = einsums.parse_ir(models.pt_rdm_block_ir("neo-ccsd(t)", "tep21", "D2", "oovv"))
    seen = {o["name"] for st in ir for o in st["operands"]}
    assert "t3_ep21" in seen and "l3_ep21" in seen, sorted(seen)

    for bad in (lambda: models.pt_rdm_block_ir("ccsd(t)", "tep21", "D1", "oo"),
                lambda: models.pt_rdm_graph("ccsd", "t3", "e1(i,j)")):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass
    print("test_perturbative_rdm OK")


def test_perturbative_triples():
    """CCSD(T) / NEO-CCSD(T): the perturbative blocks, and the CONSUMER CONTRACT that
    ties the two generated equations together.

    Symbolic: the electronic model reproduces the canonical ``examples/ccsd_t.py``
    derivation term for term.

    Numeric, per block, with physically-symmetric integrals (hermitian, correctly
    antisymmetrized -- the identities below use W = W^dagger, so random unsymmetric
    integrals do NOT satisfy them):

    1. the Fock coupling is ``<proj|[F, T_pt]|0> = -D * t``, ``D = sum(e_vir) -
       sum(e_occ)``, so the amplitude is ``t = numerator / D``  -- the denominator
       :func:`models.pt_amplitude_graph` documents, and the same sign rule the doubles
       residuals of this library already use;
    2. ``E_[T] = -(1/w) sum(numerator * t)`` with ``w`` the product of the index-group
       factorials (36 for eee, 4 for eep/epp) -- this is the pairing an on-the-fly
       implementation relies on, and it pins the RELATIVE index order and sign of the
       two equations;
    3. therefore ``E_[T] = -(1/w) sum(numerator^2 / D) < 0`` STRICTLY, for any
       integrals -- the physical sign of a perturbative-triples correction;
    4. the per-block energies sum to the fused ``pt_energy_ir(name)``, so generating
       the blocks separately (which is what lets a consumer contract the triples away
       on the fly) loses nothing.
    """
    import math
    from pdaggerq._pdaggerq import pq_helper

    # --- structure -----------------------------------------------------------------
    assert {"pt_amps", "pt_amplitude_ir", "pt_energy_ir"} <= set(models.__all__)
    assert models.pt_amps("ccsd(t)") == ["t3"]
    assert models.pt_amps("neo-ccsd(t)") == ["t3", "tep21", "tep12", "tp3"]  # eee/eep/epp/ppp
    assert models.pt_amps("neo-ccsd(t)-1p") == ["t3", "tep21"]   # epp needs 2 protons, ppp 3
    assert models.pt_amps("ccsd") == []
    assert models.model("neo-ccsd(t)").T == models.model("neo-ccsd").T  # CCSD cluster
    assert "t3" not in models.model("neo-ccsd(t)").T                    # NOT iterated
    for nm in ("ccsd(t)", "neo-ccsd(t)"):
        for amp in models.pt_amps(nm):
            assert amp in models.PROJECTION, (nm, amp)
    for bad in (lambda: models.pt_amplitude_graph("ccsd(t)", "tep21"),
                lambda: models.pt_energy_graph("ccsd", "t3"),
                lambda: models.pt_energy_graph("ccsd(t)", "tep12")):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass

    # --- symbolic: the canonical model IS examples/ccsd_t.py -------------------------
    def raw(left, calls):
        pq = pq_helper("fermi")
        pq.set_left_operators(left)
        for op, amp in calls:
            pq.add_commutator(1.0, [op], [amp])
        pq.simplify()
        return sorted(pq.strings())

    m = models.model("ccsd(t)")
    assert sorted(models._pt_pq(m, [[models.PROJECTION["t3"]]], m.T).strings()) == \
        raw([[models.PROJECTION["t3"]]], [("v", "t2")]), "ccsd(t) triples numerator"
    assert sorted(models._pt_pq(m, [["l1"], ["l2"]], ["t3"]).strings()) == \
        raw([["l1"], ["l2"]], [("v", "t3")]), "ccsd(t) (T) energy"

    # numeric: >=3 occupied protons, else the rank-3 proton block (ppp) is
    # identically zero and the checks below would be vacuous
    _check_pt_contract(("ccsd(t)", "neo-ccsd(t)-1p", "neo-ccsd(t)"), (3, 4, 3, 4))
    print("test_perturbative_triples OK")


def test_perturbative_quadruples():
    """CCSDT(Q) / NEO-CCSDT(Q): the same construction one rank up.

    ``(Q)`` is ``(T)`` with the CCSDT cluster and a rank-4 projection, so the consumer
    contract is identical and is checked by the same helper: the Fock coupling fixes the
    denominator, the numerator/energy pairing fixes the relative index order and sign,
    and together they force the correction negative. What is new here is the block set --
    eeee, eeep, eepp, eppp and pppp -- and the pairing weight, which is
    ``(n_e!)^2 (n_p!)^2`` and so reaches 576 for the pure-species blocks.

    The many-proton blocks need that many quantum protons to be nonzero. They are
    expected to be negligible for protons, but the machinery is not proton-specific --
    for a heavier or more numerous second species they need not be -- so the set is
    carried complete and a consumer gates each block on its own particle count. The
    ``-1p`` reduction does exactly that, keeping only eeee and eeep.
    """
    import math

    # --- structure -------------------------------------------------------------------
    assert models.pt_amps("ccsdt(q)") == ["t4"]
    assert models.pt_amps("neo-ccsdt(q)") == ["t4", "tep31", "tep22", "tep13", "tp4"]
    assert models.pt_amps("neo-ccsdt(q)-1p") == ["t4", "tep31"]   # >=2-proton blocks dropped
    # the base cluster is CCSDT: the quadruples are perturbative, never iterated
    assert models.model("ccsdt(q)").T == ("t1", "t2", "t3")
    assert models.model("neo-ccsdt(q)").T == models.model("neo-ccsdt").T
    for nm in ("ccsdt(q)", "neo-ccsdt(q)"):
        for amp in models.pt_amps(nm):
            assert amp in models.PROJECTION, (nm, amp)
            assert amp not in models.model(nm).T, (nm, amp)

    # the pairing weight is (n_e!)^2 (n_p!)^2 -- 576 for eeee/pppp, 36 for eeep/eppp,
    # 16 for eepp. spot-check it against the projection's index classes.
    expect_w = {"t4": 576, "tep31": 36, "tep22": 16, "tep13": 36, "tp4": 576}
    for amp, w in expect_w.items():
        np_ = models._proton_count(amp)
        ne = {"t4": 4, "tep31": 3, "tep22": 2, "tep13": 1, "tp4": 0}[amp]
        assert math.factorial(ne) ** 2 * math.factorial(np_) ** 2 == w, (amp, w)

    for bad in (lambda: models.pt_amplitude_graph("ccsdt(q)", "tep31"),
                lambda: models.pt_energy_graph("ccsdt", "t4"),
                lambda: models.pt_energy_graph("ccsdt(q)", "tp4")):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass

    # --- numeric: same consumer contract, rank-4 blocks -------------------------------
    # >=4 orbitals per species per space: a rank-4 antisymmetric block (eeee, pppp) is
    # identically zero below that, and the checks would be vacuous
    _check_pt_contract(("ccsdt(q)", "neo-ccsdt(q)-1p", "neo-ccsdt(q)"), (4, 5, 4, 5))
    print("test_perturbative_quadruples OK")



def test_equations_ir():
    """equations_ir emits the energy and every residual of a model in ONE pq_graph, so
    intermediates are shared across equations (cross-equation CSE). Verify (1) every
    target of the combined emission matches its per-equation opt0 ground truth
    numerically, and (2) the combined emission is no larger than the sum of the
    per-equation emissions (the sharing must not backfire)."""
    import re, itertools
    import numpy as np
    from collections import defaultdict

    DIM = {"o": 3, "v": 4, "O": 1, "V": 4, "Q": 6}
    VIR, OCC = {"v", "V"}, {"o", "O"}
    is_amp = lambda nm: re.fullmatch(r"t\d+(_n|_ep)?", nm) is not None

    def antisym(a, cl):                        # CC antisymmetry over same-class vir/occ axes
        out = a.copy()
        groups = defaultdict(list)
        for ax, c in enumerate(cl):
            groups[c].append(ax)
        for c, axes in groups.items():
            if len(axes) >= 2 and (c in VIR or c in OCC):
                perms = list(itertools.permutations(range(len(axes))))
                acc = np.zeros_like(out)
                for p in perms:
                    par = sum(1 for i in range(len(p)) for j in range(i + 1, len(p))
                              if p[i] > p[j]) & 1
                    src = list(range(out.ndim))
                    for k, ax in enumerate(axes):
                        src[ax] = axes[p[k]]
                    acc += (-1 if par else 1) * np.transpose(out, src)
                out = acc / len(perms)
        return out

    def interp(ir, inp):                       # evaluate an IR statement list -> {target: value}
        st = {}
        val = lambda o: st[o["name"]] if o["name"] in st else inp[o["name"]]
        for s in ir:
            subs = ",".join("".join(o["indices"]) for o in s["operands"])
            out = "".join(s["target"]["indices"])
            c = s["coeff"] * np.einsum(subs + "->" + out,
                                       *[val(o) for o in s["operands"]], optimize=True)
            t = s["target"]["name"]
            st[t] = c.copy() if s["is_assignment"] else st[t] + c
        return st

    for name in ("ccd", "neo-ccd(ep)"):
        m = models.model(name)

        # combined emission and per-equation opt0 ground truths
        comb = einsums.parse_ir(models.equations_ir(name))
        separate = {"energy": einsums.parse_ir(models.energy_graph(name, opt_level=0)
                                               .to_strings("ir"))}
        for amp in m.T:
            separate[f"R_{amp}"] = einsums.parse_ir(
                models.residual_graph(name, amp, opt_level=0).to_strings("ir"))

        # inputs: every external operand name (not produced by any statement), from the
        # union of the combined and separate emissions; amplitudes antisymmetrized
        produced = {s["target"]["name"] for ir in [comb, *separate.values()] for s in ir}
        names = {o["name"]: tuple(o["classes"])
                 for ir in [comb, *separate.values()] for s in ir for o in s["operands"]
                 if o["name"] not in produced}
        rng = np.random.default_rng(11)
        inp = {}
        for nm, cl in sorted(names.items()):
            if nm.startswith("Id["):
                # Kronecker delta from reference traces: must be a TRUE identity. The
                # optimizer reindexes through delta identities (sum_j Id(ij) X(j..) =
                # X(i..)), which only hold for the actual identity matrix -- a random
                # Id makes algebraically equal emissions evaluate differently.
                inp[nm] = np.eye(DIM[cl[0]])
                continue
            a = rng.standard_normal(tuple(DIM[c] for c in cl))
            inp[nm] = antisym(a, cl) if is_amp(nm) else a

        # (1) each combined target reproduces its per-equation opt0 ground truth
        st = interp(comb, inp)
        for tgt, ir0 in separate.items():
            truth = interp(ir0, inp)[tgt if tgt == "energy" else "R"]
            err = float(np.max(np.abs(st[tgt] - truth)))
            assert err < 1e-9, (name, tgt, err)

        # (2) sharing must not backfire: combined no larger than the sum of separate
        # default-opt emissions
        sep_default = sum(len(einsums.parse_ir(models.residual_ir(name, amp)))
                          for amp in m.T)
        sep_default += len(einsums.parse_ir(models.energy_graph(name).to_strings("ir")))
        assert len(comb) <= sep_default, (name, len(comb), sep_default)
    print("test_equations_ir OK")


if __name__ == "__main__":
    test_models_present_and_projected()
    test_single_proton_models()
    test_bad_lookups_raise()
    test_cheap_models_generate()
    test_spin_axis()
    test_lambda_and_gradient()
    test_lambda_consistency()
    test_rdm()
    test_energy_from_rdm()
    test_orbital_gradient_hessian()
    test_gradient_ir_matches_orbital_gradient()
    test_hessian_ir_matches_orbital_hessian()
    test_orbital_gradient_finite_difference()
    test_orbital_hessian_diag()
    test_orbital_sigma()
    test_orbital_gradient_active_space()
    test_orbital_diag_active_space()
    test_orbital_sigma_active_space()
    test_orbital_proton_gradient_active_space()
    test_orbital_cross_sigma_active_space()
    test_neo_gep_normal_ordered()
    test_opt_level_safe_default()
    test_rdm_block_ir()
    test_equations_ir()
    test_ir_pairing()
    test_perturbative_triples()
    print("\nall model tests passed")
