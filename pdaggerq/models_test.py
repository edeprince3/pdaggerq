"""Tests for the CC model library (pdaggerq.models).

Structural checks over every model are exhaustive and instant; generation is
exercised only on the cheap models (the high-rank residuals -- ccsdt/ccsdtq and
the full/hybrid NEO triples/quadruples -- are correct but slow to build, so they
are covered by the dedicated examples, not here). Run: python -m pdaggerq.models_test
"""

from pdaggerq import einsums, models


def test_models_present_and_projected():
    expected = {
        "ccd", "ccsd", "ccsdt", "ccsdtq",
        "neo-ccd", "neo-ccsd", "neo-ccsdt", "neo-ccsdtq",
        "neo-ccd(ep)", "neo-ccsdt(eep)", "neo-ccsdtq(eeep)",
    }
    assert expected <= set(models.MODELS), expected - set(models.MODELS)
    # every amplitude of every model has a conjugate projection
    for m in models.MODELS.values():
        for amp in m.T:
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
    """The orbital gradient (and Hessian) must be the EXACT derivative of the fixed-RDM
    energy that energy_from_rdm_ir traces -- as an algebraic identity in the integrals,
    hence for BOTH signs of gep (a consumer chooses gep's charge sign; see the charge
    convention in the module docstring). Verified against finite differences of the
    energy with the integrals rotated by exp(kappa).

    This is the test that was missing: the block-structure tests below only check
    block-sum == full-contraction, so they could not see that the two-body terms were
    contracting D2 in pq_helper's slot order instead of the consumer's. D2 is
    antisymmetric in exactly those slots, so the electron g.D2 gradient came out as
    MINUS the true derivative (ratio -1.00000000) while h.D1 was right -- see
    models._d2_to_consumer. The proton row was always correct (hand-derived from h_p and
    gep; it never touches pq_helper's D2), including its gep sign tracking."""
    import numpy as np

    ref = models.rdm_energy_reference("neo-ccd(ep)", seed=17)
    d = ref["dims"]; no, nv, nO, nV = d["o"], d["v"], d["O"], d["V"]
    ne, npr = no + nv, nO + nV
    SL = {"o": slice(0, no), "v": slice(no, ne), "O": slice(0, nO), "V": slice(nO, npr)}

    def assemble(base, shape):
        A = np.zeros(shape)
        for (b, blk), arr in ref["rdm"].items():
            if b == base:
                A[tuple(SL[c] for c in blk)] = arr
        return A

    D1 = assemble("D1", (ne, ne)); D2 = assemble("D2", (ne,) * 4)
    D1n = assemble("D1_n", (npr, npr)); D2ep = assemble("D2_ep", (npr, ne, ne, npr))
    h0, g0, hp0 = ref["h"], ref["g"], ref["hp"]

    def energy(h, g, hp, gep):                       # the energy_from_rdm contract
        return float(np.einsum("pq,pq->", h, D1)
                     + 0.5 * np.einsum("abcd,abdc->", g, D2)
                     + np.einsum("PQ,PQ->", hp, D1n)
                     + np.einsum("ePfQ,PefQ->", gep, D2ep))

    # the assembled trace must reproduce the library's own energy (pins the pairings)
    assert abs(energy(h0, g0, hp0, ref["gep"]) - ref["E_rdm"]) < 1e-10

    def expm(A):                                     # A is tiny & antisymmetric
        R = np.eye(A.shape[0]); T = np.eye(A.shape[0])
        for k in range(1, 18):
            T = T @ A / k
            R = R + T
        return R

    def rotated(species, t, a, i, h, g, hp, gep):
        """Energy with the `species` orbitals rotated by kappa[vir a, occ i] = t."""
        nocc, n = (nO, npr) if species == "proton" else (no, ne)
        K = np.zeros((n, n)); K[nocc + a, i] = t; K[i, nocc + a] = -t
        U = expm(K)
        if species == "proton":
            return energy(h, g, U.T @ hp @ U,
                          np.einsum("ePfQ,PA,QB->eAfB", gep, U, U))
        return energy(U.T @ h @ U,
                      np.einsum("pqrs,pA,qB,rC,sD->ABCD", g, U, U, U, U),
                      hp, np.einsum("ePfQ,eA,fB->APBQ", gep, U, U))

    def emitted(fn, species, h, g, hp, gep, **kw):
        ir = einsums.parse_ir(fn("neo-ccd(ep)", species, **kw))
        FULL = {"h": h, "g": g, "hp": hp, "gep": gep,
                "D1": D1, "D2": D2, "D1_n": D1n, "D2_ep": D2ep}
        return _interp_blocks(ir, FULL, SL)

    eps = 1e-5
    for sign in (+1.0, -1.0):                        # gep's charge sign must not matter
        gep = sign * ref["gep"]
        for species in ("electron", "proton"):
            nocc, nvir = (nO, nV) if species == "proton" else (no, nv)
            fd = np.zeros((nvir, nocc))
            for a in range(nvir):
                for i in range(nocc):
                    fd[a, i] = (rotated(species, +eps, a, i, h0, g0, hp0, gep)
                                - rotated(species, -eps, a, i, h0, g0, hp0, gep)) / (2 * eps)
            (_, ana), = emitted(models.orbital_gradient_ir, species,
                                h0, g0, hp0, gep).items()
            err = float(np.max(np.abs(ana - fd)))
            assert err / max(float(np.max(np.abs(fd))), 1e-30) < 1e-6, \
                (species, "gep sign", sign, err, ana, fd)

    # ---- SECOND derivative: Hessian (all three blocks), its diagonal, and sigma -------
    # These used to come from pq_helper's DOUBLE commutator, whose two-body piece is wrong
    # (the single commutator is fine) and which emitted the UNSYMMETRIZED <[[H,A],B]> --
    # not symmetric under (a,i)<->(b,j), while the true fixed-RDM Hessian is exactly
    # symmetric. They are now rotation-derivatives of the same energy as the gradient
    # (models._same_species_hessian / _cross_hessian_terms). Check every one against FD.
    ZE, ZP = np.zeros((ne, ne)), np.zeros((npr, npr))

    def kmat(sp, a, i, t):
        n, nocc = (npr, nO) if sp == "p" else (ne, no)
        A = np.zeros((n, n))
        A[nocc + a, i] += t; A[i, nocc + a] -= t
        return A

    def energy_rot(ke, kp):
        Ue, Up = expm(ke), expm(kp)
        return energy(Ue.T @ h0 @ Ue,
                      np.einsum("pqrs,pA,qB,rC,sD->ABCD", g0, Ue, Ue, Ue, Ue),
                      Up.T @ hp0 @ Up,
                      np.einsum("ePfQ,eA,fB,PC,QD->ACBD", gep, Ue, Ue, Up, Up))

    def fd_hess(rs, cs):
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
                            return energy_rot(ke, kp)
                        H[a, i, b, j] = (E2(eps, eps) - E2(eps, -eps)
                                         - E2(-eps, eps) + E2(-eps, -eps)) / (4 * eps * eps)
        return H

    gep = ref["gep"]
    FULL = {"h": h0, "g": g0, "hp": hp0, "gep": gep, "D1": D1, "D2": D2,
            "D1_n": D1n, "D2_ep": D2ep, "Id": np.eye(max(ne, npr))}

    def ev(ir, extra=None):
        F = dict(FULL, **(extra or {}))
        by = {}
        for st in ir:
            by.setdefault(st["target"]["name"], []).append(st)
        (_, sts), = by.items()
        acc = np.zeros(tuple(SL[c].stop - SL[c].start for c in sts[0]["target"]["classes"]))
        for st in sts:
            arrs = [F[o["name"].split('["')[0]][
                        tuple(SL[c] for c in o["name"].split('["')[1].rstrip('"]'))]
                    if '["' in o["name"] else F[o["name"]] for o in st["operands"]]
            subs = ",".join("".join(o["indices"]) for o in st["operands"])
            acc += st["coeff"] * np.einsum(
                f"{subs}->{''.join(st['target']['indices'])}", *arrs, optimize=True)
        return acc

    H_ee, H_pp, H_ep = fd_hess("e", "e"), fd_hess("p", "p"), fd_hess("e", "p")
    for tag, rs, cs, FD in (("ee", "electron", "electron", H_ee),
                            ("pp", "proton", "proton", H_pp),
                            ("ep", "electron", "proton", H_ep)):
        M = ev(einsums.parse_ir(models.orbital_hessian_ir(
            "neo-ccd(ep)", row_species=rs, col_species=cs))).transpose(0, 2, 1, 3)
        err = float(np.max(np.abs(M - FD))) / max(float(np.max(np.abs(FD))), 1e-30)
        assert err < 1e-5, ("hessian", tag, err)
        # the same-species Hessian must be exactly symmetric under (a,i)<->(b,j)
        if rs == cs:
            assert np.max(np.abs(M - M.transpose(2, 3, 0, 1))) / \
                   max(float(np.max(np.abs(M))), 1e-30) < 1e-10, ("hessian not symmetric", tag)

    for sp, FD in (("electron", H_ee), ("proton", H_pp)):
        dg = ev(einsums.parse_ir(models.orbital_hessian_diag_ir("neo-ccd(ep)", sp)))
        d_fd = np.einsum("aiai->ai", FD)
        err = float(np.max(np.abs(dg - d_fd))) / max(float(np.max(np.abs(d_fd))), 1e-30)
        assert err < 1e-5, ("hessian diag", sp, err)

    rng2 = np.random.default_rng(9)
    tr_e = rng2.standard_normal((nv, no)); tr_p = rng2.standard_normal((nV, nO))
    Ke = np.zeros((ne, ne)); Ke[no:, :no] = tr_e
    Kp = np.zeros((npr, npr)); Kp[nO:, :nO] = tr_p
    # sigma carries the CROSS coupling too: s^e = H^ee.k^e + H^ep.k^p ; s^p = H^pp.k^p + (H^ep)^T.k^e
    for sp, FD in (("electron", np.einsum("aibj,bj->ai", H_ee, tr_e)
                                + np.einsum("aiBJ,BJ->ai", H_ep, tr_p)),
                   ("proton",   np.einsum("AIBJ,BJ->AI", H_pp, tr_p)
                                + np.einsum("aiAI,ai->AI", H_ep, tr_e))):
        sg = ev(einsums.parse_ir(models.orbital_sigma_ir("neo-ccd(ep)", sp)),
                {"kappa": Ke, "kappa_n": Kp})
        err = float(np.max(np.abs(sg - FD))) / max(float(np.max(np.abs(FD))), 1e-30)
        assert err < 1e-5, ("sigma", sp, err)

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

    # reference generalized-Fock gradient (full sums, active-only RDMs)
    import pdaggerq
    pq = pdaggerq.pq_helper("true"); pq.set_use_rdms(True)
    for xg, s in (("e1(a,i)", 1.0), ("e1(i,a)", -1.0)):
        pq.add_commutator(s, ["h"], [xg]); pq.add_commutator(0.5 * s, ["g"], [xg])
    pq.simplify()
    Gref = np.zeros((nmo, nmo))
    # NOTE: the reference must use the CONSUMER's D2 pairing, exactly as the emitter
    # does (models._d2_to_consumer). Without this the reference is built from the same
    # raw pq_helper D2 slot order the emitter used, so this test compared the emitter
    # against its own source rather than against physics -- which is precisely how the
    # two-body sign error survived. The physics is guarded by
    # test_orbital_gradient_finite_difference.
    for t in models._d2_to_consumer(" ".join(x) for x in pq.strings()):
        c, ts = models._parse_rdm_term(t)
        Gref += c * np.einsum(",".join("".join(i) for _, i in ts) + "->ai", *[FULL[n] for n, _ in ts], optimize=True)

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
    F = {"g": gep, "h": hp, "D1": D1n, "D2_ep": D2ep, "gep": gep, "hp": hp, "D1_n": D1n}
    Gp = _interp_terms(models._HP_GRAD_TERMS + models._GEP_PROTON_GRAD_TERMS, ["na", "ni"], (npp, npp), F)
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

    for mdl in ("neo-ccd(ep)", "neo-ccsd", "neo-ccsdt(eep)"):
        rng2 = np.random.default_rng(7)
        gep = rng2.standard_normal((ne, npr, ne, npr))
        amp = {}                                          # random tensor per amplitude name

        def getamp(name, classes):
            classes = tuple(classes)
            if name not in amp:
                amp[name] = rng2.standard_normal(tuple(DIM[c] for c in classes))
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
    print("\nall model tests passed")
