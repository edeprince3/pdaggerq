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


def test_energy_from_rdm():
    import itertools
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
    ref = np.einsum("pq,pq->", full["h"], full["D1"]) + 0.5 * np.einsum("pqsr,pqsr->", full["g"], full["D2"])
    assert abs(E - ref) < 1e-10, (E, ref)
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
    for t in pq.strings():
        c, ts = models._parse_rdm_term(" ".join(t))
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


def _active_space_H4(FULL, nmo):
    """Reference full electron Hessian H4[a,b,i,j] over nmo with active-only RDMs."""
    import pdaggerq
    import numpy as np
    pq = pdaggerq.pq_helper("true"); pq.set_use_rdms(True)
    for op, c in (("h", 1.0), ("g", 0.5)):
        for xg, sx in (("e1(a,i)", 1.0), ("e1(i,a)", -1.0)):
            for yg, sy in (("e1(b,j)", 1.0), ("e1(j,b)", -1.0)):
                pq.add_double_commutator(c * sx * sy, [op], [xg], [yg])
    pq.simplify()
    H4 = np.zeros((nmo,) * 4)
    for t in pq.strings():
        coeff, ts = models._parse_rdm_term(" ".join(t))
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

    # reference full Hessian H4 over nmo (active-only RDMs)
    pq = pdaggerq.pq_helper("true"); pq.set_use_rdms(True)
    for op, c in (("h", 1.0), ("g", 0.5)):
        for xg, sx in (("e1(a,i)", 1.0), ("e1(i,a)", -1.0)):
            for yg, sy in (("e1(b,j)", 1.0), ("e1(j,b)", -1.0)):
                pq.add_double_commutator(c * sx * sy, [op], [xg], [yg])
    pq.simplify()
    H4 = np.zeros((nmo,) * 4)
    for t in pq.strings():
        coeff, ts = models._parse_rdm_term(" ".join(t))
        arrs = [FULL["Id"] if nm == "d" else FULL[nm] for nm, _ in ts]
        H4 += coeff * np.einsum(",".join("".join(i) for _, i in ts) + "->abij", *arrs, optimize=True)

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
    F = {"g": gep, "gep": gep, "D2_ep": D2ep, "kappa_n": kp}
    Hep = _interp_terms(models._GEP_CROSS_HESS_TERMS, ["a", "nb", "i", "nj"], (nme, nmp, nme, nmp), F)
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
    # (3) the fusion fix holds: opt_level 6 is deterministic (identical text across
    # emissions) AND correct (matches opt0) on the equation that used to break.
    # If this fails, the fusion regression is back -- re-cap generation and see
    # edeprince3/pdaggerq#114.
    texts, worst = set(), 0.0
    for _ in range(3):
        ir6 = models.residual_ir("neo-ccsdt(eep)", "tep11", opt_level=6)
        texts.add("\n".join(l for l in ir6 if l.strip().startswith("{")))
        worst = max(worst, float(np.max(np.abs(interp(einsums.parse_ir(ir6), inp) - truth))))
    assert len(texts) == 1, f"opt6 emission nondeterministic again ({len(texts)} variants)"
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

    blocks = [("D1", "oo"), ("D1", "ov"), ("D1", "vv"),
              ("D2", "oovv"), ("D2", "ovvo"), ("D2", "vvoo"),
              ("D1_n", "OO"), ("D1_n", "OV"), ("D1_n", "VV")]
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
        perm = [list(final["indices"]).index(L) for L, _ in consumer]
        assert np.max(np.abs(blk - np.transpose(native, perm))) < 1e-10, (t, b)

    # (3) unpopulatable block -> empty (consumer zero-fills)
    assert models.rdm_block_ir("neo-ccd(ep)", "D1", "ov") == []
    print("test_rdm_block_ir OK")


if __name__ == "__main__":
    test_models_present_and_projected()
    test_single_proton_models()
    test_bad_lookups_raise()
    test_cheap_models_generate()
    test_spin_axis()
    test_lambda_and_gradient()
    test_rdm()
    test_energy_from_rdm()
    test_orbital_gradient_hessian()
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
    print("\nall model tests passed")
