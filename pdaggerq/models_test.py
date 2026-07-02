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
    # NEO models carry the proton + cross-species Hamiltonian
    assert models.model("neo-ccsd").H == ("f", "v", "fp", "gep")
    assert models.model("ccsd").H == ("f", "v")
    print("test_models_present_and_projected OK")


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
    assert models.lambda_amps("neo-ccsd") == ["l1", "l2", "lp1", "lep11"]
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
    # electron Hessian: rank-4 vir-vir-occ-occ block
    hee = einsums.parse_ir(models.orbital_hessian_ir("ccsd"))
    assert einsums.target_shape(hee, "H") == (4, ["v", "v", "o", "o"])
    # NEO OO (cross-species gep) and proton rows are not yet supported (raise cleanly)
    for bad, exc in ((lambda: models.orbital_gradient_ir("ccsd", "muon"), ValueError),
                     (lambda: models.orbital_gradient_ir("neo-ccd(ep)", "electron"), NotImplementedError),
                     (lambda: models.orbital_gradient_ir("neo-ccd(ep)", "proton"), NotImplementedError),
                     (lambda: models.orbital_hessian_ir("neo-ccd(ep)", "electron", "proton"), NotImplementedError)):
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


if __name__ == "__main__":
    test_models_present_and_projected()
    test_bad_lookups_raise()
    test_cheap_models_generate()
    test_spin_axis()
    test_lambda_and_gradient()
    test_rdm()
    test_energy_from_rdm()
    test_orbital_gradient_hessian()
    test_orbital_hessian_diag()
    print("\nall model tests passed")
