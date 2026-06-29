"""Tests for the CC model library (pdaggerq.models).

Structural checks over every model are exhaustive and instant; generation is
exercised only on the cheap models (the high-rank residuals -- ccsdt/ccsdtq and
the full/hybrid NEO triples/quadruples -- are correct but slow to build, so they
are covered by the dedicated examples, not here). Run: python -m pdaggerq.models_test
"""

from pdaggerq import models


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


if __name__ == "__main__":
    test_models_present_and_projected()
    test_bad_lookups_raise()
    test_cheap_models_generate()
    test_spin_axis()
    print("\nall model tests passed")
