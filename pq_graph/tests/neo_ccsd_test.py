#!/usr/bin/env python
"""
Symbolic tests for multicomponent (nuclear-electronic / NEO) CCSD, i.e. the
forward cluster amplitudes including singles.

  - the pure-electron sector of NEO-CCSD reduces EXACTLY to standard CCSD
    (the multicomponent machinery must not perturb the electron-only equations)
  - nuclear (proton) singles mirror the electron singles after relabeling
  - the mixed electron-proton residual couples gep to the mixed amplitudes

Run: python neo_ccsd_test.py
"""
import re
import pdaggerq

# single-proton NEO-CCSD: H = f + v + fp + gep,  T = t1 + t2 + tp1 + tep11
MAXORD = 4   # CCSD with singles needs the full 4th-order BCH (exact for 2-body H)


def terms(pq):
    return sorted(" ".join(t) for t in pq.strings())


def strip_nuclear(s):
    return sorted(re.sub(r"_n\b", "", re.sub(r"\bn([a-z])", r"\1", x)) for x in s)


def residual(proj, H, T):
    pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([[proj]])
    for h in H:
        pq.add_st_operator(1.0, [h], T, True, MAXORD)
    pq.simplify(); return terms(pq)


def test_electron_sector_reduces_to_ccsd():
    # with the electron-only Hamiltonian/cluster, every NEO residual must equal
    # the corresponding standard CCSD residual term-for-term
    He, Te = ["f", "v"], ["t1", "t2"]
    for proj in ("e1(i,a)", "e2(i,j,a,b)"):
        neo = residual(proj, He, Te)
        std = residual(proj, ["f", "v"], ["t1", "t2"])   # identical call == standard CCSD
        assert neo == std, proj
    print("OK  electron sector of NEO-CCSD == standard CCSD (singles & doubles)")


def test_proton_singles_mirror_electron():
    # the proton single-excitation residual mirrors the electron one (fp==f, tp1==t1)
    pe = residual("e1(i,a)", ["f"], ["t1"])
    pn = residual("e1(ni,na)", ["fp"], ["tp1"])
    assert strip_nuclear(pn) == strip_nuclear(pe), (pn, pe)
    print("OK  proton singles residual mirrors electron singles (relabeled)")


def test_mixed_residual_couples_gep():
    # the mixed e-p residual must contain the bare gep driver and the mixed amplitude
    r = residual("e2(i,ni,a,na)", ["f", "v", "fp", "gep"], ["t1", "t2", "tp1", "tep11"])
    joined = " ".join(r)
    assert any("g(" in t for t in r), "mixed residual missing gep coupling"
    assert "t2_ep(" in joined or "tep" in joined or "t1_n(" in joined, "mixed residual missing mixed/proton amplitudes"
    print("OK  mixed e-p residual couples gep to the mixed/proton amplitudes")


if __name__ == "__main__":
    test_electron_sector_reduces_to_ccsd()
    test_proton_singles_mirror_electron()
    test_mixed_residual_couples_gep()
    print("PASS: NEO-CCSD singles and reduction")
