#!/usr/bin/env python
"""
Symbolic tests for the multicomponent (nuclear-electronic / NEO) lambda
amplitudes and reduced density matrices.

  - nuclear / mixed lambda (de-excitation) amplitudes  lp#, lep##
  - species-distinguished RDM names  D#_n (nuclear), D#_ep (mixed)
  - species-aware cumulant expansion of the mixed two-particle RDM

Run: python neo_lambda_rdm_test.py
"""
import re
import pdaggerq


def terms(pq):
    return [" ".join(t) for t in pq.strings()]


def strip_nuclear(s):
    # map nuclear labels/names to their electron counterparts for structural comparison
    return sorted(re.sub(r"_n\b", "", re.sub(r"\bn([a-z])", r"\1", x)) for x in s)


def test_nuclear_lambda_parity():
    # the nuclear lambda doubles residual must equal the electron lambda doubles
    # residual after relabeling (vp == v, lp2 == l2)
    pe = pdaggerq.pq_helper("fermi"); pe.set_left_operators([["l2"]])
    pe.add_operator_product(1.0, ["v"]); pe.simplify()
    pn = pdaggerq.pq_helper("fermi"); pn.set_left_operators([["lp2"]])
    pn.add_operator_product(1.0, ["vp"]); pn.simplify()
    assert strip_nuclear(terms(pe)) == strip_nuclear(terms(pn)), "nuclear lambda != electron lambda"
    print("OK  nuclear lambda doubles == electron lambda doubles (relabeled)")


def test_mixed_lambda():
    # the mixed lambda l2_ep contracts with the e-p operator gep
    pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([["lep11"]])
    pq.add_operator_product(1.0, ["gep"]); pq.simplify()
    joined = " ".join(terms(pq))
    assert "l2_ep(" in joined, "mixed lambda l2_ep not produced"
    print("OK  mixed lambda l2_ep:", terms(pq)[0])


def test_rdm_species_names():
    # nuclear and mixed RDMs carry distinct names
    def rdms(op):
        pq = pdaggerq.pq_helper("true"); pq.set_use_rdms(True); pq.set_left_operators([["1"]])
        pq.add_operator_product(1.0, [op]); pq.simplify()
        return sorted(set(re.findall(r"D[0-9]_?[a-z]*", " ".join(terms(pq)))))
    assert rdms("v")   == ["D1", "D2"],       rdms("v")
    assert rdms("vp")  == ["D1_n", "D2_n"],   rdms("vp")
    assert rdms("gep") == ["D2_ep"],          rdms("gep")
    print("OK  RDM names: electron D1/D2, nuclear D1_n/D2_n, mixed D2_ep")


def test_rdm_energy_formula():
    # the energy expressed in RDMs must mirror across species:
    #   v   -> -<p,i||q,i> D1   + 1/4 <p,q||s,r> D2
    #   vp  -> -<nP,nI||nQ,nI> D1_n + 1/4 <nP,nQ||nS,nR> D2_n   (electron formula, relabeled)
    #   gep -> g(p,nP,q,nQ) D2_ep(nP,p,q,nQ)                    (single mixed 2-RDM contraction)
    # This is the structure behind the numerical check that the multicomponent RDMs
    # reconstruct the energy obtained from the t-amplitudes (electron/nuclear/mixed all consistent).
    def erdm(op):
        pq = pdaggerq.pq_helper("true"); pq.set_use_rdms(True); pq.set_left_operators([["1"]])
        pq.add_operator_product(1.0, [op]); pq.simplify()
        return terms(pq)
    # nuclear two-body energy mirrors the electron two-body energy after relabeling
    assert strip_nuclear(erdm("vp")) == strip_nuclear(erdm("v")), (erdm("vp"), erdm("v"))
    # mixed energy is a single contraction of gep with the mixed 2-RDM D2_ep (no D1 term)
    g = erdm("gep")
    assert len(g) == 1 and "D2_ep(" in g[0] and "D1" not in g[0], g
    print("OK  energy-in-RDMs: vp mirrors v (D1_n + D2_n); gep = g . D2_ep")


def test_mixed_cumulant_no_cross_species_exchange():
    # cumulant-expanding the mixed 2-RDM must give D1(electron) * D1_n(nuclear) only,
    # with no spurious cross-species exchange (no mixed-species one-particle RDM)
    pq = pdaggerq.pq_helper("true"); pq.set_use_rdms(True, [2]); pq.set_left_operators([["1"]])
    pq.add_operator_product(1.0, ["gep"]); pq.simplify()
    joined = " ".join(terms(pq))
    assert "D1_ep(" not in joined, "spurious cross-species exchange term present"
    assert "D1(" in joined and "D1_n(" in joined, "expected D1 * D1_n product"
    print("OK  mixed 2-RDM cumulant = D1 * D1_n (no cross-species exchange)")


if __name__ == "__main__":
    test_nuclear_lambda_parity()
    test_mixed_lambda()
    test_rdm_species_names()
    test_rdm_energy_formula()
    test_mixed_cumulant_no_cross_species_exchange()
    print("PASS: NEO lambda amplitudes and density matrices")
