#!/usr/bin/env python
"""
Regression for species-aware spin-tracing of the multicomponent (NEO) equations.

For each residual we generate the spin-orbital equation, spin-block it (electrons
alpha/beta, the high-spin proton pinned to a single channel), and freeze the
resulting spin-blocked symbolic equations.  Re-running must reproduce the frozen
reference byte-for-byte, so any change to the spin-tracing machinery is caught.

Covers the NEO-CCSD residuals (electron singles/doubles, proton singles, the mixed
e-p double) AND the NEO-CCSDT(eep) mixed triple, so the higher-order ladder is
guarded from the start.

Run: python neo_spin_codegen.py            # check against neo_spin_traced.ref
     python neo_spin_codegen.py --write     # (re)generate the reference
"""
import os
import sys
import pdaggerq

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from extract_spins import get_spin_labels

H = ["f", "v", "fp", "gep"]          # single quantum proton: electron Fock+ee, proton Fock, e-p


def spin_traced(proj, T):
    """spin-orbital residual -> {spin_case: sorted list of spin-blocked term strings}."""
    pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([[proj]])
    for op in H:
        pq.add_st_operator(1.0, [op], T, True)
    pq.simplify()
    out = {}
    for name, m in get_spin_labels([[proj]], "high-spin").items():
        p = pq.clone(); p.block_by_spin(m)
        out[name] = sorted(" ".join(t) for t in p.strings())
    return out


def build():
    blocks = {}
    Tccsd = ["t1", "t2", "tp1", "tep11"]
    blocks["rt1_e"]  = spin_traced("e1(i,a)", Tccsd)            # electron singles
    blocks["rt2_e"]  = spin_traced("e2(i,j,a,b)", Tccsd)        # electron doubles
    blocks["rt1_p"]  = spin_traced("e1(ni,na)", Tccsd)          # proton singles
    blocks["rt2_ep"] = spin_traced("e2(i,ni,a,na)", Tccsd)      # mixed e-p double
    # NEO-CCSDT_eep (Pavosevic & Hammes-Schiffer, JCP 157, 074104 (2022), eq 6):
    # NEO-CCSD + the eep triple t^{abA}_{ijI} ONLY -- no electron t3. Single proton -> no tp2.
    Teep = ["t1", "t2", "tp1", "tep11", "tep21"]
    blocks["rt3_eep"] = spin_traced("e3(i,j,ni,a,b,na)", Teep)  # NEO-CCSDT_eep mixed triple
    return blocks


def serialize(blocks):
    lines = []
    for label in sorted(blocks):
        for spin in sorted(blocks[label]):
            terms = blocks[label][spin]
            lines.append("### %s [%s]  (%d terms)" % (label, spin, len(terms)))
            lines.extend(terms)
    return "\n".join(lines) + "\n"


def main():
    ref = os.path.join(os.path.dirname(os.path.realpath(__file__)), "neo_spin_traced.ref")
    text = serialize(build())
    if "--write" in sys.argv or not os.path.exists(ref):
        with open(ref, "w") as f:
            f.write(text)
        print("WROTE reference: neo_spin_traced.ref (%d lines)" % text.count("\n"))
        return
    with open(ref) as f:
        old = f.read()
    if old == text:
        print("PASS: spin-traced NEO equations match neo_spin_traced.ref")
    else:
        print("FAIL: spin-traced NEO equations differ from neo_spin_traced.ref")
        sys.exit(1)


if __name__ == "__main__":
    main()
