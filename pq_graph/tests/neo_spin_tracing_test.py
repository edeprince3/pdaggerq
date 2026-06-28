#!/usr/bin/env python
"""
Tests for species-aware spin-tracing (spin-blocking) of multicomponent (NEO) equations.

Electrons spin-block into the usual alpha/beta cases; nuclear ('n'-prefix) labels are a
separate spin species: "high-spin" pins them to one channel (single quantum proton /
positron, closed-shell-electron default), "full" gives them their own alpha/beta manifold
(>=2 quantum nuclei, where the opposite-spin nuclear doubles are the pairing channel).

Run: python neo_spin_tracing_test.py
"""
import pdaggerq
from extract_spins import get_spin_labels


def test_electron_only_unchanged():
    m = get_spin_labels([['e2(i,j,a,b)']])
    assert list(m.keys()) == ['aaaa', 'abab', 'bbbb'], m
    assert m['abab'] == {'a': 'a', 'b': 'b', 'i': 'a', 'j': 'b'}, m['abab']
    print("OK  electron-only spin-blocking is unchanged")


def test_highspin_nuclear_single_channel():
    m = get_spin_labels([['e2(i,ni,a,na)']], 'high-spin')   # mixed e-p doubles: 2 e + 2 n labels
    assert list(m.keys()) == ['aa_n', 'bb_n'], m
    assert all(m['aa_n'][l] == 'a' for l in ('ni', 'na')), "nuclear not pinned to one channel"
    print("OK  high-spin proton -> single nuclear spin channel")


def test_full_nuclear_pairing_channel():
    m = get_spin_labels([['e2(ni,nj,na,nb)']], 'full')      # nuclear doubles, two quantum nuclei
    assert '_nabab' in m, m
    assert m['_nabab']['na'] == 'a' and m['_nabab']['nb'] == 'b', m['_nabab']
    print("OK  full nuclear alpha/beta manifold exposes the pairing channel (_nabab)")


def test_codegen_keeps_species_distinct():
    # spin-blocking renames t2_ep -> "t2_aaaa" in the symbolic string, but pq_graph codegen
    # must keep electron doubles (t2) and the mixed amplitude (t2_ep) as DISTINCT tensors,
    # with the spin as a block key -- else the spin-traced code conflates species.
    import re
    pq = pdaggerq.pq_helper("fermi"); pq.set_left_operators([["e2(i,ni,a,na)"]])
    pq.add_st_operator(1.0, ["gep"], ["t2", "tep11"], True, 2); pq.simplify()
    pq.block_by_spin(get_spin_labels([["e2(i,ni,a,na)"]], "high-spin")["aa_n"])
    g = pdaggerq.pq_graph({"opt_level": 0}); g.add(pq, "R"); g.optimize()
    code = "\n".join(g.to_strings("python"))
    assert re.search(r't2\["(aaaa|abab)"\]', code), "electron doubles t2 block missing"
    assert re.search(r't2_ep\["(aaaa|abab)"\]', code), "mixed t2_ep block missing / conflated with t2"
    print("OK  codegen keeps electron t2 and mixed t2_ep distinct (spin as block key)")


if __name__ == "__main__":
    test_electron_only_unchanged()
    test_highspin_nuclear_single_channel()
    test_full_nuclear_pairing_channel()
    test_codegen_keeps_species_distinct()
    print("PASS: NEO species-aware spin-tracing")
