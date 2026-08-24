#!/usr/bin/env python
"""
Regression test for lep/leb (mixed electron-nuclear / electron-boson left-hand
amplitudes) under IP/EA/DIP/DEA left_operators_type.

lep/leb serve two roles: the ground-state lambda amplitude (left_operators_type
"EE", the default) AND the left EOM amplitude for IP/EA/DIP/DEA variants -- the
same dual role that bare "l" already has (see the IP/EA/DIP/DEA rank adjustment
at the top of its branch in pq_helper.cc). lep/leb were originally built as
plain de-excitation mirrors of tep/teb with no such adjustment, unlike yep/yeb
(the analogous right-hand-side-of-the-bra-ket amplitudes for the *left* EOM
operator X/Y pair), which had it correctly from the start.

Because no NEO or QED IP/EA/DIP/DEA EOMCC test existed, this went undetected:
lep/leb silently ignored left_operators_type and always built the full,
EE-shaped (symmetric) rank, rather than the reduced/asymmetric rank IP/EA/DIP/DEA
require.

Two complementary checks are used:

  1. lep/leb must always produce the SAME result as the already-correct
     yep/yeb, for every left_operators_type, when paired against the same
     probe. rep/reb never change shape with left_operators_type (R has no
     excitation/de-excitation split), so pairing a lep/leb bra against a
     rep/reb ket isolates exactly the IP/EA/DIP/DEA rank adjustment: pre-fix,
     lep/leb ignored left_operators_type and stayed shape-matched with
     rep/reb (spuriously nonzero for IP/EA/DIP/DEA); post-fix, they correctly
     go to the same (empty) result as yep/yeb.

  2. leb, directly probed with an explicit right-hand operator of the rank
     IP/EA/DIP/DEA actually require, produces the expected nonzero,
     single-amplitude result -- confirming the adjustment is not just
     "different from before" but the *correct* reduced rank.

Run: python left_operators_type_neo_qed_test.py
"""
import pytest
import pdaggerq

LEFT_TYPES = ["EE", "IP", "EA", "DIP", "DEA"]


def probe(bra_op, left_type, ket_ops):
    pq = pdaggerq.pq_helper("fermi")
    pq.set_left_operators_type(left_type)
    pq.set_left_operators([[bra_op]])
    pq.add_operator_product(1.0, ket_ops)
    pq.simplify()
    return sorted(" ".join(t) for t in pq.strings())


def strip_amplitude_letter(strings, letter, replacement):
    """Normalize a leading amplitude-name letter (e.g. 'l1_1p' -> 'y1_1p') so
    that two amplitudes differing only by name compare equal."""
    return sorted(s.replace(f"{letter}1_1p", f"{replacement}1_1p")
                   .replace(f"{letter}2_1p", f"{replacement}2_1p")
                   .replace(f"{letter}1_n", f"{replacement}1_n")
                   .replace(f"{letter}2_ep", f"{replacement}2_ep")
                  for s in strings)


@pytest.mark.parametrize("left_type", LEFT_TYPES)
def test_leb_matches_yeb_paired_with_reb(left_type):
    leb = strip_amplitude_letter(probe("leb11", left_type, ["reb11"]), "l", "y")
    yeb = strip_amplitude_letter(probe("yeb11", left_type, ["reb11"]), "y", "y")
    assert leb == yeb
    # EE is the only shape-matched (symmetric) case against reb11
    if left_type == "EE":
        assert len(leb) > 0
    else:
        assert leb == []


@pytest.mark.parametrize("left_type", LEFT_TYPES)
def test_lep_matches_yep_paired_with_rep(left_type):
    lep = strip_amplitude_letter(probe("lep11", left_type, ["rep11"]), "l", "y")
    yep = strip_amplitude_letter(probe("yep11", left_type, ["rep11"]), "y", "y")
    assert lep == yep
    if left_type == "EE":
        assert len(lep) > 0
    else:
        assert lep == []


def test_leb_ip_reduces_to_single_occupied_index():
    leb = probe("leb11", "IP", ["a(i)", "B+"])
    assert leb == ["+1.000 l1_1p(i)"]


def test_leb_ea_reduces_to_single_virtual_index():
    leb = probe("leb11", "EA", ["a*(a)", "B+"])
    assert leb == ["+1.000 l1_1p(a)"]


def test_leb_dip_reduces_to_two_occupied_indices():
    leb = probe("leb21", "DIP", ["a(i)", "a(j)", "B+"])
    assert leb == ["+1.000 l2_1p(j,i)"]


def test_leb_dea_reduces_to_two_virtual_indices():
    leb = probe("leb21", "DEA", ["a*(a)", "a*(b)", "B+"])
    assert leb == ["+1.000 l2_1p(a,b)"]


if __name__ == "__main__":
    print("Please use pytest to run the tests")
    print("Syntax: python -m pytest left_operators_type_neo_qed_test.py")
    import sys
    sys.exit(1)
