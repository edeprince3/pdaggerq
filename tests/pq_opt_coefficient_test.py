# -*- coding: utf-8 -*-
"""
Coefficient snapping in pq_opt.

pdaggerq computes coefficients in floating point, so an exact fraction such as 1
or 1/24 can arrive a few ulps off. pq_opt replaces each coefficient by the
simplest nearby fraction (see snap_to_fraction in pq_opt/ingest.cc) and prints
it so that it reads back as exactly that double. These tests feed perturbed
fractions (including unusual ones from Bernoulli-number expansions) through
pdaggerq and pq_opt and check the printed coefficient, and check that values
that are not simple fractions are left unchanged.
"""

import math
import re
from fractions import Fraction

import numpy as np
import pdaggerq
import pytest

FRACTIONS = [Fraction(n, d) for n, d in (
    (1, 1), (1, 2), (1, 3), (2, 3), (1, 6), (1, 12), (1, 24), (1, 120), (1, 720), (1, 5040),
    (-1, 30), (1, 42), (5, 66), (-691, 2730), (7, 6), (-3617, 510), (3, 8), (11, 24), (-13, 720),
    (1, 30240), (1, 1209600), (1, 47900160), (5, 4), (17, 3), (-1, 1),
)]

# values that are not simple fractions; they must come out unchanged
NOT_FRACTIONS = [math.pi / 10, math.sqrt(2) / 7, 0.1234567890123, math.e / 100, -1.0 / math.pi]


def ulps(x, n):
    """x moved by n units in the last place"""
    for _ in range(abs(n)):
        x = np.nextafter(x, np.inf if n > 0 else -np.inf)
    return float(x)


def printed_coefficients(values):
    """the coefficient pq_opt prints for f(a,i) scaled by each value"""
    g = pdaggerq.pq_opt({'print_comments': False})
    for n, value in enumerate(values):
        pq = pdaggerq.pq_helper('fermi')
        pq.set_left_operators([['e1(i,a)']])
        pq.add_operator_product(value, ['f'])
        pq.simplify()
        g.add(pq, f'r{n}', ['a', 'i'])

    coefficients = []
    for line in g.to_strings('python'):
        match = re.match(r'\s*r\d+ = (\S+) \* f\["vo"\]', line)
        if match:
            coefficients.append(float(match.group(1)))
    assert len(coefficients) == len(values)
    return coefficients


@pytest.mark.parametrize("shift", (0, 1, -1, 2, -2, 4, -4))
def test_perturbed_fractions_snap_to_exact(shift):
    exact = [f.numerator / f.denominator for f in FRACTIONS]
    printed = printed_coefficients([ulps(x, shift) for x in exact])
    for f, x, p in zip(FRACTIONS, exact, printed):
        assert p == x, f"{f}: shifted by {shift} ulp, printed {p!r}, expected {x!r}"


@pytest.mark.parametrize("relative", (2e-12, -2e-12))
def test_large_roundoff_snaps(relative):
    # 10x the worst roundoff measured in QUCCSD coefficients (1.8e-13)
    exact = [f.numerator / f.denominator for f in FRACTIONS]
    printed = printed_coefficients([x * (1 + relative) for x in exact])
    for f, x, p in zip(FRACTIONS, exact, printed):
        assert p == x, f"{f}: relative error {relative}, printed {p!r}, expected {x!r}"


def test_roundoff_from_products_snaps():
    # the kind of roundoff pdaggerq produces: products and sums that should be exact
    values = [(1 / 6) * 3 * 2, (1 / 24) * 12, 0.1 * 3, 1 / 3 + 1 / 3 + 1 / 3, (2 / 3) * (3 / 4)]
    exact = [1.0, 0.5, 0.3, 1.0, 0.5]
    assert printed_coefficients(values) == exact


def test_non_fractions_unchanged():
    assert printed_coefficients(NOT_FRACTIONS) == NOT_FRACTIONS


if __name__ == "__main__":
    print("Please use pytest to run the tests")
