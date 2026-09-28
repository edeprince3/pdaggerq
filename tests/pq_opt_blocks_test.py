# -*- coding: utf-8 -*-
"""
Spin blocks and paired permutations in pq_opt, checked against spin-orbital equations.

pdaggerq can write one residual several ways: with plain permutation operators
P(i,j) or with paired ones (PP2, PP3, PP6), and unblocked or blocked by spin.
The forms agree only for tensors with the physical symmetries, so random
spin-orbital tensors are built with them (antisymmetric, spin-conserving, real
integrals), and every block the generated code asks for is sliced out of them.

Checks, on CCSD doubles and CCSDT triples:
  - paired and plain permutations give the same residual (unblocked and blocked),
    which tests pq_opt's expansion of PP2/PP3/PP6
  - each spin block (e.g. rt3_aabaab) equals the matching slice of the unblocked
    spin-orbital residual, which tests spin blocking, block keys, and the choice of
    eri blocks

Code is generated in a subprocess per case because spin blocking sets a
process-wide flag in pdaggerq.
"""

import itertools
import json
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

script_path = os.path.dirname(os.path.realpath(__file__))

# spatial orbitals; spin orbitals are ordered occ alpha, occ beta, vir alpha, vir beta
NO, NV = 3, 3  # at least 3 of each per spin, or the all-alpha triples blocks vanish
ORBITALS = {("o", "a"): range(0, NO), ("o", "b"): range(NO, 2 * NO),
            ("v", "a"): range(2 * NO, 2 * NO + NV), ("v", "b"): range(2 * NO + NV, 2 * NO + 2 * NV)}
SPIN = {p: s for (_, s), r in ORBITALS.items() for p in r}
NSO = 2 * (NO + NV)

# name -> (projection, T operators, lhs label order)
CASES = {
    "ccsd_doubles": ([["e2(i,j,b,a)"]], ["t1", "t2"], ["a", "b", "i", "j"]),
    "ccsdt_triples": ([["e3(i,j,k,c,b,a)"]], ["t1", "t2", "t3"], ["a", "b", "c", "i", "j", "k"]),
}

GENERATOR = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
import pdaggerq
from extract_spins import get_spin_labels

left, T, order, paired, spin = json.loads(sys.argv[1])
pq = pdaggerq.pq_helper('fermi')
pq.set_find_paired_permutations(paired)
pq.set_left_operators(left)
pq.add_st_operator(1.0, ['f'], T)
pq.add_st_operator(1.0, ['v'], T)
pq.simplify()

g = pdaggerq.pq_opt({'opt_level': 1})
names = []
for spins, label_to_spin in (get_spin_labels(left + [T]) if spin else {'': {}}).items():
    if spin:
        pq.block_by_spin(label_to_spin)
    name = 'r' if spins == '' else 'r_' + spins
    g.add(pq, name, order)
    names.append(name)
print(json.dumps({'names': names, 'code': g.str('python')}))
"""


def generate(case, paired, spin):
    left, T, order = CASES[case]
    args = json.dumps([left, T, order, paired, spin])
    harness = os.path.join(script_path, "..", "pq_graph", "tests")
    result = subprocess.run([sys.executable, "-c", GENERATOR, args, harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def antisymmetrize(x, groups):
    """antisymmetrize x over each group of axes"""
    for axes in groups:
        total = np.zeros_like(x)
        for perm in itertools.permutations(range(len(axes))):
            sign = np.linalg.det(np.eye(len(axes))[list(perm)])
            order = list(range(x.ndim))
            for a, p in zip(axes, perm):
                order[a] = axes[p]
            total += sign * np.transpose(x, order)
        x = total
    return x


def conserves_spin(*groups):
    """mask: the spins of the first group of axes match those of the second, as multisets.
    for groups of equal size that means the same number of beta spin orbitals"""
    beta = np.array([SPIN[p] == "b" for p in range(NSO)], dtype=int)
    rank = sum(len(g) for g in groups)
    count = [0, 0]
    for axis in range(rank):
        shape = [1] * rank
        shape[axis] = NSO
        count[0 if axis in groups[0] else 1] = count[0 if axis in groups[0] else 1] + beta.reshape(shape)
    return count[0] == count[1]


def spin_orbital_tensors(seed=11):
    """random spin-orbital f, <pq||rs>, t1, t2, t3 with the physical symmetries"""
    rng = np.random.default_rng(seed)
    n = NO + NV
    spatial = lambda p: p % NO if p < 2 * NO else NO + (p - 2 * NO) % NV

    # real, symmetric Fock matrix, diagonal in spin
    h = rng.standard_normal((n, n)); h = h + h.T
    f = np.array([[h[spatial(p), spatial(q)] if SPIN[p] == SPIN[q] else 0.0 for q in range(NSO)]
                  for p in range(NSO)])

    # real chemists' integrals (pq|rs) with 8-fold symmetry, then <pq||rs> = <pq|rs> - <pq|sr>
    c = rng.standard_normal((n, n, n, n))
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        c = c + c.transpose(perm)
    g = np.zeros((NSO,) * 4)
    for p, q, r, s in itertools.product(range(NSO), repeat=4):
        coulomb = c[spatial(p), spatial(r), spatial(q), spatial(s)] if SPIN[p] == SPIN[r] and SPIN[q] == SPIN[s] else 0
        exchange = c[spatial(p), spatial(s), spatial(q), spatial(r)] if SPIN[p] == SPIN[s] and SPIN[q] == SPIN[r] else 0
        g[p, q, r, s] = coulomb - exchange

    # amplitudes t(a..., i...): antisymmetric within each group, spin-conserving
    t1 = 0.1 * rng.standard_normal((NSO, NSO)) * conserves_spin((0,), (1,))
    t2 = antisymmetrize(0.1 * rng.standard_normal((NSO,) * 4), [(0, 1), (2, 3)]) * conserves_spin((0, 1), (2, 3))
    t3 = antisymmetrize(0.1 * rng.standard_normal((NSO,) * 6), [(0, 1, 2), (3, 4, 5)]) * conserves_spin((0, 1, 2), (3, 4, 5))
    return {"f": f, "eri": g, "t1": t1, "t2": t2, "t3": t3}


def orbitals(space, spin):
    """spin orbitals of a space ('o'/'v') and spin ('a', 'b', or '' for both)"""
    return [p for s in (spin or "ab") for p in ORBITALS[(space, s)]]


def block(x, spaces, spins):
    """the block of a full spin-orbital tensor"""
    return x[np.ix_(*[orbitals(o, s) for o, s in zip(spaces, spins or [""] * len(spaces))])]


def residual_block(r, spaces, spins):
    """the block of a residual, whose axes already span only their own space"""
    positions = lambda o, s: [orbitals(o, "").index(p) for p in orbitals(o, s)]
    return r[np.ix_(*[positions(o, s) for o, s in zip(spaces, spins)])]


class Blocks(dict):
    """blocks of a spin-orbital tensor, sliced on first use from their map key"""

    def __init__(self, x, amplitude_spaces=None):
        super().__init__()
        self.x, self.amplitude_spaces = x, amplitude_spaces

    def __missing__(self, key):
        if self.amplitude_spaces:  # amplitudes: the key is the spin string
            self[key] = block(self.x, self.amplitude_spaces, key)
        else:                      # integrals: "<spins>_<spaces>" or "<spaces>"
            spins, _, spaces = key.rpartition("_")
            self[key] = block(self.x, spaces, spins)
        return self[key]


def evaluate(generated, tensors):
    code = generated["code"]
    ns = {"einsum": np.einsum, "np": np}
    ns["f"], ns["eri"] = Blocks(tensors["f"]), Blocks(tensors["eri"])
    ns["Id"] = Blocks(np.eye(NSO))
    for name in ("t1", "t2", "t3"):
        order = int(name[1])
        spaces = "v" * order + "o" * order
        ns[name] = Blocks(tensors[name], spaces) if f'{name}["' in code else block(tensors[name], spaces, "")
    exec(textwrap.dedent(code), ns)
    return {name: ns[name] for name in generated["names"]}


@pytest.fixture(scope="module")
def tensors():
    return spin_orbital_tensors()


@pytest.mark.parametrize("spin", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_paired_permutations(case, spin, tensors):
    plain = evaluate(generate(case, paired=False, spin=spin), tensors)
    paired = evaluate(generate(case, paired=True, spin=spin), tensors)
    assert plain.keys() == paired.keys()
    for name in plain:
        assert np.abs(plain[name]).max() > 1e-3, f"{case} {name}: residual is trivially zero"
        np.testing.assert_allclose(paired[name], plain[name], rtol=1e-10, atol=1e-12,
                                   err_msg=f"{case}: {name} differs with paired permutations")


@pytest.mark.parametrize("case", CASES)
def test_spin_blocks_match_spin_orbital(case, tensors):
    order = CASES[case][2]
    full = evaluate(generate(case, paired=False, spin=False), tensors)["r"]
    blocks = evaluate(generate(case, paired=False, spin=True), tensors)
    for name, value in blocks.items():
        spins = name.split("_")[1]  # spin of each lhs label, in the (sorted) label order
        spaces = "".join("o" if label in "ijklmn" else "v" for label in order)
        expected = residual_block(full, spaces, spins)
        assert np.abs(expected).max() > 1e-3, f"{case} {name}: block is trivially zero"
        np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12,
                                   err_msg=f"{case}: spin block {name} differs from the spin-orbital residual")


if __name__ == "__main__":
    print("Please use pytest to run the tests")
