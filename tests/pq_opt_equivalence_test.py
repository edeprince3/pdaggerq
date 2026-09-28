# -*- coding: utf-8 -*-
"""
Random-tensor equivalence tests for pq_opt.

For each case, python code is generated at every optimization level and run on
the same random tensors; every level must give the same results as level 0,
which evaluates pdaggerq's terms exactly as written. Two evaluation orders of the
same expression agree for any values, but shared intermediates (opt_level 2) are
matched using tensor antisymmetry, so the eri blocks and amplitudes are
antisymmetric within each block, exactly as pq_opt assumes; nothing else is
imposed, so this checks the optimizer, not the physics.

Code is generated in a subprocess per case because spin blocking sets a
process-wide flag in pdaggerq.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import zlib
from itertools import permutations

import numpy as np
import pytest

script_path = os.path.dirname(os.path.realpath(__file__))

# name -> pq_opt options; every variant must agree with "0"
VARIANTS = {
    "0": {"opt_level": 0},
    "1": {"opt_level": 1},
    "2": {"opt_level": 2},
    "2, no antisymmetry": {"opt_level": 2, "use_antisymmetry": False},
    "3": {"opt_level": 3},
    "3, one call": {"opt_level": 3, "calls": 1},
    "3, many calls": {"opt_level": 3, "calls": 1e9},
    "4": {"opt_level": 4},
    "4, no antisymmetry": {"opt_level": 4, "use_antisymmetry": False},
}

# the Hamiltonian as (coefficient, operator) pairs, passed to add_st_operator
ELECTRONIC = [(1.0, ['f']), (1.0, ['v'])]
QED = ELECTRONIC + [(1.0, ['w0']), (-1.0, ['d+']), (-1.0, ['d-'])]

# lambda residuals <0|(1 + L) e(-T) [H, e1(a,i)] e(T)|0>: l varies, t does not, and the terms
# without l are constant (opt_level 3 computes them once)
LAMBDA = [(sign * c, ops) for c, h in ELECTRONIC for sign, ops in ((1.0, h + ['e1(a,i)']), (-1.0, ['e1(a,i)'] + h))]

# name -> (equations {lhs: (left ops, right ops, label order)}, T operators, options, spin blocked,
#          Hamiltonian)
CASES = {
    "ccsd": ({"rt1": ([["e1(i,a)"]], [["1"]], ["a", "i"]),
              "rt2": ([["e2(i,j,b,a)"]], [["1"]], ["a", "b", "i", "j"])}, ["t1", "t2"], {}, False, ELECTRONIC),
    "ccsdt": ({"rt3": ([["e3(i,j,k,c,b,a)"]], [["1"]], ["a", "b", "c", "i", "j", "k"])},
              ["t1", "t2", "t3"], {}, False, ELECTRONIC),
    "eom_sigma": ({"sigma1": ([["e1(i,a)"]], [["r1"], ["r2"]], ["a", "i"]),
                   "sigma2": ([["e2(i,j,b,a)"]], [["r1"], ["r2"]], ["a", "b", "i", "j"])},
                  ["t1", "t2"], {"use_trial_index": True}, False, ELECTRONIC),
    "hbar": ({"H00": ([["1"]], [["1"]], []),
              "Hss": ([["e1(i,a)"]], [["e1(e,m)"]], ["a", "i", "e", "m"]),
              "Hsd": ([["e1(i,a)"]], [["e2(e,f,n,m)"]], ["a", "i", "e", "f", "m", "n"])},
             ["t1", "t2"], {}, False, ELECTRONIC),
    "lambda": ({"rl1": ([["1"], ["l1"], ["l2"]], [["1"]], ["a", "i"])}, ["t1", "t2"], {}, False, LAMBDA),
    "ccsd_spin": ({"rt1": ([["e1(i,a)"]], [["1"]], ["a", "i"]),
                   "rt2": ([["e2(i,j,b,a)"]], [["1"]], ["a", "b", "i", "j"])}, ["t1", "t2"], {}, True, ELECTRONIC),
    # QED-CCSD-21 residuals: a single cavity mode (no boson labels), dipole couplings, the scalar w0,
    # photon amplitudes
    "qed_ccsd": ({"r0_1p": ([["B-"]], [["1"]], []),
                  "r1": ([["e1(i,a)"]], [["1"]], ["a", "i"]),
                  "r1_1p": ([["B-", "e1(i,a)"]], [["1"]], ["a", "i"])},
                 ["t1", "t2", "tb1", "teb11", "teb21"], {}, False, QED),
}

GENERATOR = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
import pdaggerq
from extract_spins import get_spin_labels

eqs, T, options, spin, variants, hamiltonian = json.loads(sys.argv[1])
names = []
graphs = {name: pdaggerq.pq_opt({**options, **variant}) for name, variant in variants.items()}
for name, (left, right, order) in eqs.items():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators(left)
    pq.set_right_operators(right)
    for coefficient, operator in hamiltonian:
        pq.add_st_operator(coefficient, operator, T)
    pq.simplify()
    blocks = get_spin_labels(left + right + [T]) if spin else {'': {}}
    for spins, label_to_spin in blocks.items():
        if spin:
            pq.block_by_spin(label_to_spin)
        lhs = name if spins == '' else name + '_' + spins
        for g in graphs.values():
            g.add(pq, lhs, order)
        names.append(lhs)
code = {level: g.str('python') for level, g in graphs.items()}
cpp = {level: g.str('c++') for level, g in graphs.items()}
print(json.dumps({'names': names, 'code': code, 'cpp': cpp}))
"""

NO, NV, NL = 3, 4, 2


def generate(case):
    eqs, T, options, spin, hamiltonian = CASES[case]
    args = json.dumps([eqs, T, options, spin, VARIANTS, hamiltonian])
    harness = os.path.join(script_path, "..", "pq_graph", "tests")
    result = subprocess.run([sys.executable, "-c", GENERATOR, args, harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def random(name, shape):
    """a random tensor that depends only on its name, not on when it is drawn"""
    return np.random.default_rng(zlib.crc32(name.encode())).standard_normal(shape)


def antisymmetrize(x, groups):
    """antisymmetrize x over each group of axes"""
    for axes in groups:
        total = np.zeros_like(x)
        for perm in permutations(range(len(axes))):
            sign = (-1) ** sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))
            order = list(range(x.ndim))
            for axis, p in zip(axes, perm):
                order[axis] = axes[p]
            total += sign * np.transpose(x, order)
        x = total
    return x


def same_kind_parts(groups, spaces, spins):
    """each group of axes split into the parts that share a space and spin (as pq_opt does)"""
    parts = []
    for group in groups:
        by_kind = {}
        for axis in group:
            by_kind.setdefault((spaces[axis], spins[axis]), []).append(axis)
        parts += [p for p in by_kind.values() if len(p) > 1]
    return parts


class Tensors(dict):
    """random tensors created on first use, shaped (and symmetrized) by their map key"""

    def __init__(self, name, shape_of, groups_of=lambda key: []):
        super().__init__()
        self.name, self.shape_of, self.groups_of = name, shape_of, groups_of

    def __missing__(self, key):
        self[key] = antisymmetrize(random(self.name + key, self.shape_of(key)), self.groups_of(key))
        return self[key]


def dims(spaces):
    return tuple({"o": NO, "v": NV, "L": NL}[c] for c in spaces)


# t2, l1, r2, and with photons t1_1p, t0_2p, ...
AMPLITUDE = r"([tlr])([0-4])(?:_([1-4])p)?"


def amplitude_spaces(name, trial):
    # photon amplitudes carry no mode index: the one cavity mode is implied
    kind, order = re.fullmatch(AMPLITUDE, name).groups()[:2]
    order = int(order)
    spaces = "o" * order + "v" * order if kind == "l" else "v" * order + "o" * order
    return ("L" if trial and kind in "rl" else "") + spaces


def amplitude_groups(name, trial, spins):
    """an amplitude's antisymmetric axes: its creation and its annihilation labels"""
    order = int(re.fullmatch(AMPLITUDE, name).group(2))
    offset = 1 if trial and name[0] in "rl" else 0
    spaces = amplitude_spaces(name, trial)
    spins = " " * offset + (spins or " " * (2 * order)) + " " * (len(spaces) - offset - 2 * order)
    groups = [list(range(offset, offset + order)), list(range(offset + order, offset + 2 * order))]
    return same_kind_parts(groups, spaces, spins)


def eri_groups(key):
    """<p,q||r,s> is antisymmetric in p,q and in r,s"""
    spins, _, spaces = key.rpartition("_")
    return same_kind_parts([[0, 1], [2, 3]], spaces, spins or "    ")


def inputs(code, trial, antisymmetric=True):
    """random inputs for the code; antisymmetric=False leaves every tensor without symmetry"""
    # tmps_: intermediates (opt_level >= 2); reused_: what does not vary between calls (opt_level 3)
    ns = {"einsum": np.einsum, "np": np, "tmps_": {}, "reused_": {}}
    symmetrize = antisymmetrize if antisymmetric else (lambda x, groups: x)

    # integrals and identities are keyed by "<blocks>_<spaces>" or "<spaces>"; the cavity
    # frequency w0 is a scalar
    for name in ("f", "eri", "dp"):
        ns[name] = Tensors(name, lambda key: dims(key.split("_")[-1]),
                           eri_groups if name == "eri" and antisymmetric else lambda key: [])
    ns["w0"] = random("w0", ())
    ns["Id"] = {key: np.eye(dims(key.split("_")[-1])[0]) for key in re.findall(r'Id\["([^"]+)"\]', code)}

    # amplitudes are keyed by spin block when blocked, bare arrays otherwise
    for name in {m.group(0) for m in re.finditer(r"\b" + AMPLITUDE + r"\b", code)}:
        spaces = amplitude_spaces(name, trial)
        if f'{name}["' in code:
            ns[name] = Tensors(name, lambda key, spaces=spaces: dims(spaces),
                               lambda key, name=name: amplitude_groups(name, trial, key) if antisymmetric else [])
        else:
            ns[name] = symmetrize(random(name, dims(spaces)), amplitude_groups(name, trial, ""))
    return ns


@pytest.mark.parametrize("case", CASES)
def test_levels_agree(case):
    generated = generate(case)
    names = generated["names"]
    trial = CASES[case][2].get("use_trial_index", False)

    reference = None
    for variant in VARIANTS:
        ns = inputs(generated["code"][variant], trial)
        exec(textwrap.dedent(generated["code"][variant]), ns)
        result = {name: np.asarray(ns[name]) for name in names}
        if reference is None:
            reference = result
            continue
        for name in names:
            np.testing.assert_allclose(result[name], reference[name], rtol=1e-10, atol=1e-10,
                                       err_msg=f"{case}: {name} differs at opt_level {variant}")

    # the c++ printer must at least produce balanced code for every level
    for level, cpp in generated["cpp"].items():
        assert cpp.count("(") == cpp.count(")"), f"{case}: unbalanced parentheses at level {level}"
        assert cpp.count("{") == cpp.count("}"), f"{case}: unbalanced braces at level {level}"


@pytest.mark.parametrize("case", CASES)
def test_no_antisymmetry_needs_no_symmetry(case):
    # with use_antisymmetry off, levels 2 and 4 match only identical products and terms, so they
    # must agree with level 0 even for tensors that have no symmetry at all (e.g. a user-defined
    # tensor)
    generated = generate(case)
    trial = CASES[case][2].get("use_trial_index", False)
    results = {}
    for variant in ("0", "2, no antisymmetry", "4, no antisymmetry"):
        ns = inputs(generated["code"][variant], trial, antisymmetric=False)
        exec(textwrap.dedent(generated["code"][variant]), ns)
        results[variant] = {name: np.asarray(ns[name]) for name in generated["names"]}
    for variant in ("2, no antisymmetry", "4, no antisymmetry"):
        for name in generated["names"]:
            np.testing.assert_allclose(results[variant][name], results["0"][name], rtol=1e-10, atol=1e-10,
                                       err_msg=f"{case}: {name} differs at opt_level {variant}")


if __name__ == "__main__":
    print("Please use pytest to run the tests")
