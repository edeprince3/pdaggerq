# -*- coding: utf-8 -*-
"""
pq_graph's binarize option (split every contraction into two-operand steps).

Each case is generated with binarize off and on, and both versions of the python
code are run on the same random tensors; they must agree. The binarized code is
also checked statically: a binarization temporary (tmps_["bin..."]) must not be
reassigned while it is still in use, or read after it is deleted. Reusing one
name for two intermediates of the same shape within a term used to break both.

Code is generated in a subprocess per case because spin blocking sets a
process-wide flag in pdaggerq.
"""

import json
import os
import re
import subprocess
import sys
import zlib

import numpy as np
import pytest

script_path = os.path.dirname(os.path.realpath(__file__))

NO, NV = 3, 4

# name -> (projection, T operators, lhs label order, spin blocked)
CASES = {
    "ccsd_doubles_spin": ([["e2(i,j,b,a)"]], ["t1", "t2"], ["a", "b", "i", "j"], True),
    "ccsdt_triples": ([["e3(i,j,k,c,b,a)"]], ["t1", "t2", "t3"], ["a", "b", "c", "i", "j", "k"], False),
}

GENERATOR = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
import pdaggerq
from extract_spins import get_spin_labels

left, T, order, spin = json.loads(sys.argv[1])
pq = pdaggerq.pq_helper('fermi')
pq.set_left_operators(left)
pq.add_st_operator(1.0, ['f'], T)
pq.add_st_operator(1.0, ['v'], T)
pq.simplify()

graphs = {b: pdaggerq.pq_graph({'opt_level': 1, 'print_level': 0, 'binarize': b, 'nthreads': 1})
          for b in (False, True)}
names = []
for spins, label_to_spin in (get_spin_labels(left + [T]) if spin else {'': {}}).items():
    if spin:
        pq.block_by_spin(label_to_spin)
    name = 'r' if spins == '' else 'r_' + spins
    for g in graphs.values():
        g.add(pq, name, order)
    names.append(name)
code = {}
for b, g in graphs.items():
    g.optimize()
    code[str(b)] = g.str('python')
print(json.dumps({'names': names, 'code': code}))
"""


def generate(case):
    left, T, order, spin = CASES[case]
    harness = os.path.join(script_path, "..", "pq_graph", "tests")
    result = subprocess.run([sys.executable, "-c", GENERATOR, json.dumps([left, T, order, spin]), harness],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def random(name, shape):
    """a random tensor that depends only on its name, not on when it is drawn"""
    return np.random.default_rng(zlib.crc32(name.encode())).standard_normal(shape)


class Tensors(dict):
    """random tensors created on first use, shaped by their map key"""

    def __init__(self, name, shape_of):
        super().__init__()
        self.name, self.shape_of = name, shape_of

    def __missing__(self, key):
        self[key] = random(self.name + key, self.shape_of(key))
        return self[key]


def dims(spaces):
    return tuple({"o": NO, "v": NV}[c] for c in spaces)


def run(code):
    """execute generated code on random tensors; returns its namespace"""
    ns = {"np": np, "einsum": np.einsum, "tmps_": {}, "scalars_": {}}
    ns["f"] = Tensors("f", lambda key: dims(key.split("_")[-1]))
    ns["eri"] = Tensors("eri", lambda key: dims(key.split("_")[-1]))
    ns["Id"] = {key: np.eye(dims(key.split("_")[-1])[0]) for key in re.findall(r'Id\["([^"]+)"\]', code)}
    for name in {m for m in re.findall(r"\b(t[1-3])\b", code)}:
        order = int(name[1])
        spaces = "v" * order + "o" * order
        ns[name] = Tensors(name, lambda key, s=spaces: dims(s)) if f'{name}["' in code else random(name, dims(spaces))
    # the code is indented for a function body; banner comments sit at column 0
    exec("\n".join(l[4:] if l.startswith("    ") else l for l in code.split("\n")), ns)
    return ns


def misused_temporaries(code):
    """uses of binarization temporaries that read a deleted or unassigned one, or overwrite a live one"""
    live, problems = set(), []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        deleted = re.match(r'del tmps_\["(bin[^"]*)"\]', s)
        if deleted:
            live.discard(deleted.group(1))
            continue
        target = re.match(r'tmps_\["(bin[^"]*)"\]\s*=', s)
        reads = re.findall(r'tmps_\["(bin[^"]*)"\]', s.split("=", 1)[1] if target else s)
        problems += [f"reads {r} while it is not assigned: {s[:100]}" for r in reads if r not in live]
        if target:
            if target.group(1) in live:
                problems.append(f"overwrites {target.group(1)} while it is in use: {s[:100]}")
            live.add(target.group(1))
    return problems


@pytest.fixture(scope="module", params=CASES)
def generated(request):
    return request.param, generate(request.param)


def test_binarized_temporaries_are_not_reused_while_live(generated):
    case, g = generated
    code = g["code"]["True"]
    assert 'tmps_["bin' in code, f"{case}: nothing was binarized"
    problems = misused_temporaries(code)
    assert not problems, f"{case}: {len(problems)} misused binarization temporaries, e.g. {problems[0]}"


def test_binarize_gives_the_same_results(generated):
    case, g = generated
    plain, binarized = run(g["code"]["False"]), run(g["code"]["True"])
    for name in g["names"]:
        assert np.abs(plain[name]).max() > 1e-3, f"{case}: {name} is trivially zero"
        np.testing.assert_allclose(binarized[name], plain[name], rtol=1e-10, atol=1e-10,
                                   err_msg=f"{case}: {name} differs with binarize on")


if __name__ == "__main__":
    print("Please use pytest to run the tests")
