# -*- coding: utf-8 -*-
"""
Numerical tests for pq_opt, using the pq_graph harnesses.

Each pq_graph/tests/{name}_codegen.py script is run with pdaggerq.pq_graph
replaced by pdaggerq.pq_opt, in a scratch copy of pq_graph/tests, and the
generated {name}_code.py is executed. The harnesses assert on energies; the
EOM harnesses only print their eigenvalues, so those are compared against
reference_outputs/{name}_eigenvalues.txt. Those were made once from pq_graph
output (pq_graph's EOM output is not deterministic and is sometimes invalid, so
it is not rerun here).
"""

import os
import re
import shutil
import subprocess
import sys

import pytest

script_path = os.path.dirname(os.path.realpath(__file__))
harness_dir = os.path.join(script_path, "..", "pq_graph", "tests")

# cisd is left out: cisd_codegen.py sets no_scalars, which pq_opt does not support
# (pq_graph used it to drop the E_HF terms). Pending a decision on how that harness
# should handle the reference energy.
tests = (
    "ccsd",
    "eom_ccsd_quick",
    "cc3",
    "ccsdt",
    "lambda_ccsd",
    "eom_ccsd",
)

try:
    import psi4  # noqa: F401
    tests += ("ccsd_with_spin", "ccsdt_with_spin")
except ImportError:
    pass

eom_tests = ("eom_ccsd_quick", "eom_ccsd")

# runs a codegen script with pq_graph swapped for pq_opt at a given opt_level
runner = """
import runpy, sys
import pdaggerq
level = int(sys.argv[2])
pdaggerq.pq_graph = lambda options: pdaggerq.pq_opt({**options, 'opt_level': level})
sys.argv = sys.argv[1:2]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def run(args, cwd):
    result = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(args)} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout


def generate_and_run(name, level, workdir):
    """run the codegen with pq_opt and then the generated code"""
    run(["runner.py", f"{name}_codegen.py", str(level)], workdir)
    return run([f"{name}_code.py"], workdir)


def eigenvalues(stdout):
    """the excitation energies printed after 'eigenvalues of e(-T) H e(T):'"""
    block = stdout.split("eigenvalues of e(-T) H e(T):", 1)[1]
    values = []
    for line in block.splitlines():
        fields = line.split()
        if len(fields) == 2 and all(re.fullmatch(r"-?\d+\.\d+", f) for f in fields):
            values.append(float(fields[1]))
    return sorted(values)


@pytest.fixture
def workdir(tmp_path):
    for f in os.listdir(harness_dir):
        if f.endswith((".py", ".ref")) and not f.endswith("_code.py"):
            shutil.copy(os.path.join(harness_dir, f), tmp_path)
    (tmp_path / "runner.py").write_text(runner)
    return str(tmp_path)


@pytest.mark.parametrize("level", (0, 1))
@pytest.mark.parametrize("name", tests)
def test_harness(name, level, workdir):
    stdout = generate_and_run(name, level, workdir)

    if name in eom_tests:
        with open(os.path.join(script_path, "reference_outputs", f"{name}_eigenvalues.txt")) as f:
            expected = [float(x) for x in f.read().split()]
        found = eigenvalues(stdout)
        assert len(found) == len(expected) > 0
        assert max(abs(a - b) for a, b in zip(found, expected)) < 1e-6


if __name__ == "__main__":
    print("Please use pytest to run the tests")
