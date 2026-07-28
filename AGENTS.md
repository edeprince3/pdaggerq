# AGENTS.md — pdaggerq Development Guide

## Architecture

`pdaggerq` is a Python-wrapped C++17 algebra library that normal-orders fermionic operator strings. Everything Python-visible comes from a single pybind11 module: `pdaggerq/__init__.py` is just `from ._pdaggerq import *` (exposes `pq_helper`, `pq_graph`, etc.). The module is compiled from:

- `pdaggerq/*.cc` — core algebra engine
- `pq_graph/src/*`, `pq_graph/include/*` — graph-based contraction optimizer / code generator (uses OpenMP)
- pybind exports live in `pdaggerq/pq_helper.cc` and `pq_graph/src/pq_graph.cc` (`export_pq_graph`) — any Python API change means editing C++ and rebuilding

`pdaggerq/` also holds a pure-Python layer (`algebra.py`, `parser.py`, `latex.py`, `chronus.py`, `term_grouper.py`) with co-located pytest files (`*_test.py`). `CMakeLists.txt` is the single build entrypoint (fetches pybind11 v2.11.1 via FetchContent); `setup.py`'s `CMakeBuild` class drives cmake from pip. See `pq_graph/README.md` for the `pq_graph({...})` options dict (`opt_level` 0–6, `print_level`, `nthreads`, ...).

## Build

```bash
conda create -n pdev python=3.12 cmake pybind11 setuptools numpy pytest -c conda-forge -y
conda run -n pdev pip install -e .                               # compiles _pdaggerq.so via CMake
conda run -n pdev pip install -r tests/requirements.txt          # pytest, numpy, pyscf, openfermion(pyscf)
conda install psi4 -c conda-forge/label/psi4_dev -c conda-forge -y   # optional; only 2 numerical tests need it
```

- Use `conda run -n pdev` for pip/Python when multiple Pythons exist — CMake's FindPython may otherwise bind the wrong interpreter.
- **No incremental/auto rebuild:** after editing C++, re-run `pip install -e .`. If the package was installed non-editable (a plain copy in site-packages), edits to the pure-Python files also require a reinstall to take effect.
- **Import shadowing:** the source tree contains no compiled `.so`, so whenever the repo root lands on `sys.path` (`python -m pytest`, `python -c`, `pytest pdaggerq/`), the local `pdaggerq/` shadows the installed package and fails with `No module named 'pdaggerq._pdaggerq'`. Use the `pytest` console script from the repo root, or run Python from another directory.

## Test

```bash
conda run -n pdev pytest tests/pq_test.py -v              # 37 algebra tests (fast)
conda run -n pdev pytest tests/pq_test.py -k ccsd_energy  # single test
conda run -n pdev pytest tests/numerical_test.py -v       # 9 numerical tests (~10 min)
```

**`pq_test.py` (golden-file algebra tests):** each test runs `examples/{name}.py` as a subprocess and diffs normalized stdout (terms sorted, floats to 6 dp) against `tests/reference_outputs/{name}.ref`. To add/update a test: create the example, add its name to the tuples in `pq_test.py`, and regenerate the golden file with `python examples/{name}.py > tests/reference_outputs/{name}.ref`. On failure, check `tests/test_outputs/difference/{name}_diff.out`.

**`numerical_test.py` (generated-code energy tests):** two stages per test — it runs `pq_graph/tests/{name}_codegen.py`, which injects pq_graph-optimized code into the template `{name}_code.ref` at the `# INSERTED CODE` marker to produce `{name}_code.py` (gitignored), then runs that file, which asserts CC/EOM energies against pyscf (the two `*_with_spin` tests also need psi4). So `.ref` files in `pq_graph/tests/` are harness templates, not pure snapshots. Debug via `numerical_test.log`.

- `pq_test.log` / `numerical_test.log` are written to the directory pytest was invoked from (repo root if run as above), not to `tests/`.
- The co-located Python tests (`pdaggerq/algebra_test.py`, `parser_test.py`, `blocking_leak_test.py`) import the compiled module, so under a non-editable install they error out from the repo root (same shadowing issue as above).
