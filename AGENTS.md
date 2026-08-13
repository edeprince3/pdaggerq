# AGENTS.md - pdaggerq Development Guide

## Architecture

- Python imports a single C++17 pybind11 module: `pdaggerq/__init__.py` re-exports `._pdaggerq`. Core algebra is in `pdaggerq/*.cc`; graph optimization and code generation are in `pq_graph/src/` and `pq_graph/include/`.
- Python-visible API changes usually belong in `pdaggerq/pq_helper.cc` or `pq_graph/src/pq_graph.cc` (`PQGraph::export_pq_graph`) and require a rebuild.
- Pure Python parsing/formatting lives beside the C++ sources in `pdaggerq/`; numerical codegen, methods, and solvers are under `pdaggerq/numerical/{codegen,methods,solvers}`.
- `CMakeLists.txt` is the native build source of truth. It builds one `_pdaggerq` module, fetches pybind11 v2.11.1, and enables OpenMP when available. `setup.py` drives CMake for pip builds.
- `pq_graph/README.md` documents graph options. For linkage/term/addition/fusion representation details and invariants, read `REPRESENTATION_ANALYSIS.md` before changing fusion, pruning, or printers.

## Build

```bash
conda create -n pdev python=3.12 cmake setuptools numpy pytest -c conda-forge -y
conda run -n pdev pip install -e .
conda run -n pdev pip install -r tests/requirements.txt
```

- Use `conda run -n pdev` consistently: `setup.py` passes that interpreter to CMake, avoiding an extension built for a different Python.
- The editable build writes `pdaggerq/_pdaggerq*.so` into the source package. Repo-root imports work only after that artifact exists.
- C++ changes are not auto-rebuilt; rerun `conda run -n pdev pip install -e .`. If CMake state is suspect, remove `build/` first and rebuild.
- `tests/requirements.txt` does not install Psi4. Install Psi4 separately via conda for spin-traced graph tests and for all of `tests/pq_numerical_test.py`.

## Focused Verification

```bash
conda run -n pdev pytest tests/pq_test.py -q
conda run -n pdev pytest tests/pq_test.py -k ccsd_energy -q
conda run -n pdev pytest tests/pq_graph_numerical_test.py -q
conda run -n pdev pytest tests/pq_graph_numerical_test.py -k ccsdt_with_spin -q
conda run -n pdev pytest tests/pq_numerical_test.py -m ccsdt -q
```

- `tests/pq_test.py` currently collects 36 golden-output cases. It runs `examples/{name}.py`, normalizes term order and floats, then uses system `diff` against `tests/reference_outputs/{name}.ref`. Failures are written under `tests/test_outputs/difference/`.
- Regenerate a deliberate golden change with `conda run -n pdev python examples/{name}.py > tests/reference_outputs/{name}.ref`; inspect the normalized diff before accepting it.
- `tests/pq_graph_numerical_test.py` runs `{name}_codegen.py` and then generated `{name}_code.py`. It collects 7 PySCF cases without Psi4 and adds `ccsd_with_spin`/`ccsdt_with_spin` when Psi4 imports successfully.
- `pq_graph/tests/*_code.ref` files are executable harness templates containing `# INSERTED CODE`, not snapshots. Generated `*_code.py` files are gitignored, and importing the graph numerical test deletes existing generated files.
- `tests/pq_numerical_test.py` is a separate Psi4-dependent generated-method suite; use its pytest markers from `tests/pytest.ini` for focused runs.
- Test logs (`pq_test.log`, `pq_graph_numerical_test.log`, `pq_numerical_test.log`) are written in pytest's invocation directory.
- `pdaggerq/algebra_test.py` and `parser_test.py` currently fail collection because they reference removed `T1amps`/`T2amps` symbols; do not treat them as a valid regression suite without repairing them first.
- No repository lint, formatter, or typecheck task is configured. The README's pylint mention is prose, not an executable check.

## pq_graph Gotchas

- After fusion/pruning/printer changes, always verify the CCSDT spin case with `batched=True`, `expand_permutations=True`, `opt_level=6`, and `max_temps=-1`; finite limits can hide later invalid fusion groups:

```bash
conda run -n pdev python pq_graph/tests/ccsdt_with_spin_codegen.py
conda run -n pdev python pq_graph/tests/ccsdt_with_spin_code.py
```

- `LinkMerger::merge()` rewrites target terms and nulls merge-term LHSs in place; accepted fusion groups must own pairwise-disjoint `Term*` sets.
- Addition linkages are opaque to default `is_expandable()`. When an addition is an operand of multiplication, printers must group it (`is_expandable(false, true)`); keep Einsum, TAMM, and TiledArray printer behavior aligned.
