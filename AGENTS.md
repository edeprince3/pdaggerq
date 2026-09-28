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
- `tests/requirements.txt` does not install Psi4. Only the spin-traced graph tests need it; `tests/pq_numerical_test.py` uses PySCF when it is installed and falls back to Psi4.

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
- `tests/pq_numerical_test.py` is a separate generated-method suite (PySCF, or Psi4 if PySCF is missing); use its pytest markers from `tests/pytest.ini` for focused runs.
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

## pq_opt (lightweight pq_graph replacement)

- `pq_opt/` is exposed as `pdaggerq.pq_opt`. It takes pq_graph's options dict and `add/optimize/print/str/to_strings/analysis`, so a codegen script switches by changing the class name. Unlike pq_graph, `add()` requires the output index order (`label_order`, or a name like `rt2(a,b,i,j)`) listing every external index exactly once (an empty list for a scalar); a trial index is always the first axis and is not listed. Pipeline: `ingest.cc` (pq_string -> `Term`) -> `order.cc` (subset-DP contraction order) -> `build_program.cc` (-> `Program`) -> `printer.h` backends (`numpy_printer.cc`, `tiledarray_printer.cc`). Types are in `ir.h`.
- `pq_opt/pdaggerq_access.h` holds every read of pdaggerq's data structures that is specific to how this release represents bosons; `ingest.cc` goes through it. The single cavity mode carries no labels: `w0` is a flag on the string (printed as a scalar factor), `dp` has no mode axis, and `t0_1p` is a scalar.
- IR invariant: every label in a `Term` appears once (external, also on the lhs) or twice in two different tensors. Ingest turns traces into `Id` contractions, so passes and printers never see a repeated label.
- `opt_level` 0 prints terms as given; >= 1 uses the optimal contraction order; >= 2 also extracts shared intermediates (`intermediates.cc`: two-tensor nodes of the terms' optimal trees that recur across terms, greedily by flops saved, into `tmps_`, freed after last use; `max_temps` caps them). Products are matched up to the antisymmetry pdaggerq itself assumes (eri, and amplitudes with `has_permutational_symmetry`; never user-defined `o<name>` operators, other integrals, or bosons); `use_antisymmetry: False` restricts matching to identical products. >= 3 also hoists what does not change between calls (`hoist_invariants` in `intermediates.cc`): when some tensor varies between calls (`TensorRef::varies`, from the `varying` option: names or name prefixes followed by a digit, default `["r", "l"]`; everything else is assumed fixed), each term's tree minimizes flops per call, with fixed subtrees (t amplitudes, integrals) computed once into `reused_` if no intermediate in them is larger than the term's largest fixed tensor and if that pays off over `calls` calls (default 10); terms with no varying tensor are summed once per equation. >= 4 also merges (`merge_terms`): per-call terms of an equation that contract the same varying tensors the same way (up to antisymmetry, same permutation operator) with one fixed tensor each become one term reading the sum of those tensors, a `reused_` built once (Hbar elements, in EOM-CC); pieces only needed for such sums are freed after use. Then `merge_by_anchor` merges terms that contract the same anchor tensor the same way (same permutation operator) with different other parts into one term reading the sum of those parts, a `tmps_["s0001_..."]` built on every call, when building each part first costs less than the term's own best order and the flops saved exceed the one remaining contraction. The anchor is a term's largest fixed tensor when the graph has varying tensors (only terms with varying tensors take part), and its largest tensor otherwise (e.g. CC residuals: 1.48x fewer flops for the CCSDT t3 residual). `to_strings(type, part)` prints `"all"`, `"reused"` (run once), or `"per_call"`; autogen's `graph_code` runs the reused part on a generated function's first call and caches it in `self.reused_` (keyed by a hash of that code, since generated function names repeat), so the t amplitudes must not change while that object is in use; `configure_graph(options, varying)` widens the list where more changes between calls (`cc_response_terms`: `x` amplitudes and the perturbation `h`). Graphs with no r/l tensors (e.g. CC residuals) are unaffected. Cost uses numeric `sizes` (default `{o: 20, v: 100, O: 1, V: 20, L: 10, b: 1}`; `O`/`V` are the nuclear (NEO) spaces; `b` is a labeled cavity-mode space, unused while the cavity mode carries no labels).
- `pdaggerq/numerical/codegen/autogen.py` (`configure_graph`) generates code with pq_opt at `opt_level` 1. `PDAGGERQ_CODEGEN_BACKEND=pq_graph` switches every equation to pq_graph, and `PDAGGERQ_CODEGEN_OPT_LEVEL` overrides `opt_level`; use them to compare the two on `tests/pq_numerical_test.py`.
- Verify with `tests/pq_opt_equivalence_test.py` (seconds: every level must match level 0 on random tensors), `tests/pq_opt_blocks_test.py` (~15 s: paired permutations and spin blocks against spin-orbital equations with symmetric random tensors), `tests/pq_opt_coefficient_test.py` (coefficient snapping), `tests/pq_opt_neo_test.py` (NEO residuals against pq_graph at opt_level 0), and `tests/pq_opt_numerical_test.py` (~12 min: the pq_graph harnesses run through pq_opt; EOM eigenvalues are checked against `tests/reference_outputs/*_eigenvalues.txt` because pq_graph's EOM output is nondeterministic and sometimes uses a temporary before defining it).
