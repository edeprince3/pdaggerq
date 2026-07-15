# AGENTS.md — pdaggerq Development Guide

## Architecture

`pdaggerq` is a Python-wrapped C++17 algebra library that normal-order fermionic creation/annihilation operator strings. The shared lib `_pdaggerq.so` (C++ core → pybind11) is compiled via CMake during pip install. Graph-based contraction optimization lives in `pq_graph/`.

**Source layout:**
- `pdaggerq/*` — C++ algebra engine sources and headers
- `pq_graph/src/*` — C++ sources for the graph-based contraction optimization
- `pq_graph/include/*` — C++ headers for the graph-based contraction optimization
- `CMakeLists.txt` — entrypoint for the entire build; fetches pybind11 v2.11.1 via FetchContent
- `setup.py` — custom `CMakeBuild` extension class; drives cmake from pip

## Install / Build

```bash
conda create -n pdev python=3.12 cmake pybind11 setuptools numpy pytest -c conda-forge -y
conda activate pdev
conda install psi4 -c conda-forge/label/psi4_dev -c conda-forge -y   # optional, needed by numerical tests
conda run -n pdev pip install -e .                                     # builds + installs in editable mode
conda run -n pdev pip install openfermionpyscf                         # test dependencies
```

**Mandatory:** always use `conda run -n pdev` for pip/Python commands when multiple Pythons exist — CMake's FetchContent may pick the wrong interpreter otherwise.

## Test

```bash
conda run -n pdev pytest tests/pq_test.py -v           # 37 algebraic output tests (fast)
conda run -n pdev pytest tests/numerical_test.py -v    # 9 psi4 energy verification tests (~10 min total)
```

- `tests/reference_outputs/` — golden files for pq_test. If you change algebra output, update these to match. Update in one place so all references stay consistent. Check here when debugging files in `pdaggerq/`
- `tests/numerical_test.log` — written by numerical tests. Check here when debugging files in `pq_graph/`
