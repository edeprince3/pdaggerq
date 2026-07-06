# AGENTS.md — pdaggerq Developer Setup Guide

This document describes how to set up a development environment for the `pdaggerq` repository and run its test suite. It is intended for AI agents and developers onboarding to this project.

---

## Repository Overview

`pdaggerq` is a fermionic computer algebra package (C++/Python) for bringing strings of creation/annihilation operators to normal order. It uses:

- **C++17** for the core library
- **pybind11** for Python bindings (fetched automatically via CMake `FetchContent`)
- **CMake** (≥ 3.12) as the build system
- **setuptools** + a custom `CMakeBuild` class in `setup.py` to drive the CMake build from pip

---

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- A C++17-compatible compiler (GCC 13+ recommended)
- Git

---

## 1. Create and Activate a Conda Environment

Create a dedicated conda environment with all required build and test dependencies:

```bash
conda create -n pdev python=3.12 cmake pybind11 setuptools numpy pytest -c conda-forge -y
conda activate pdev
```

### Optional: psi4 (required for spin-trace numerical tests)

psi4 is available on conda-forge and must be installed via conda (not pip):

```bash
conda install psi4 -c conda-forge/label/psi4_dev -c conda-forge -y
```

> **Note:** Ensure `conda activate pdev` has been run before installing psi4 so it lands in the correct environment.

---

## 2. Install the Repository

From the repository root, install in editable mode. The `conda run` approach is recommended to ensure CMake resolves the correct Python interpreter from the active environment:

```bash
conda run -n pdev pip install -e .
```

> **Important:** Do **not** run `pip install -e .` with a bare shell `pip` if multiple Python versions exist on the system. CMake's `FetchContent`-based pybind11 build will pick up whichever Python appears first on `PATH`, which may not match the pip invoking the build. Using `conda run -n pdev` ensures PATH is set correctly before the build starts.

The build:
1. Fetches pybind11 v2.11.1 via CMake `FetchContent`
2. Compiles the C++17 shared library (`_pdaggerq.cpython-*.so`)
3. Installs the `pdaggerq` Python package in editable mode

---

## 3. Install Test Dependencies

```bash
conda run -n pdev pip install openfermionpyscf
```

This installs:
- `openfermion`
- `pyscf`
- `openfermionpyscf`
- `cirq-core`, `sympy`, and other transitive dependencies

---

## 4. Running the Test Suite

Tests live in `test/` (not `tests/`). Run with:

```bash
conda run -n pdev pytest test/ -v
```

The suite collects **46 tests** across two files:

| File | Description |
|------|-------------|
| `test/pq_test.py` | Algebraic output tests — compare pdaggerq equation output against reference files |
| `test/numerical_test.py` | Numerical tests — run generated pq_graph code against psi4 to verify energies |

A successful run looks like:
```
=================== 46 passed in ~628s (0:10:27) ===================
```

### Known Test Status

- **`test/pq_test.py`** — all 37 tests pass ✅
- **`test/numerical_test.py`** — all 9 tests pass ✅

### Test Logs

Numerical tests write detailed output (generated equations, intermediate code, tracebacks) to `numerical_test.log`. Check this file to monitor progress of long-running numerical tests.

> The numerical tests invoke psi4 quantum chemistry calculations and take **~10 minutes** to complete in total.

---

## 5. Environment Summary

| Package | Version (tested) | Source |
|---------|-----------------|--------|
| Python | 3.12 | conda-forge |
| cmake | 4.3.4 | conda-forge |
| pybind11 | 3.0.3 (conda) + 2.11.1 (CMake FetchContent) | conda-forge / GitHub |
| setuptools | 82.0.1 | conda-forge |
| numpy | 2.5.0 | conda-forge |
| pytest | 9.1.1 | conda-forge |
| psi4 | 1.11 | conda-forge/label/psi4_dev |
| openfermionpyscf | 0.5 | PyPI |
| pyscf | 2.13.1 | PyPI |
| openfermion | 1.7.1 | PyPI |

---

## 6. Project Structure

```
pdaggerq/
├── CMakeLists.txt          # CMake build definition (C++17, pybind11 via FetchContent)
├── setup.py                # Custom CMakeBuild setuptools extension
├── pyproject.toml          # Build system declaration
├── pdaggerq/               # C++ source + Python package
├── pq_graph/               # Graph-based contraction optimizer + generated test scripts
│   └── tests/              # Generated *_codegen.py and *_code.py scripts
├── test/                   # pytest test suite
│   ├── pq_test.py
│   ├── numerical_test.py
│   ├── reference_outputs/  # Reference text outputs for pq_test
│   └── requirements.txt
└── examples/               # Usage examples
```
