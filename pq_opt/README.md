# pq_opt

pq_opt digests pdaggerq-derived equations and translates them into usable
Python or C++ code. Tensor contractions are expressed via
[NumPy](https://numpy.org/) `einsum` calls (for Python-based code
execution) or
[TiledArray](https://valeevgroup.github.io/tiledarray/dox-master/index.html)
expressions (for C++-based code execution). Before printing code, pq_opt
finds the optimal order of each contraction within a term, identifies
common intermediates between terms, and merges terms that differ in one
part. For two-step procedures like EOMCC, pq_opt will also identify
reusable intermediate expressions that do not change between calls.

pq_opt is a lightweight replacement for [pq_graph](../pq_graph/README.md).
Both modules use the same Python API,  so a code-generation script can
change between them simply by referring to the appropriate class name.
pq_opt is built for efficient code optimization, so we recommend using it
at the maximum optimization level ("opt_level": 4, see below).

## Quick start

```python
import pdaggerq

# derive the CCSD singles residual
pq = pdaggerq.pq_helper("fermi")
pq.set_left_operators([["e1(i,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()

# optimize it and print code
opt = pdaggerq.pq_opt({"opt_level": 4}) # pass pq_opt options as a dictionary
opt.add(pq, "r1", ["a", "i"])           # the lhs is r1(a,i): every external index, in this order
opt.optimize()
print(opt.str("python"))                # or "c++" for TiledArray
opt.analysis()                          # flops and scaling
```

which prints statements like

```python
    # tmps_[0001_ov](k,c)  = eri[oovv](k,j,c,b) t1(b,j)
    # flops: o2v2
    tmps_["0001_ov"] = 1.0 * einsum('kjcb,bj->kc', eri["oovv"], t1)

    # r1  = +1.000 f(a,i)
    r1 = 1.0 * f["vo"]

    # r1 += -1.000 f(j,i) t1(a,j)
    # flops: o2v1
    r1 -= einsum('ji,aj->ai', f["oo"], t1)
```

Each statement comes with the pdaggerq term it evaluates and the scaling of its costliest
contraction.

## Adding equations

```python
opt.add(pq, equation_name, label_order)
```

- `pq` is a `pq_helper` object with which the user has already called
  `simplify()`, and optionally after `block_by_spin(...)`.  Call `add`
  once per spin block, with a distinct name such as `"r2_abab"`.
- `label_order` is a required argument that lists every external
  (non-summed) index of the equation exactly once, in the order of the
  left-hand side's axes. Alternatively, the order can be specified in the
  name, e.g.  `opt.add(pq, "r2(a,b,i,j)")`. An equation without external
  indices (e.g., a scalar such as the energy) takes an empty list.
- With `use_trial_index`, r and l amplitudes get a leading trial-vector
  axis, and so does the left-hand side. The trial index is always the
  first axis and is never listed in `label_order`.
- Boson (cavity-mode) indices may be listed, all or none. If they are not
  listed, they follow the fermion indices.
- Adding the same name again appends terms to that equation. Its index
  order must match.

## Optimization levels

| `opt_level` | What it adds |
|---|---|
| 0 | Terms exactly as pdaggerq wrote them. |
| 1 | The cheapest contraction order for each term (the default). |
| 2 | Shared intermediates: products that occur in several terms are computed once, into `tmps_`, and freed after their last use. Products are matched up to renaming of contraction labels and up to the antisymmetry pdaggerq assumes. |
| 3 | Hoisting: parts that do not change between calls of the generated code are computed once, into `reused_` (see [Hoisting](#hoisting-opt_level-3)). |
| 4 | Merging: terms that differ in one part are summed before the expensive contraction (see [Merging](#merging-opt_level-4)). |

Each level includes the levels below it. For the same equations,
optimization levels 2–4 cover what pq_graph does at its levels 2–6.

### Hoisting (opt_level 3)

EOMCC sigma builds, CC lambda residuals, and similar code are called many
times by an iterative solver. The r or l amplitudes change between calls;
the t amplitudes and integrals do not.  Level 3 marks the tensors that
change as *varying* and orders each term to minimize the flops per call. A
product of fixed tensors is computed once and stored in `reused_` if two
conditions hold:

- no intermediate in it is larger than the term's largest fixed tensor,
  which keeps storage bounded;
- computing it once pays off over the expected number of calls (option
  `calls`, default 10).

Terms containing with no varying tensors are summed once per equation.
Code with no varying tensors, such as CC residuals where the t amplitudes
change every iteration, is not hoisted.

Which tensors vary is set by user via the `varying` option. Each entry is
a tensor name, or a prefix followed by a digit (`"r"` matches `r1`, `r2`,
`r0_1p`). The default is `["r", "l"]`.  **All other tensors are assumed
not to change between calls.** If a tensor is expected to vary, specify
it.  For example, response theory also varies the perturbation and the `x`
amplitudes: `{"varying": ["r", "l", "x", "h"]}`.

The generated code reads the reuasable intermediates from `reused_`. Print
it separately with `part="reused"` and run it once, or let the method
generator do that for you (see [Use in pdaggerq's numerical
methods](#use-in-pdaggerqs-numerical-methods)).

### Merging (opt_level 4)

Two passes sum terms that differ in one part, then contract once:

- **Fixed side.** Terms that multiply the same varying tensors, in the same way, by one fixed
  tensor each are combined into a single term. It reads the sum of those fixed tensors, built once as a
  `reused_`. In EOMCC these sums are elements of the similarity-transformed Hamiltonian.
- **Anchor.** Terms that multiply the same tensor (their largest fixed tensor, or their largest
  tensor when nothing varies) by different parts become one term. It reads the sum of those
  parts, `tmps_["s0001_..."]`, built on every call. A term takes part only if building its part
  first costs less than its own best order, and a group only if the flops saved exceed the one
  remaining contraction. This pass applies to CC residuals too.

Terms are merged only when they carry the same permutation operator.

## Printing code

```python
opt.str(print_type="python", part="all")         # code is a multi-line string (returns str). no printing
opt.to_strings(print_type="python", part="all")  # code is a list of strings (returns list[str]). no printing
opt.print(print_type="python", part="all")       # writes the code to stdout
```

- `print_type` is `"python"` (also `"einsum"`, `"numpy"`) or `"c++"` (also `"cpp"`,
  `"tiledarray"`, `"ta"`).
- `part` is `"all"`, `"reused"` (what runs once, at level 3 and above) or `"per_call"`.

The generated code expects its inputs under these names:

| Name | Holds |
|---|---|
| `f["oo"]`, `f["ov"]`, ... | Fock matrix blocks, keyed by the spaces of their indices |
| `eri["oovv"]`, ... | antisymmetrized two-electron integrals `<pq\|\|rs>`, in the blocks pq_opt permutes them to (see `permute_eri`) |
| `t1`, `t2`, `r1`, `l2`, ... | amplitudes, bare arrays; with spin blocking, dicts keyed by block (`t2["abab"]`) |
| `Id["oo"]`, `Id["vv"]` | identity matrices (pq_opt writes a trace as a contraction with one) |

Intermediates live in dicts that the calling code creates: `tmps_` (per call; entries are
deleted after their last use), and `reused_` (kept between calls). In C++, scalar intermediates
live in `scalars_` and `reused_scalars_`. A term with a permutation operator is computed once
into `perm_tmp`, and its permuted copies are added to the target.

With spin blocking, keys carry the spin block and the spaces:
`f["aa_ov"]`, `eri["abab_vvoo"]`, `t2["abab"]`.

## Options

Options are passed as a dict to the constructor or to `set_options`.

| Option | Default | Meaning |
|---|---|---|
| `opt_level` | 1 | See [Optimization levels](#optimization-levels). |
| `sizes` | `{"o": 20, "v": 100, "O": 1, "V": 20, "L": 10, "b": 1}` | The size of each index space, for the cost model (see below). |
| `nocc`, `nvirt` | – | Set `sizes["o"]` and `sizes["v"]` when positive (pq_graph's names). |
| `max_temps` | -1 | The most shared intermediates to create at level 2 and above; -1 means no limit. |
| `use_antisymmetry` | `True` | Match products up to the antisymmetry pdaggerq assumes. This covers `eri` and amplitudes, never user-defined operators, other integrals, or bosons. With `False`, only identical products are matched. |
| `varying` | `["r", "l"]` | The tensors that change between calls (level 3 and above). |
| `calls` | 10 | How many calls the generated code is expected to serve (level 3 and above). |
| `use_trial_index` | `False` | Give r and l amplitudes a leading axis over trial vectors. |
| `permute_eri` | `True` | Bring each `eri` block to a standard set (`oooo`, `vvvv`, `oovv`, `vvoo`, `vovo`, `vooo`, `oovo`, `vovv`, `vvvo`). |
| `has_symmetric_eri` | `False` | `eri` also has bra-ket symmetry, `<pq\|rs> = <rs\|pq>`. |
| `print_comments` | `True` | Write each term's pdaggerq string and scaling above its code. |
| `deallocate` | `True` | Free each intermediate after its last use. |

**Sizes.** The keys are single characters, one per index space:

| Key | Space |
|---|---|
| `o`, `v` | electron occupied, virtual |
| `O`, `V` | nuclear occupied, virtual (NEO) |
| `L` | trial vectors per call |
| `b` | boson (cavity) modes |

Only the ones given change. The sizes affect only the choices pq_opt makes
(contraction order, intermediates, hoisting, merging); the generated code
works for any actual size. It is worth setting them near your target
system parameters, especially when the occupied and virtual counts are
close.

```python
opt = pdaggerq.pq_opt({"opt_level": 4, "sizes": {"o": 50, "v": 60}})
```

**pq_graph compatibility.** These pq_graph options are accepted and ignored: `batched`,
`batch_size`, `max_depth`, `low_memory`, `nthreads`, `expand_permutations`, `separate_sigma`,
`reindex_temps`, `binarize`, `cache_elements`, `cache_depth`, `max_shape`, `dims`, and
`print_level`.

These options would change the equations and are not yet supported, so
pq_opt raises an error when they are turned on: `decompose_eri`,
`density_fitting`, `no_scalars`, `no_trace`.

`assemble()` does nothing (it is part of `optimize()`), and `write_dot()` prints a warning.

## Checking the cost

```python
opt.analysis()   # a table of flops and worst scaling per target
opt.costs()      # {"once": ..., "per_call": ..., "stored": ...}
```

`analysis()` prints one row per target: each equation, all `tmps_`, and all `reused_`. At level
3 and above, the table is split into what runs once and what runs on each call, and it reports
how many elements `reused_` stores. Flops are counted at `sizes`: a contraction costs the
product of the sizes of every index it touches. Adding a term's permuted copies, and plain
transposes, cost nothing in this model.

To compare levels:

```python
for level in range(5):
    opt = pdaggerq.pq_opt({"opt_level": level})
    opt.add(pq, "r1", ["a", "i"])
    opt.optimize()
    print(level, opt.costs()["per_call"])
```

## Use in pdaggerq's numerical methods

`pdaggerq/numerical/codegen/autogen.py` generates the code for the methods in
`pdaggerq/numerical/methods` with pq_opt.

- **Level.** Without options it runs at `opt_level` 4. A method's `pq_graph_options` replace
  those defaults, e.g. `CCSD(wfn, mol, pq_graph_options={"opt_level": 4, "sizes": {...}})`. When
  passing options, include `opt_level`.
- **Once-only code.** Each generated function computes its `reused_` part on its first call and
  keeps it in `self.reused_`, keyed by a hash of that code. The t amplitudes must therefore not
  change while the same object is in use.
- **Varying tensors.** `cc_response_terms` declares `x` and `h` as varying in addition to r and l.
- **Environment variables.** `PDAGGERQ_CODEGEN_BACKEND=pq_graph` switches to pq_graph, which
  then defaults to its `opt_level` 1. `PDAGGERQ_CODEGEN_OPT_LEVEL` overrides `opt_level` for
  either backend. Together they are handy for comparisons on `tests/pq_numerical_test.py`.

## How it works

```
pq_helper strings --ingest--> Equation --> passes --> build_program --> Program --> printer --> code
```

| File | Role |
|---|---|
| `ir.h` | The data structures: `Index`, `TensorRef`, `Term`, `Equation`, `Expr` (a contraction tree), `Stmt`, `Program`. |
| `ingest.cc` | Reads pdaggerq's strings into terms: tensor names and blocks, `eri` permutation, antisymmetry, traces, permutation operators, coefficients snapped to exact fractions. |
| `pdaggerq_access.h` | Every read of pdaggerq's data structures that differs between releases (e.g. labeled vs. unlabeled bosons). |
| `order.cc` | The cheapest contraction tree of a term, by dynamic programming over subsets of its tensors, and the cost model. |
| `intermediates.cc` | Shared intermediates (level 2), hoisting (level 3), and merging (level 4). |
| `build_program.cc` | Turns terms into statements and frees intermediates after their last use. |
| `printer.h`, `printer.cc`, `numpy_printer.cc`, `tiledarray_printer.cc` | The code printers. A new backend implements one statement at a time. |
| `pq_opt.cc`, `pq_opt.h` | The Python class and its options. |

**The label rule.** In a term, every fermion label appears either once (external, and then also
on the left-hand side) or twice, in two different tensors (summed). Ingest rewrites traces as
contractions with `Id`, so later passes never see a repeated label within one tensor. For
bosons, the kind of label decides instead: a free mode label is external, and a canonical dummy
is summed.

## Tests

| Test | Checks |
|---|---|
| `tests/pq_opt_equivalence_test.py` | Code from every level, on random antisymmetric tensors, agrees with level 0. Includes CC, EOM, Hbar, lambda, spin-blocked and QED cases. |
| `tests/pq_opt_blocks_test.py` | Spin blocks and permutation operators, including paired ones, against spin-orbital equations. |
| `tests/pq_opt_coefficient_test.py` | Coefficient snapping. |
| `tests/pq_opt_neo_test.py` | Nuclear-electronic (NEO) residuals against pq_graph. |
| `tests/pq_opt_numerical_test.py` | The pq_graph test harnesses, run through pq_opt. |
| `tests/pq_numerical_test.py` | Every generated method, on real molecules, at `opt_level` 4. |
