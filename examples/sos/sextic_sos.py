"""Full level-(2,3) SOS with Wick maps, ADMM, and an optional quartic number shift.

Independent PSD Grams span every charge sector through three ladder operators.
This contains T1, T2, and T2' certificates; it does not restrict the cubic Grams
to the shared-Gram ansatz for those individual conditions. For a two-body H,
match C2 and C4 and require total C6=0, including the number shift when present.

Run from the repository root in sosnullspacer-pdaggerq:
    OMP_NUM_THREADS=1 python -u -m examples.sos.sextic_sos

ADMM reports warmup timings for its U and V updates before starting iterations.
For more detailed JAX tracing, lowering, and compilation messages:
    JAX_LOG_COMPILES=1 OMP_NUM_THREADS=1 python -u -m examples.sos.sextic_sos
"""
SET_ID = 1
import os
# Restrict JAX to only see the selected GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = f"{SET_ID}"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

if __package__:
    from .sos_null import (
        antisymmetrize,
        build_wick_maps as _build_wick_maps,
        build_wick_maps_multipliers,
        hamiltonian_coefficients,
        particle_number_shift_coefficients,
        solve_hamiltonian_admm,
    )
else:
    from sos_null import (
        antisymmetrize,
        build_wick_maps as _build_wick_maps,
        build_wick_maps_multipliers,
        hamiltonian_coefficients,
        particle_number_shift_coefficients,
        solve_hamiltonian_admm,
    )


def build_wick_maps(norb):
    """Return C0,C2,C4,C6 maps for all seven charge sectors through three ladders."""
    # The cubic/linear cross terms include T2' certificates; +/-3 completes T1.
    sectors = {0: ("", "xo", "ox"), 1: ("oxx", "x"), -1: ("xoo", "o"),
               2: ("xx",), -2: ("oo",), 3: ("xxx",), -3: ("ooo",)}
    return _build_wick_maps(norb, sectors, max_degree=6)


def build_problem(norb, nparticles=None, multiplier_degree=4):
    """Match H-E*I = SOS + f*r + r*f^dagger with f of ladder degree <=0, 2, or 4.

    Each included degree has a plus/minus PSD pair. nparticles=None omits f.
    """
    if nparticles is not None and (int(nparticles) != nparticles or not 0 <= nparticles <= norb):
        raise ValueError("nparticles must be an integer between 0 and norb")
    if not isinstance(multiplier_degree, (int, np.integer)) or multiplier_degree not in (0, 2, 4):
        raise ValueError("multiplier_degree must be 0, 2, or 4")
    gram_coefficients, dimensions = build_wick_maps(norb)
    gram_names = tuple(dimensions)
    print(f"{gram_names=}")

    # Only f+f^dagger contributes because f conserves number. Represent each
    # Hermitian coefficient matrix as F_plus-F_minus with both matrices PSD.
    # Degree 0 has one label, degree 2 has norb labels, and degree 4 has norb**2
    # ordered pair labels. Allocate only degrees up to multiplier_degree: at eight
    # modes the PSD block sizes are (1,1), (1,1,8,8), or (1,1,8,8,64,64).
    if nparticles is not None:
        multiplier_coefficients = build_wick_maps_multipliers(norb, nparticles, multiplier_degree)
        dimensions.update({f"shift_{sign}_{degree}": size
                           for degree, size in ((0, 1), (2, norb), (4, norb**2))
                           if degree <= multiplier_degree
                           for sign in ("plus", "minus")})

    @jax.jit
    def canonical_coefficients(grams):
        # Add the Gram and multiplier Wick maps in the same canonical tensors.
        sos = gram_coefficients({name: grams[name] for name in gram_names})
        if nparticles is None:
            return sos
        # These differences are coefficients of f itself. The multiplier map
        # takes its Hermitian part and includes the factor of two from fr+rf^dagger.
        # F4 rows (i,j) and columns (k,l) reshape directly to canonical (i,j,k,l).
        free_coefficients = [
            (grams[f"shift_plus_{degree}"] - grams[f"shift_minus_{degree}"])
            .reshape((norb,) * degree, order="C")
            for degree in range(0, multiplier_degree + 1, 2)]
        shift = multiplier_coefficients(*free_coefficients)
        return tuple(c + q for c, q in zip(sos, shift, strict=True))

    # Match TOTAL coefficients. For a two-body Hamiltonian the C6 target is zero,
    # but SOS C6 and shift C6 can be nonzero and cancel each other (Eq. 97-98).
    @jax.jit
    def AX_fn(grams):
        return {"y": jnp.concatenate([c.ravel() for c in canonical_coefficients(grams)[1:]])}

    @jax.jit
    def CX_fn(grams):
        return canonical_coefficients(grams)[0]

    return canonical_coefficients, dimensions, AX_fn, CX_fn


def run_sextic_sos(g, nparticles=None, tolerance=5e-6, rank=None, maxiter=100_000,
                    disable_jit=False, multiplier_degree=4, solver="admm",
                    rrsdp_options=None):
    """Solve with ADMM, allowing multiplier ladder degree 0, 2, or 4 (default).

    rank caps SOS factors only, and maxiter counts ADMM steps. solver="rrsdp"
    swaps in libsdp's low-rank ALM solver instead, where maxiter caps the inner
    L-BFGS solve and rrsdp_options tunes the outer loop.
    """
    if __package__:
        from .spinless_exact import R, spinless_integrals
    else:
        from spinless_exact import R, spinless_integrals

    hamiltonian_tensors = hamiltonian_coefficients(*spinless_integrals(g, scale=16.0))
    hamiltonian_tensors += (np.zeros((R,) * 6),)
    problem = build_problem(R, nparticles=nparticles, multiplier_degree=multiplier_degree)
    gram_dimensions = {k: n for k, n in problem[1].items() if not k.startswith("shift_")}
    print(f"{gram_dimensions=}")
    print(f"Full cubic SOS: g={g}, nparticles={nparticles}; Gram dimensions: {gram_dimensions}", flush=True)
    if nparticles is not None:
        shift_dimensions = {k: n for k, n in problem[1].items() if k.startswith("shift_")}
        print(f"Free multiplier f (degree <= {multiplier_degree}): "
              f"PSD difference block dimensions: {shift_dimensions}", flush=True)
    return solve_hamiltonian_admm(hamiltonian_tensors, problem, tolerance=tolerance,
                                 rank=rank, maxiter=maxiter, disable_jit=disable_jit,
                                 solver=solver, rrsdp_options=rrsdp_options)


def main(g=2.0, nparticles=None, multiplier_degree=4, solver="admm",
         rrsdp_options=None):
    result = run_sextic_sos(g, nparticles=nparticles, multiplier_degree=multiplier_degree,
                            solver=solver, rrsdp_options=rrsdp_options)
    # matrix = get_sparse_operator(hamiltonian(g), n_qubits=R)
    # if nparticles is not None:
    #     matrix = jw_number_restrict_operator(matrix, nparticles, n_qubits=R)
    # exact = np.linalg.eigvalsh(matrix.toarray())[0]
    # print(f"SOS: {result['energy']:.9f}; exact: {exact:.9f}")
    # print(f"Coefficient error: {result['coefficient_error']:.3e}; gap: {result['gap']:.3e}; "
    #       f"minimum dual slack: {result['min_slack']:.3e}")
    # assert result["converged"], "ADMM did not meet the certificate/duality/consensus tolerances"
    # assert result["energy"] <= exact + 1e-5
    return result


if __name__ == "__main__":
    # Choose multiplier_degree=0, 2, or 4; nparticles=None disables the number shift.
    # main() # nparticles=3, multiplier_degree=0)
    # exit()
    # Include the quartic number shift; set nparticles=None for ordinary SOS.
    gvals = np.linspace(0, 2, 20)
    gvals = np.arange(0.0, 2.05, 0.05)
    g_sets = np.array_split(gvals, 8)
    print(gvals)
    print(g_sets)
    wdqgt_energies = []
    # for gg in gvals:
    for gg in [0.25]:# g_sets[SET_ID]:
        print(gg, SET_ID)
        result = main(g=0.25, nparticles=3, multiplier_degree=0)
        print(result['energy'])
        exit()
        # result = {'energy': 0}
        wdqgt_energies.append((gg, result['energy']))
        np.save(f"dqgt_multiplier_f0_sos_energies_set_idx_{SET_ID}.npy", wdqgt_energies)
