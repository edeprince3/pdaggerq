"""Level-2 SOS with Wick maps, ADMM or RRSDP, and optional number shifts.

Run from the repository root in sosnullspacer-pdaggerq:
    OMP_NUM_THREADS=1 python -m examples.sos.quartic_sos
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

if __package__:
    from .sos_null import (
        build_wick_maps as _build_wick_maps,
        hamiltonian_coefficients,
        quartic_particle_number_shift_coefficients as particle_number_shift_coefficients,
        solve_hamiltonian_admm,
    )
else:
    from sos_null import (
        build_wick_maps as _build_wick_maps,
        hamiltonian_coefficients,
        quartic_particle_number_shift_coefficients as particle_number_shift_coefficients,
        solve_hamiltonian_admm,
    )


def build_wick_maps(norb):
    """Return C0,C2,C4 maps for all charge sectors through two ladders."""
    sectors = {0: ("", "xo", "ox"), 1: ("x",), -1: ("o",), 2: ("xx",), -2: ("oo",)}
    return _build_wick_maps(norb, sectors, max_degree=4)


def build_problem(norb, nparticles=None):
    """Match H-E*I = SOS + f*(Nhat-N) + (Nhat-N)*f^dagger.

    Omitting nparticles gives the original unweighted SOS problem.
    """
    if nparticles is not None and (int(nparticles) != nparticles or not 0 <= nparticles <= norb):
        raise ValueError("nparticles must be an integer between 0 and norb")
    gram_coefficients, dimensions = build_wick_maps(norb)

    # f has one identity coefficient and two full ordered norb-by-norb tensors.
    # Represent every unrestricted real f_k as p_k-m_k, with p_k,m_k >= 0.
    # These are separate 1x1 PSD blocks; f, xo, and ox themselves are not PSD.
    # At eight modes, 129 free coefficients require 258 scalar blocks.
    nshift = 1 + 2 * norb**2
    if nparticles is not None:
        dimensions.update({f"shift_{sign}_{k}": 1
                           for k in range(nshift) for sign in ("plus", "minus")})
    @jax.jit
    def canonical_coefficients(grams):
        """Return canonical coefficients of the SOS plus its optional number shift."""
        sos = gram_coefficients(grams)
        if nparticles is None:
            return sos
        # Unpack f in the order [constant, xo.ravel(), ox.ravel()], in C order.
        free_coefficients = jnp.stack([
            grams[f"shift_plus_{k}"][0, 0] - grams[f"shift_minus_{k}"][0, 0]
            for k in range(nshift)])
        xo = free_coefficients[1:1 + norb**2].reshape(norb, norb)
        ox = free_coefficients[1 + norb**2:].reshape(norb, norb)
        shift = particle_number_shift_coefficients(free_coefficients[0], xo, ox, nparticles)
        return tuple(c + q for c, q in zip(sos, shift, strict=True))

    # Flatten full coefficient tensors only to match libsdp's single-array interface.
    @jax.jit
    def AX_fn(grams):
        _, c2, c4 = canonical_coefficients(grams)
        return {"y": jnp.concatenate((c2.ravel(), c4.ravel()))}

    # Minimize the total constant (SOS + shift) to maximize E=H0-C0.
    # In particular, the shift's -N*s0 constant must enter this objective.
    @jax.jit
    def CX_fn(grams):
        return canonical_coefficients(grams)[0]

    return canonical_coefficients, dimensions, AX_fn, CX_fn


def run_quartic_sos(g, nparticles=None, tolerance=5e-6, rank=32, maxiter=5000,
                    disable_jit=False, solver="admm", rrsdp_options=None):
    """Solve the Richardson model with the quartic SOS maps and optional shift."""
    if __package__:
        from .spinless_exact import R, spinless_integrals
    else:
        from spinless_exact import R, spinless_integrals

    hamiltonian_tensors = hamiltonian_coefficients(*spinless_integrals(g, scale=16.0))
    return solve_hamiltonian_admm(
        hamiltonian_tensors, build_problem(R, nparticles=nparticles),
        tolerance=tolerance, rank=rank, maxiter=maxiter, disable_jit=disable_jit,
        solver=solver, rrsdp_options=rrsdp_options)


def main(g=0.0, nparticles=3, maxiter=5000, disable_jit=False,
         solver="admm", rrsdp_options=None):
    result = run_quartic_sos(
        g, nparticles=nparticles, tolerance=1e-5, maxiter=maxiter, disable_jit=disable_jit,
        solver=solver, rrsdp_options=rrsdp_options)
    print({key: value for key, value in result.items() if key != "grams"})
    return result


if __name__ == "__main__":
    # At g=0, N=3 the exact energy is 4. Set nparticles=None for ordinary SOS.
    main()
