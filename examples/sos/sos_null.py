"""Shared coefficient maps, number shifts, and solvers for the SOS examples.

The quartic and sextic scripts choose their own generator families and number-
shift parameterizations. This module does not import either example or its model.
"""
from contextlib import chdir
from itertools import permutations
from math import factorial
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from pdaggerq.sosnull import symbolic


def hamiltonian_coefficients(oei, tei):
    """Convert spinless_integrals output to C0, C2, C4 for the pairing model.

    Our C4[p,q,r,s] multiplies a_p^dagger a_q^dagger a_s a_r, with no
    prefactor outside the sum over all ordered indices.
    """
    # tei[i,j,bar(i),bar(j)] stores -g*scale*X[i]*X[j]/4. Move the
    # two creation indices first and include the integral convention's 1/2:
    # C4[i,bar(i),j,bar(j)] = -g*scale*X[i]*X[j]/8, as in scaled Eq. (65).
    c4 = 0.5 * tei.transpose(0, 2, 1, 3)

    # Here X[bar(i)] = -X[i], so C4 is already antisymmetric in each pair.
    # No additional antisymmetrization factor or annihilator reversal is needed.
    return 0.0, oei, c4


def antisymmetrize(coefficient):
    """Normalize antisymmetry separately over creator and annihilator axes."""
    # C4 has normalization (2!)**2; C6 has (3!)**2. Project each group in turn.
    rank = coefficient.ndim // 2
    for start in (0, rank):
        terms = []
        for permutation in permutations(range(rank)):
            inversions = sum(permutation[i] > permutation[j]
                             for i in range(rank) for j in range(i + 1, rank))
            axes = list(range(2 * rank))
            axes[start:start + rank] = [start + i for i in permutation]
            terms.append((-1)**inversions * coefficient.transpose(axes))
        coefficient = sum(terms) / factorial(rank)
    return coefficient


def quartic_particle_number_shift_coefficients(constant, xo, ox, nparticles):
    """Canonical coefficients of f*(Nhat-N) + (Nhat-N)*f^dagger.

    As in fermionic_dual Eqs. (47)-(50), f has free real coefficients in
    {I, a_i^dagger a_j, a_i a_j^dagger}. These coefficients need not be PSD.
    The weighted SDP represents each real coefficient as a difference of
    two nonnegative 1x1 PSD blocks.
    """
    # Normal order f and add its adjoint: f+f^dagger = s0*I + sum_ij s_ij a_i^ a_j.
    # The ox term contributes tr(ox)*I - sum_ij ox[j,i]*a_i^ a_j.
    s0 = 2 * (constant + jnp.trace(ox))
    one_body = xo - ox.T
    s = one_body + one_body.T
    identity = jnp.eye(xo.shape[0])

    # All terms in f conserve particle number, so the shift is (f+f^dagger)*(Nhat-N).
    # Normal ordering gives Q0=-N*s0, Q2=s0*I+(1-N)*s, and Q4=antisym(s tensor I).
    c4 = jnp.einsum("ij,kl->ikjl", s, identity)
    c4 = (c4 - c4.swapaxes(0, 1) - c4.swapaxes(2, 3) + c4.transpose(1, 0, 3, 2)) / 4
    return -nparticles * s0, s0 * identity + (1 - nparticles) * s, c4


def particle_number_shift_coefficients(constant, one_body, two_body, nparticles):
    """Return C0,C2,C4,C6 of f*r + r*f^dagger, where r=Nhat-nparticles.

    f has free canonical coefficients through FOUR ladder operators:
    two_body[i,j,k,l] multiplies a_i^dagger a_j^dagger a_l a_k.
    No factorial prefactors are implicit. For Hermitian f the result is 2*f*r.
    """
    identity = jnp.eye(one_body.shape[0])

    # Normal order f*r. Contracting Nhat's creator gives one copy of a one-body
    # term or two copies of a two-body term. Uncontracted products add two ladders;
    # their canonical axes are (i,p,j,q) and (i,j,p,k,l,q), with delta[p,q].
    fr0 = -nparticles * constant
    fr2 = constant * identity + (1 - nparticles) * one_body
    fr4 = antisymmetrize((2 - nparticles) * two_body
                        + jnp.einsum("ij,pq->ipjq", one_body, identity))
    fr6 = antisymmetrize(jnp.einsum("ijkl,pq->ijpklq", two_body, identity))

    # Add (f*r)^dagger = r*f^dagger. Canonical adjoints conjugate and exchange
    # the complete creator/annihilator groups; no within-group reversal is needed.
    return (fr0 + jnp.conj(fr0), fr2 + fr2.conj().T,
            fr4 + fr4.conj().transpose(2, 3, 0, 1),
            fr6 + fr6.conj().transpose(3, 4, 5, 0, 1, 2))


def build_wick_maps_multipliers(norb, nparticles, multiplier_degree=4):
    """Build a JAX map from the selected coefficients of f to C0,C2,C4,C6.

    multiplier_degree=0 takes f0; 2 takes (f0,F2); 4 takes (f0,F2,F4).
    The map returns f*r + r*f^dagger for r=Nhat-nparticles.
    Inputs use canonical coefficient order, with no factorial prefactors.
    The analytical particle_number_shift_coefficients remains as a reference.
    """
    # Validate the orbital count, particle-number sector, and maximum ladder degree.
    if not isinstance(norb, (int, np.integer)) or norb < 1:
        raise ValueError("norb must be a positive integer")
    if int(nparticles) != nparticles or not 0 <= nparticles <= norb:
        raise ValueError("nparticles must be an integer between 0 and norb")
    if not isinstance(multiplier_degree, (int, np.integer)) or multiplier_degree not in (0, 2, 4):
        raise ValueError("multiplier_degree must be 0, 2, or 4")

    # Normal order f*N and f*I for each degree of f once, before JAX tracing.
    # All generators have charge zero; tau=xo supplies N through delta[p,q].
    patterns = ("", "xo", "xxoo")[:multiplier_degree // 2 + 1]
    labels = {d: (tuple("IJK"[:d // 2]), tuple("MNO"[:d // 2])) for d in (0, 2, 4, 6)}
    updates = []
    for sigma in patterns:
        for tau in ("xo", ""):
            for branch in symbolic.generate_wickbranch_antisymm(sigma, tau, antisymmetrize=True):
                assert (branch.sigma, branch.tau) == (sigma, tau)
                updates.append(symbolic.generate_einsum(branch, labels))
    wick_code = compile("\n".join(updates), "<particle-number multiplier Wick maps>", "exec")

    @jax.jit
    def canonical_coefficients(constant, one_body=None, two_body=None):
        # Supply exactly the degrees selected when building this map.
        if (one_body is not None) != (multiplier_degree >= 2) or (
                (two_body is not None) != (multiplier_degree >= 4)):
            raise ValueError("supply coefficients only through multiplier_degree")

        # Since [f,N]=0, fr+rf^dagger = 2*HermitianPart(f)*r. This also handles
        # ADMM's nonsymmetric intermediate matrices while preserving a linear map.
        words = [jnp.real(constant)]
        if multiplier_degree >= 2:
            words.append((one_body + one_body.conj().T) / 2)
        if multiplier_degree >= 4:
            two_body = (two_body + two_body.conj().transpose(2, 3, 0, 1)) / 2
            # F4[i,j,k,l] multiplies a_i^dagger a_j^dagger a_l a_k. Convert its
            # canonical axes to xxoo's literal word labels; f0 and F2 need no swap.
            words.append(two_body.swapaxes(2, 3))
        dtype = jnp.result_type(*words)
        identity = jnp.eye(norb, dtype=dtype)
        blocks = {}
        for sigma, word_coefficients in zip(patterns, words):
            # axes=0 forms an outer product: keep all f and N indices for Wick
            # contraction. These coefficient blocks represent 2*f*N - 2*eta*f.
            blocks[sigma, "xo"] = 2 * jnp.tensordot(word_coefficients, identity, axes=0)
            blocks[sigma, ""] = -2 * nparticles * word_coefficients

        # WickBranch already includes normalized antisymmetry for full tensors.
        values = {"einsum": jnp.einsum, "blocks_tensors": {0: blocks}}
        values.update({f"C_{d}": jnp.zeros((norb,) * d, dtype=dtype) for d in labels})
        exec(wick_code, values)
        return tuple(values[f"C_{d}"] for d in labels)

    return canonical_coefficients


def build_wick_maps(norb, sectors, max_degree):
    """Build coefficient maps for the supplied charge-sector generator families.

    ``max_degree`` selects C0,C2,C4 (4) or C0,C2,C4,C6 (6). Each
    Gram keeps the full ordered orbital tuples and all within-sector pairs.
    """
    if not isinstance(norb, (int, np.integer)) or norb < 1:
        raise ValueError("norb must be a positive integer")
    if not isinstance(max_degree, (int, np.integer)) or max_degree not in (4, 6):
        raise ValueError("max_degree must be 4 or 6")
    slices, dimensions = {}, {}
    for charge, patterns in sectors.items():
        # A length-k generator has norb**k ordered orbital tuples. Reserve its
        # consecutive row/column slice without reducing or reversing those labels.
        slices[charge], start = {}, 0
        for pattern in patterns:
            stop = start + norb**len(pattern)
            slices[charge][pattern], start = slice(start, stop), stop
        dimensions[str(charge)] = start
    # Include every within-sector family pair, including off-diagonal blocks.
    labels = {d: (tuple("IJK"[:d // 2]), tuple("MNO"[:d // 2])) for d in range(0, max_degree + 1, 2)}
    updates = []
    for charge, patterns in sectors.items():
        for sigma in patterns:
            for tau in patterns:
                for branch in symbolic.generate_wickbranch_antisymm(sigma, tau, antisymmetrize=True):
                    assert (branch.sigma, branch.tau) == (sigma, tau)
                    assert branch.sigma.count("x") - branch.sigma.count("o") == charge
                    updates.append(symbolic.generate_einsum(branch, labels))
    wick_code = compile("\n".join(updates), "<SOS Wick maps>", "exec")

    @jax.jit
    def canonical_coefficients(grams):
        # Unroll each block in C order: Gram row I and column J become tensor
        # axes (I0,...,J0,...). WickBranch handles the adjoint operator ordering.
        tensors = {
            charge: {(s, t): grams[str(charge)][rows, cols].reshape((norb,) * (len(s) + len(t)))
                     for s, rows in blocks.items() for t, cols in blocks.items()}
            for charge, blocks in slices.items()}
        values = {"einsum": jnp.einsum, "blocks_tensors": tensors}
        dtype = jnp.result_type(*[grams[str(charge)] for charge in sectors])
        values.update({f"C_{d}": jnp.zeros((norb,) * d, dtype=dtype) for d in labels})
        exec(wick_code, values)
        return tuple(values[f"C_{d}"] for d in labels)

    return canonical_coefficients, dimensions


def solve_hamiltonian_admm(hamiltonian_tensors, problem, tolerance=1e-5,
                           rank=32, maxiter=5000, disable_jit=False,
                           solver="admm", rrsdp_options=None):
    """Supply A, its adjoint, and C directly to libsdp; no assembled A matrix.

    solver="admm" keeps the operator-splitting solver. solver="rrsdp" hands the
    same maps to libsdp's low-rank augmented-Lagrangian (Burer-Monteiro) solver,
    whose outer ALM loop converges more stably on some instances. Both paths
    return the identical result dictionary. rrsdp_options is merged over the
    RRSDP defaults chosen below, e.g. rrsdp_options={"sigma_scaling": 5.0}.
    """

    _, dimensions, AX_fn, CX_fn = problem
    # Match every nonconstant degree, including C6 when called by sextic_sos.
    rhs = jnp.concatenate([jnp.asarray(c).ravel() for c in hamiltonian_tensors[1:]])

    # ADMM additionally needs A*: {'y': coefficients} -> the Gram-block dictionary.
    # Transpose the linear einsum map automatically, using zeros only for its shapes.
    template = {name: jnp.zeros((size, size)) for name, size in dimensions.items()}

    # Build the transpose inside JIT so diagonal masks and zero tensors remain
    # traced operations instead of large arrays captured by the adjoint's closure.
    @jax.jit
    def adjoint_coefficients(y):
        return jax.linear_transpose(AX_fn, template)(y)[0]

    # Partial makes the callback a valid argument to libsdp's compiled kernels.
    AT_fn = jax.tree_util.Partial(adjoint_coefficients)

    # Keep multiplier factors full-width so each PSD matrix difference remains
    # free. An optional rank cap applies only to the SOS Grams.
    block_names = list(dimensions)
    shapes = [(size, size) for size in dimensions.values()]
    primal_rank = [size if rank is None or name.startswith("shift_") else min(size, rank)
                   for name, size in dimensions.items()]

    # Use the library's solvers with the supplied SOS and multiplier blocks.
    # ADMM always writes a final checkpoint, so put that scratch file in a
    # temporary cwd. disable_jit=True is useful for small comparisons;
    # primitive operations still compile.
    np.random.seed(23)
    start = perf_counter()
    if solver == "admm":
        from libsdp.run_admm import run_admm

        with TemporaryDirectory(prefix="sos-admm-") as scratch, chdir(scratch), jax.disable_jit(disable_jit):
            U, V, y, _ = run_admm(
                AX_fn=jax.tree_util.Partial(AX_fn), AT_fn=AT_fn,
                CX_fn=jax.tree_util.Partial(CX_fn), b_jax=rhs,
                shapes=shapes, block_names=block_names, primal_rank=primal_rank,
                # Allow headroom for the independent PSD-Gram and dual-slack checks.
                maxiter=maxiter, stopping_epsilon=0.01 * tolerance,
                gap_stopping_epsilon=0.01 * tolerance, save_frequency=None,
                update_frequency=1,
                tau=20,
                # Retain residual balancing, but disable the automatic penalty ramp
                # near feasibility: that ramp degraded dual convergence in this example.
                sigma_scaling=1.01,
                heuristic_factor=1.1, post_heuristic_sigma_scaling=1.0,
            )
        # ADMM uses U V^T, which need not be PSD until U=V. Average the factors
        # to obtain actual PSD Grams.
        factors = {name: (U[name] + V[name]) / 2 for name in dimensions}
        consensus = float(jnp.sqrt(sum(jnp.sum((U[k] - V[k])**2) for k in dimensions)))
    elif solver == "rrsdp":
        from libsdp.run_rrsdp import run_rrsdp

        # RRSDP needs no adjoint, and its maxiter counts *outer* ALM updates
        # rather than the inner steps ADMM's maxiter counts, so the two are not
        # interchangeable; maxiter caps the inner L-BFGS solve instead.
        options = dict(sigma=10.0, sigma_scaling=10.0, maxiter=10,
                       lbfgs_max_iter=min(maxiter, 50_000), lbfgs_history_size=6,
                       lbfgs_tol_min=1e-8, print_frequency=1, verbose=True)
        options.update(rrsdp_options or {})
        with jax.disable_jit(disable_jit):
            R_factors, s_params = run_rrsdp(
                AX_fn=jax.tree_util.Partial(AX_fn),
                CX_fn=jax.tree_util.Partial(CX_fn), b_jax=rhs,
                shapes=shapes, block_names=block_names, primal_rank=primal_rank,
                stopping_epsilon=0.01 * tolerance,
                gap_stopping_epsilon=0.01 * tolerance, **options)
        # RRSDP optimizes X = R R^T directly, so the Grams are PSD by
        # construction and there is no consensus gap to close.
        factors = R_factors
        y = s_params["y"]
        consensus = 0.0
    else:
        raise ValueError(f"solver must be 'admm' or 'rrsdp', got {solver!r}")
    elapsed = perf_counter() - start

    # Recheck the constraints and objective on the resulting PSD Grams.
    grams = {name: factor @ factor.T for name, factor in factors.items()}
    objective = float(CX_fn(grams))
    # Preserve both residual measures used by quartic_sos and its callers.
    coefficient_residual = AX_fn(grams)["y"] - rhs
    residual = float(jnp.linalg.norm(coefficient_residual))
    coefficient_error = float(jnp.max(jnp.abs(coefficient_residual)))
    gap = abs(objective - float(jnp.vdot(rhs, y["y"]).real))

    # With libsdp's multiplier sign, dual slack is C-A*y. Check every symmetric
    # block, including both signs of each free number-shift coefficient.
    slacks = jax.grad(lambda g: CX_fn(g) - jnp.vdot(y["y"], AX_fn(g)["y"]).real)(grams)
    min_slack = min(float(jnp.linalg.eigvalsh((s + s.T) / 2)[0]) for s in slacks.values())
    return {"grams": grams, "energy": float(hamiltonian_tensors[0]) - objective,
            "objective": objective, "residual": residual, "gap": gap,
            "coefficient_error": coefficient_error, "solver": solver,
            "min_slack": min_slack, "consensus": consensus, "seconds": elapsed,
            "converged": max(residual, gap, consensus) <= tolerance and min_slack >= -tolerance}
