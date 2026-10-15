#!/usr/bin/env python
"""
Exact ground-state energy of the Fig. 1 pairing model as 8 SPINLESS modes with N = 3,
for a single input g.  OpenFermion Hamiltonian + dense diagonalization of the
56 x 56 three-fermion block (all "Sz" sectors, i.e. only N is fixed).

    H = sum_i eps_i a_i^+ a_i  -  (g*scale/8) sum_ij x_i x_j a_i^+ a_ibar^+ a_jbar a_j ,   ibar = i ^ 1

scale = 16 reproduces Fig. 1 (see reproduce_fig1.py); scale = 1 is Eqs. (65)-(67) as printed.

usage:  python spinless_exact.py G [--scale 16] [--nroots 4]
"""
import argparse
import numpy as np
from openfermion import FermionOperator
from openfermion.linalg import get_sparse_operator, jw_number_restrict_operator

R = 8                                                             # modes
N = 3                                                             # fermions
EPS = np.array([1., 1., 2., 2., 3., 3., 4., 4.])                   # Eq. (66)
X = np.array([1., -1., 2., -2., 3., -3., 4., -4.]) / (2.0 * np.sqrt(15.0))   # Eq. (67)


def hamiltonian(g, scale=16.0):
    """FermionOperator for Eq. (65) with the interaction multiplied by scale."""
    H = FermionOperator()
    for i in range(R):
        H += FermionOperator(((i, 1), (i, 0)), EPS[i])
    for i in range(R):
        for j in range(R):
            H += FermionOperator(((i, 1), (i ^ 1, 1), (j ^ 1, 0), (j, 0)),
                                 -g * scale * X[i] * X[j] / 8.0)
    return H


def exact(g, scale=16.0, nroots=4):
    """Lowest eigenvalues of H in the N-fermion block (dimension C(8,3) = 56)."""
    Hs = get_sparse_operator(hamiltonian(g, scale), n_qubits=R)
    H3 = jw_number_restrict_operator(Hs, N, n_qubits=R).toarray().real
    return np.linalg.eigvalsh(H3)[:nroots]

R = 8
EPS = np.array([1., 1., 2., 2., 3., 3., 4., 4.])
X = np.array([1., -1., 2., -2., 3., -3., 4., -4.]) / (2.0 * np.sqrt(15.0))
bar = lambda i: i ^ 1

def spinless_integrals(g, scale=16.0):
    lam = g * scale
    oei = np.diag(EPS)
    tei = np.zeros((R, R, R, R))
    for i in range(R):
        for j in range(R):
            tei[i, j, bar(i), bar(j)] += -(lam / 4.0) * X[i] * X[j]
    return oei, tei

def csr_to_F(c, A, dims):
    from libsdp.sdp_helper import sdp_solver, sdp_matrix
    offs = np.cumsum([0] + [d*d for d in dims])
    def dec(idx):
        blk = int(np.searchsorted(offs, idx, side='right') - 1); loc = int(idx - offs[blk]); n = dims[blk]
        return blk + 1, loc // n + 1, loc % n + 1
    F = []; m = sdp_matrix()
    for idx in np.nonzero(c)[0]:
        b_, r_, c_ = dec(idx); m.block_number.append(b_); m.row.append(r_); m.column.append(c_); m.value.append(float(c[idx]))
    F.append(m); A = A.tocsr()
    for i in range(A.shape[0]):
        m = sdp_matrix()
        for idx, v in zip(A.indices[A.indptr[i]:A.indptr[i+1]], A.data[A.indptr[i]:A.indptr[i+1]]):
            b_, r_, c_ = dec(idx); m.block_number.append(b_); m.row.append(r_); m.column.append(c_); m.value.append(float(v))
        F.append(m)
    return F

def run_bpsdp(g, tol=1e-6):
    import time
    from g2_v2rdm.g2_v2rdm_sdp_auto import g2_v2rdm_sdp
    import libsdp
    from libsdp.sdp_helper import sdp_solver, sdp_matrix
    ALG = 'bpsdp'
    oei, tei = spinless_integrals(g)
    sdp = g2_v2rdm_sdp(3, 0, R, oei, tei, q1=True, q2=True, g2=True, constrain_spin=False,)
    b = np.array(sdp.b); c = np.array(sdp.c); dims = list(sdp.block_dimensions)
    F = csr_to_F(c, sdp.A_mat, dims)
    o = libsdp.sdp_options(); o.sdp_algorithm = ALG; o.sdp_error_convergence = tol; o.sdp_objective_convergence = tol
    o.cg_convergence = 1e-12; o.mu_update_frequency = 500; o.guess_type = 'random' if ALG == 'rrsdp' else 'zero'; o.print_level = 5 
    s = sdp_solver(o, F, dims); t0 = time.time()
    x = np.array(s.solve(b, 200000)); cc = np.array(s.get_c()); y = np.array(s.get_y())
    perr = float(np.linalg.norm(np.array(s.get_Au(x)) - b))
    return float(cc @ x), abs(float(cc @ x) - float(b @ y)), perr, sdp.n_primal, len(b), time.time() - t0

def make_plot(gvals=None):
    import matplotlib.pyplot as plt
    import libsdp
    from libsdp.sdp_helper import sdp_solver
    from libsdp.sdpa_file_io import clean_sdpa_problem
    from sosfermion.fermionic_sos import FermionicSOS
    import jax
    if __package__:
        from .quartic_sos import run_quartic_sos
    else:
        from quartic_sos import run_quartic_sos
    if gvals is None:
        gvals = [2]# np.linspace(0, 2, 4)

    # Use boundary-point SDP, as in run_bpsdp, with fixed CG accuracy.
    maxiter = 200000
    options = libsdp.sdp_options()
    options.sdp_algorithm = 'bpsdp'
    options.sdp_error_convergence = 1e-6
    options.sdp_objective_convergence = 1e-6
    options.cg_convergence = 1e-12
    options.dynamic_cg_convergence = False
    # Keep the penalty fixed: at g=0 its residual-ratio update can divide by
    # an almost-zero dual residual and destabilize this SOS problem.
    options.mu_update_frequency = maxiter + 1
    options.guess_type = 'zero'
    options.print_level = 5
    options.outfile = ''

    gs_energies = []
    dqg_energies = []
    fock_dqg_energies = []
    fermionic_sos_energies = []
    weighted_sos_energies = []
    for gg in gvals:
        gs_energy = exact(gg, nroots=1)
        gs_energies.append(gs_energy)
        print(f"{gs_energy=}")
        exit()
        res = run_bpsdp(gg)
        dqg_energies.append(res[0])
        res = run_quartic_sos(gg)
        fock_dqg_sos = res['energy']
        fock_dqg_energies.append(fock_dqg_sos)

        # Embed each original mode i as i_alpha, using all R orbital labels.
        # The full integrals already have the right scaling for same-spin terms.
        # FermionicSOS also creates R beta modes; their Hamiltonian will be zero.
        oei, tei = spinless_integrals(gg)
        sdp = FermionicSOS(R, oei, tei,
                           d1=True, q1=True, d2=True, q2=True,
                           g2_aa=True, g2_ab=True, basis_has_one=False)

        # Its constructor copies the integrals into both spins. Set every beta
        # or mixed-spin target to zero, keeping the rows to enforce those zeros.
        # F[0] is the objective; F[1:] corresponds to the entries of b.
        for k, row in enumerate(sdp.F[1:]):
            if 'b' in row.spin_idx:
                sdp.b[k] = row.b_val = 0.0

        # The g2 basis already spans identity via a^dagger a + a a^dagger.
        # This SOS has no particle-number restriction, as in run_quartic_sos.
        # Beta modes are spectators; alpha-only embedding does not fix occupations.
        b, F = clean_sdpa_problem(sdp.b, sdp.F)
        solver = sdp_solver(options, F, sdp.block_dimensions)
        gram_solution = np.array(solver.solve(b, maxiter))
        # Check convergence before including this point in the comparison.
        objective = float(np.dot(solver.get_c(), gram_solution))
        residual = np.linalg.norm(np.array(solver.get_Au(gram_solution)) - b)
        gap = abs(objective - np.dot(b, solver.get_y()))
        dual_residual = np.linalg.norm(np.array(solver.get_ATu(solver.get_y()))
                                       + solver.get_z() - np.array(solver.get_c()))
        if not (residual <= 1e-6 and dual_residual <= 1e-6 and gap <= 1e-6):
            raise RuntimeError(f'FermionicSOS at g={gg}: primal={residual:.3e}, '
                               f'dual={dual_residual:.3e}, gap={gap:.3e}')
        # The objective is the SOS constant C0; H-E*I=SOS gives E=-C0.
        fermionic_sos_energies.append(-objective)

        wsosDQG = run_quartic_sos(gg, nparticles=3)
        weighted_sos_energies.append(wsosDQG['energy'])

    print(f"{gs_energies=}")
    plt.plot(gvals, gs_energies, marker='o', linestyle='-', label='Exact (N=3)')
    plt.plot(gvals, dqg_energies, marker='o', linestyle='-', label='v2RDM-DQG')
    plt.plot(gvals, weighted_sos_energies, marker='o', linestyle='-', label='wSOS-DQG', mfc=None, markersize=10, mec='k')
    plt.plot(gvals, fock_dqg_energies, marker='o', linestyle='-', label='sosnull-SOS-DQG')
    plt.plot(gvals, fermionic_sos_energies, marker='s', linestyle='--', label='FermionicSOS (alpha-only DQG)')
    plt.legend(loc='upper right', fontsize=14, frameon=False)
    plt.show()

if __name__ == '__main__':
    make_plot()
    # run_bpsdp(g=0)
    exit()
    ap = argparse.ArgumentParser()
    ap.add_argument('g', type=float)
    ap.add_argument('--scale', type=float, default=16.0)
    ap.add_argument('--nroots', type=int, default=4)
    a = ap.parse_args()
    w = exact(a.g, a.scale, a.nroots)
    print(f'g = {a.g:g}   scale = {a.scale:g}   E0 = {w[0]:.10f}')
    print('lowest roots:', '  '.join(f'{e:.10f}' for e in w))
