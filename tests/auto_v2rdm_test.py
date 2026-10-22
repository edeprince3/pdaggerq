"""
Correctness tests for pdaggerq.numerical.codegen.auto_v2rdm, independent of any other SDP
generator:

build every block of the SDP from an explicit N-electron state (Jordan-Wigner operators in
the full Fock space of a few orbitals, times photon-number states |A>, A = 0 .. n_photon, for
a cavity mode) and check that the resulting primal vector x satisfies every constraint,
A x = b, that c.x = <psi|H|psi>, and that every block is positive semidefinite.
"""
import itertools
import sys

import numpy as np
import scipy.sparse as sp

from pdaggerq.numerical.codegen.auto_v2rdm import v2rdm_sdp, ANTICOMMUTATOR, CREATOR


def fock_operators(nmo, d=1):
    """
    annihilation operators a[(p, spin)] in the Fock space of 2 nmo spin orbitals times d
    photon states (at index det * d + A), and photon operators P[A] = |0><A|
    """
    nso = 2 * nmo
    dim = 2 ** nso
    photon_identity = sp.identity(d, format='csr')
    a = {}
    for p, s in itertools.product(range(nmo), 'ab'):
        so = p + nmo * (s == 'b')
        rows, cols, vals = [], [], []
        for det in range(dim):
            if det >> so & 1:
                rows.append(det ^ (1 << so))
                cols.append(det)
                vals.append((-1) ** bin(det & ((1 << so) - 1)).count('1'))
        a[(p, s)] = sp.kron(sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim)), photon_identity, format='csr')
    P = [sp.kron(sp.identity(dim), sp.csr_matrix(([1.0], ([0], [A])), shape=(d, d)), format='csr') for A in range(d)]
    return a, P


def random_state(nmo, nalpha, nbeta, a, d=1, seed=0):
    """
    a random normalized state with nalpha alpha and nbeta beta electrons and maximal spin
    projection, S = (nalpha - nbeta) / 2, as the spin constraints assume, entangled with
    d photon states
    """
    rng = np.random.default_rng(seed)
    dets = [det * d + A for det in range(2 ** (2 * nmo)) for A in range(d)
            if bin(det & (2 ** nmo - 1)).count('1') == nalpha and bin(det >> nmo).count('1') == nbeta]

    # S^2 = Sz^2 + Sz + S- S+ in this (nalpha, nbeta) sector
    s_plus = sum(a[(p, 'a')].T @ a[(p, 'b')] for p in range(nmo))
    Sz = 0.5 * (nalpha - nbeta)
    S2 = (s_plus.T @ s_plus).toarray()[np.ix_(dets, dets)] + (Sz * Sz + Sz) * np.eye(len(dets))
    w, v = np.linalg.eigh(S2)
    v = v[:, np.abs(w - Sz * (Sz + 1)) < 1e-8]

    psi = np.zeros(2 ** (2 * nmo) * d)
    psi[dets] = v @ rng.normal(size=v.shape[1])
    return psi / np.linalg.norm(psi)


def primal_from_state(sdp, psi, a, P):
    """
    every block X(r A, c B) = < O_r O_c^dagger |A><B| > evaluated for psi, with the photon
    1-RDM d0(A, B) = < |A><B| >, and, for the photon-diagonal T1/T2/T2' block of photon state
    A, X(r, c) = < O_r O_c^dagger |A><A| > + < O_c^dagger O_r |A><A| > (the anticommutator,
    between three-body sectors). with V(r A) = (O_r^dagger |0><A|) psi and
    W(r A) = (O_r |0><A|) psi, these are V(r A).V(c B) and V(r A).V(c A) + W(c A).W(r A).
    """
    d = len(P)
    x = np.zeros(sdp.n_primal)
    for name, dim in zip(sdp.blocks, sdp.block_dimensions):
        t = sdp.block_type[name]
        A_fixed = sdp.block_photon.get(name)
        states = range(d) if A_fixed is None else [A_fixed]
        V, W = [], []
        if t == 'D0':
            V = [P[A] @ psi for A in states]
        for spins, off, sec in sdp.sectors[sdp.block_group[name]]:
            for tup in sec.tuples():
                O = sp.identity(len(psi), format='csr')
                for role, s, p in zip(sec.roles, spins, tup):
                    O = O @ (a[(p, s)].T if role == CREATOR else a[(p, s)])
                anticommutator = t in ANTICOMMUTATOR and len(spins) == 3
                for A in states:
                    V.append(O.T @ (P[A] @ psi))
                    W.append(O @ (P[A] @ psi) if anticommutator else 0.0 * psi)
        V = np.array(V)
        X = V @ V.T
        if W:
            W = np.array(W)
            X += W @ W.T
        x[sdp.block_offset[name]:sdp.block_offset[name] + dim * dim] = X.ravel()
    return x


def hamiltonian_expectation(psi, a, nmo, oei, tei, d=1, omega=0.0, coupling=None, linear=0.0):
    """
    <psi|H|psi> with H = sum h_pq p^ q + 1/2 sum (pq|rs) p^ r^ s q + omega b^ b
                         + sum g_pq p^ q (b + b^) + linear (b + b^),
    with b truncated to photon states 0 .. d-1
    """
    dim = len(psi) // d
    b = sp.kron(sp.identity(dim), sp.diags(np.sqrt(np.arange(1, d)), 1, shape=(d, d)), format='csr')
    e = psi @ ((omega * (b.T @ b) + linear * (b + b.T)) @ psi)
    for s in 'ab':
        for p, q in itertools.product(range(nmo), repeat=2):
            v = a[(p, s)].T @ (a[(q, s)] @ psi)
            e += oei[p, q] * psi @ v
            if coupling is not None:
                e += coupling[p, q] * psi @ ((b + b.T) @ v)
    for s, t in itertools.product('ab', repeat=2):
        for p, q, r, u in itertools.product(range(nmo), repeat=4):
            if tei[p, q, r, u] != 0.0:
                v = a[(u, t)] @ (a[(q, s)] @ psi)
                v = a[(p, s)].T @ (a[(r, t)].T @ v)
                e += 0.5 * tei[p, q, r, u] * psi @ v
    return e


def check(nmo, nalpha, nbeta, n_photon=None, **kwargs):
    rng = np.random.default_rng(1)
    oei = rng.normal(size=(nmo, nmo))
    oei = oei + oei.T
    tei = rng.normal(size=(nmo,) * 4)
    tei = tei + tei.transpose(1, 0, 2, 3)
    tei = tei + tei.transpose(0, 1, 3, 2)
    tei = tei + tei.transpose(2, 3, 0, 1)

    d = 1
    photon = {}
    if n_photon is not None:
        d = n_photon + 1
        coupling = rng.normal(size=(nmo, nmo))
        photon = dict(n_photon=n_photon, omega=0.7, photon_coupling=coupling + coupling.T, photon_linear=-0.3)

    sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, **photon, **kwargs)
    a, P = fock_operators(nmo, d)
    psi = random_state(nmo, nalpha, nbeta, a, d)
    x = primal_from_state(sdp, psi, a, P)

    residual = np.abs(sdp.A_mat @ x - sdp.b)
    e_sdp = sdp.c @ x
    e_psi = hamiltonian_expectation(psi, a, nmo, oei, tei, d, photon.get('omega', 0.0),
                                    photon.get('photon_coupling'), photon.get('photon_linear', 0.0))
    min_eigenvalue = min(np.linalg.eigvalsh(0.5 * (X + X.T)).min() for X in sdp.get_rdm_blocks(x))
    print('    nmo = %i, na = %i, nb = %i, n_photon = %s, %s: %i constraints, max |Ax-b| = %.2e, |c.x - <H>| = %.2e, '
          'min eigenvalue %.1e' % (nmo, nalpha, nbeta, n_photon, kwargs, len(sdp.b), residual.max(),
                                   abs(e_sdp - e_psi), min_eigenvalue))
    assert residual.max() < 1e-10, 'constraint violated by an N-representable state'
    assert abs(e_sdp - e_psi) < 1e-10, 'objective does not match <H>'
    assert min_eigenvalue > -1e-10, 'a block of an N-representable state is not positive semidefinite'


def test_exact_state_satisfies_constraints():
    check(4, 2, 1, q1=True, q2=True, g2=True, t1=True, t2=True, three_pos=True, constrain_spin=True)
    check(4, 2, 1, q2=True, g2=True, t2_prime=True, constrain_spin=True)
    check(5, 3, 2, q2=True, g2=True, t2_prime=True, constrain_spin=False)
    check(4, 2, 2, q1=True, q2=True, g2=True, three_pos=True, constrain_spin=False)
    check(4, 3, 1, q1=True, q2=True, g2=True, three_pos=True, constrain_spin=True)
    check(5, 3, 2, q1=True, q2=True, g2=True, three_pos=True, constrain_spin=True)
    check(5, 2, 2, three_pos=True, four_pos=True, constrain_spin=False)
    check(5, 3, 2, q1=True, q2=True, three_pos=True, four_pos=True, constrain_spin=True)
    check(5, 3, 2, q1=True, q2=True, g2=True, t2=True, constrain_spin=False, spin_resolved_traces=False)
    check(4, 2, 1, q1=True, q2=True, g2=True, augment_g2=True, t2=True, constrain_spin=True)


def test_photon_exact_state_satisfies_constraints():
    check(3, 2, 1, n_photon=1, q1=True, q2=True, g2=True, constrain_spin=True)
    check(3, 1, 1, n_photon=2, q1=True, q2=True, g2=True, constrain_spin=False)
    check(4, 2, 1, n_photon=1, q2=True, g2=True, t1=True, t2=True, constrain_spin=True)
    check(4, 2, 1, n_photon=2, q2=True, g2=True, t2=True, constrain_spin=True)
    check(4, 2, 1, n_photon=2, q2=True, g2=True, t2_prime=True, constrain_spin=True)
    check(4, 2, 2, n_photon=1, q2=True, g2=True, three_pos=True, constrain_spin=False)
    check(4, 3, 1, n_photon=1, q1=True, q2=True, g2=True, three_pos=True, constrain_spin=True)
    check(4, 3, 1, n_photon=1, q1=True, q2=True, g2=True, constrain_spin=False, spin_resolved_traces=False)
    check(3, 1, 1, n_photon=1, q2=True, g2=True, augment_g2=True, constrain_spin=True)
    check(4, 2, 1, n_photon=2, q2=True, g2=True, augment_g2=True, t2=True, constrain_spin=True)
    check(4, 2, 2, n_photon=1, q1=True, q2=True, g2=True, augment_g2=True, three_pos=True, constrain_spin=False,
          spin_resolved_traces=False)


def hamiltonian_matrix(a, nmo, oei, tei, d, omega, coupling, linear):
    """
    H (as in hamiltonian_expectation) as a sparse matrix in the Fock space times d photon states
    """
    dim = a[(0, 'a')].shape[0] // d
    b = sp.kron(sp.identity(dim), sp.diags(np.sqrt(np.arange(1, d)), 1, shape=(d, d)), format='csr')
    H = omega * (b.T @ b) + linear * (b + b.T)
    for s in 'ab':
        for p, q in itertools.product(range(nmo), repeat=2):
            E = a[(p, s)].T @ a[(q, s)]
            H = H + oei[p, q] * E + coupling[p, q] * (E @ (b + b.T))
    for s, t in itertools.product('ab', repeat=2):
        for p, q, r, u in itertools.product(range(nmo), repeat=4):
            if tei[p, q, r, u] != 0.0:
                H = H + 0.5 * tei[p, q, r, u] * (a[(p, s)].T @ a[(r, t)].T @ a[(u, t)] @ a[(q, s)])
    return H


def check_stationarity(nmo, nalpha, nbeta, n_photon, **kwargs):
    """
    the lowest eigenstate of H with S = (nalpha - nbeta) / 2 satisfies every constraint,
    including < [H, b] > = 0 (photon_stationarity), and a random state violates < [H, b] > = 0
    """
    rng = np.random.default_rng(1)
    oei = rng.normal(size=(nmo, nmo))
    oei = oei + oei.T
    tei = rng.normal(size=(nmo,) * 4)
    tei = tei + tei.transpose(1, 0, 2, 3)
    tei = tei + tei.transpose(0, 1, 3, 2)
    tei = tei + tei.transpose(2, 3, 0, 1)
    coupling = rng.normal(size=(nmo, nmo))
    coupling = coupling + coupling.T
    d, omega, linear = n_photon + 1, 0.7, -0.3

    sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, n_photon=n_photon, omega=omega, photon_coupling=coupling,
                    photon_linear=linear, photon_stationarity=True, **kwargs)
    a, P = fock_operators(nmo, d)

    # lowest eigenstate in the (nalpha, nbeta) sector with S = Sz
    dets = [det * d + A for det in range(2 ** (2 * nmo)) for A in range(d)
            if bin(det & (2 ** nmo - 1)).count('1') == nalpha and bin(det >> nmo).count('1') == nbeta]
    H = hamiltonian_matrix(a, nmo, oei, tei, d, omega, coupling, linear).toarray()[np.ix_(dets, dets)]
    s_plus = sum(a[(p, 'a')].T @ a[(p, 'b')] for p in range(nmo))
    Sz = 0.5 * (nalpha - nbeta)
    S2 = (s_plus.T @ s_plus).toarray()[np.ix_(dets, dets)] + (Sz * Sz + Sz) * np.eye(len(dets))
    w, v = np.linalg.eigh(H)
    k = next(k for k in range(len(w)) if abs(v[:, k] @ S2 @ v[:, k] - Sz * (Sz + 1)) < 1e-8)
    psi = np.zeros(2 ** (2 * nmo) * d)
    psi[dets] = v[:, k]

    x = primal_from_state(sdp, psi, a, P)
    residual = np.abs(sdp.A_mat @ x - sdp.b)
    random = np.abs(sdp.A_mat @ primal_from_state(sdp, random_state(nmo, nalpha, nbeta, a, d), a, P) - sdp.b)
    print('    nmo = %i, na = %i, nb = %i, n_photon = %i, %s: eigenstate max |Ax-b| = %.2e, |c.x - E| = %.2e, '
          'random state max |Ax-b| = %.2e' % (nmo, nalpha, nbeta, n_photon, kwargs, residual.max(),
                                              abs(sdp.c @ x - w[k]), random.max()))
    assert residual.max() < 1e-10, 'constraint violated by an eigenstate'
    assert abs(sdp.c @ x - w[k]) < 1e-10, 'objective does not match the eigenvalue'
    assert random.max() > 1e-6, 'the stationarity condition is satisfied by a random state'


def test_photon_stationarity():
    check_stationarity(3, 1, 1, 1, q2=True, g2=True, constrain_spin=True)
    check_stationarity(3, 2, 1, 2, q1=True, q2=True, g2=True, constrain_spin=True)
    check_stationarity(4, 2, 1, 1, q2=True, g2=True, t2=True, constrain_spin=True)


if __name__ == '__main__':
    test_exact_state_satisfies_constraints()
    print('test_exact_state_satisfies_constraints passed')
    test_photon_exact_state_satisfies_constraints()
    print('test_photon_exact_state_satisfies_constraints passed')
    test_photon_stationarity()
    print('test_photon_stationarity passed')
