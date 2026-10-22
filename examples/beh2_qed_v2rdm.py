"""
QED-v2RDM for the BeH2 insertion path (C2v): RHF orbitals and integrals from pyscf -> Pauli-Fierz
integrals in the photon-number basis -> v2RDM SDPs from pdaggerq-dev's auto_v2rdm -> solve with
pdaggerq-dev's BPSDP solver. Optionally, an exact QED-FCI reference (matrix-free, pyscf).

Geometry (Angstrom): Be at the origin, H at (x, +/-(1.344 - 0.46 x), 0), the Purvis-Bartlett
insertion path (x = 0: linear H-Be-H; x ~ 1.5: barrier; x = 2.117 = 4 bohr: end of the model path).
--separated puts H2 (r_HH = 0.74 A) 20 A from Be instead, for the Be + H2 asymptote.

Hamiltonian (photon-number basis, single cavity mode, lambda in a.u., omega in E_h):

    H = H_e + omega b^dagger b - sqrt(omega/2) lambda.mu (b + b^dagger) + 1/2 (lambda.mu)^2

mu is the total (electronic + nuclear) dipole operator, so H is unique for a neutral molecule and
does not depend on a reference determinant (no coherent-state shift). Writing lambda.mu = d + c,
with d_pq = -lambda.<p|r|q> and c = lambda.mu_nuc, the dipole self-energy is a two-electron term
d_pq d_rs, a one-electron term from quadrupole integrals (+1/2 lambda_i lambda_j <p|r_i r_j|q>, the
same convention as pdaggerq's QED-CC solver and hilbert), c d_pq, and the constant c^2 / 2.

Requirements: jax, pyscf, libsdp, pdaggerq-dev on the auto-v2rdm branch (pdaggerq.numerical.codegen.auto_v2rdm)

Examples:

    # barrier region, no cavity, DQG and T2
    python beh2_qed_v2rdm.py --x 1.5 --conditions dqg t2 --lambda 0 0 0

    # same, in a cavity polarized along the insertion direction (x), with two photons
    python beh2_qed_v2rdm.py --x 1.5 --conditions dqg t2 --lambda 0.05 0 0 --n-photon 2

    # the Be + H2 asymptote
    python beh2_qed_v2rdm.py --separated --conditions dqg --lambda 0.05 0 0 --n-photon 2

    # 3POS with the augmented G2 block ([[G, d], [d^T, D0]]), omega = 2 eV, tight convergence
    python beh2_qed_v2rdm.py --x 1.5 --conditions 3pos --augment-g2 --lambda 0.05 0 0 --omega 0.07349864 \
        --convergence 1e-5 --gap 5e-5 --mu-update-frequency 5000

    # add the stationarity condition <[H, b]> = 0
    python beh2_qed_v2rdm.py --x 1.5 --conditions 3pos --photon-stationarity --lambda 0.05 0 0 --omega 0.07349864 \
        --convergence 1e-5 --gap 5e-5 --mu-update-frequency 5000

    # save the solution (here as beh2_x1.5_3pos.npz) for later analysis
    python beh2_qed_v2rdm.py --x 1.5 --conditions 3pos --lambda 0.05 0 0 --omega 0.07349864 \
        --convergence 1e-5 --gap 5e-5 --mu-update-frequency 5000 --save beh2_x1.5
"""
import argparse
import types

import numpy as np

from pyscf import gto, scf, ao2mo

DEBYE = 2.541746473

CONDITIONS = {
    'dqg':  dict(q1=True, q2=True, g2=True),
    't2':   dict(q1=True, q2=True, g2=True, t2=True),
    't2p':  dict(q1=True, q2=True, g2=True, t2_prime=True),
    '3pos': dict(q1=True, q2=True, g2=True, three_pos=True),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument('--x', type=float, help='position along the insertion path (Angstrom)')
    where.add_argument('--separated', action='store_true', help='Be + H2 (r_HH = 0.74 A) 20 A apart')
    p.add_argument('--basis', default='sto-3g')
    p.add_argument('--lambda', dest='lam', type=float, nargs=3, default=[0.05, 0.0, 0.0],
                   help='coupling vector (a.u.); 0 0 0 for no cavity')
    p.add_argument('--omega', type=float, default=0.0813, help='cavity frequency (E_h); default 2.214 eV')
    p.add_argument('--n-photon', type=int, default=1, help='maximum photon number')
    p.add_argument('--conditions', nargs='+', default=['dqg'], choices=sorted(CONDITIONS))
    p.add_argument('--no-spin', action='store_true', help='do not impose spin constraints')
    p.add_argument('--augment-g2', action='store_true',
                   help='add the identity to the rows of G2 (with a cavity, couples G, the dressed 1-RDM, and D0)')
    p.add_argument('--photon-stationarity', action='store_true',
                   help='enforce the eigenstate condition <[H, b]> = 0 (needs a cavity)')
    p.add_argument('--convergence', type=float, default=5e-5, help='primal/dual error threshold')
    p.add_argument('--gap', type=float, default=None, help='primal-dual gap threshold (default: --convergence)')
    p.add_argument('--mu-update-frequency', type=int, default=5000, help='BPSDP penalty-parameter update frequency')
    p.add_argument('--maxiter', type=int, default=200000)
    p.add_argument('--matvec', default='scatter', choices=['scatter', 'bcoo'])
    p.add_argument('--save', default=None, metavar='PREFIX',
                   help='save each solution as PREFIX_<condition>.npz (primal vector, energy, and the settings '
                        'needed to regenerate the SDP)')
    return p.parse_args()


def molecule(args):
    """pyscf molecule on the insertion path (or the separated limit), Be at the origin"""
    if args.separated:
        atoms = [['Be', (0.0, 0.0, 0.0)], ['H', (20.0, 0.37, 0.0)], ['H', (20.0, -0.37, 0.0)]]
    else:
        y = 1.344 - 0.46 * args.x
        atoms = [['Be', (0.0, 0.0, 0.0)], ['H', (args.x, y, 0.0)], ['H', (args.x, -y, 0.0)]]
    return gto.M(atom=atoms, basis=args.basis, unit='Angstrom', verbose=0)


def pauli_fierz(mol, lam, omega):
    """
    photon-number-basis Pauli-Fierz integrals in the RHF orbital basis

    :return dict with oei, tei (chemists' notation), photon_coupling, photon_linear, constant
            (nuclear repulsion + dipole self-energy constant), electronic dipole matrices (x, y,
            z; a.u., including the electron charge), nalpha, nbeta, nmo, and the RHF energy
    """
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 500
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF did not converge')

    C = mf.mo_coeff
    nmo = C.shape[1]
    na, nb = mol.nelec
    mol.set_common_orig((0.0, 0.0, 0.0))

    oei = C.T @ mf.get_hcore() @ C
    tei = ao2mo.restore(1, ao2mo.kernel(mol, C), nmo)
    # electronic dipole integrals, -<p|r|q>
    dipole = [-C.T @ m @ C for m in mol.intor('int1e_r')]
    out = dict(nmo=nmo, nalpha=na, nbeta=nb, dipole=dipole, constant=mol.energy_nuc(),
               photon_coupling=None, photon_linear=0.0, e_rhf=mf.e_tot)

    if np.any(lam):
        # lambda-weighted electronic dipole, d_pq, and the constant c = lambda . mu_nuc
        d = sum(l * m for l, m in zip(lam, dipole))
        c = float(lam @ np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords()))

        # dipole self-energy, 1/2 (d + c)^2: two-electron d_pq d_rs; one-electron from quadrupole
        # integrals, +1/2 lambda_i lambda_j <p|r_i r_j|q>; c d_pq; constant c^2 / 2
        tei = tei + np.einsum('pq,rs->pqrs', d, d)
        rr = mol.intor('int1e_rr').reshape(3, 3, *oei.shape)
        q = C.T @ np.einsum('i,j,ijuv->uv', lam, lam, rr) @ C
        oei = oei + 0.5 * q + c * d

        out['photon_coupling'] = -np.sqrt(omega / 2.0) * d
        out['photon_linear'] = -np.sqrt(omega / 2.0) * c
        out['constant'] += 0.5 * c * c

    out['oei'], out['tei'] = oei, tei
    return out

def solve_v2rdm(h, args, condition):
    """generate and solve one QED-v2RDM SDP; returns the total energy, electronic 1-RDM, and <b+b>"""
    from pdaggerq.numerical.codegen.auto_v2rdm import v2rdm_sdp
    from libsdp.bpsdp_jax import bpsdp

    cavity = h['photon_coupling'] is not None
    photon = dict(n_photon=args.n_photon, omega=args.omega, photon_coupling=h['photon_coupling'],
                  photon_linear=h['photon_linear'], photon_stationarity=args.photon_stationarity) if cavity else {}
    sdp = v2rdm_sdp(h['nalpha'], h['nbeta'], h['nmo'], h['oei'], h['tei'], constrain_spin=not args.no_spin,
                    augment_g2=args.augment_g2, fold_constraint_matrices=True, **photon, **CONDITIONS[condition])
    gap = args.convergence if args.gap is None else args.gap
    options = types.SimpleNamespace(sdp_error_convergence=args.convergence, sdp_objective_convergence=gap,
                                    cg_convergence=1e-12, dynamic_cg_convergence=True, cg_maxiter=10000,
                                    mu_update_frequency=args.mu_update_frequency)
    x, y, z = bpsdp(np.zeros(sdp.n_primal), sdp.b, sdp.c, sdp.block_dimensions, sdp.A_mat, options,
                    args.maxiter, matvec=args.matvec)

    # electronic 1-RDM: photon partial trace of the dressed d1a/d1b blocks; <b+b> from the photon 1-RDM
    blocks = dict(zip(sdp.blocks, sdp.get_rdm_blocks(x)))
    n, dph = h['nmo'], sdp.d
    gamma = sum(blocks[b].reshape(n, dph, n, dph).trace(axis1=1, axis2=3) for b in ('d1a', 'd1b'))
    n_ph = float(np.arange(dph) @ np.diag(blocks['d0'])) if cavity else 0.0
    energy = sdp.c @ x + h['constant']
    if args.save:
        np.savez_compressed('%s_%s.npz' % (args.save, condition), x=np.asarray(x), energy=energy,
                            condition=condition, n_primal=sdp.n_primal, n_dual=sdp.n_dual,
                            args=repr(sorted(vars(args).items())))
    return energy, gamma, n_ph


def dipole_moment(h, mol, gamma):
    """total dipole moment (Debye) from the electronic 1-RDM"""
    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return (nuc + np.array([np.sum(m * gamma) for m in h['dipole']])) * DEBYE


def main():
    args = parse_args()
    lam = np.array(args.lam)
    mol = molecule(args)
    h = pauli_fierz(mol, lam, args.omega)
    print('    RHF energy: %20.12f' % h['e_rhf'])

    results = []

    for condition in args.conditions:
        e, gamma, n_ph = solve_v2rdm(h, args, condition)
        results.append((condition.upper(), e, dipole_moment(h, mol, gamma), n_ph))

    print()
    where = 'Be + H2 (20 A)' if args.separated else 'x = %.4f A' % args.x
    print('    BeH2 %s   basis: %s   lambda: %s   omega: %.6f   n_photon: %d'
          % (where, args.basis, ' '.join('%g' % l for l in lam), args.omega, args.n_photon))
    print('    %-10s %20s %14s %12s %12s' % ('method', 'energy (E_h)', 'error (mE_h)', 'mu_x (D)', '<b+b>'))


if __name__ == '__main__':
    main()
