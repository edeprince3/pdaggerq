"""
End-to-end tests for pdaggerq.numerical.codegen.auto_v2rdm: generate v2RDM SDPs for doublet H5
(STO-3G, ROHF orbitals from pyscf), solve them with the boundary-point solver in
libsdp.bpsdp_jax, and compare the electronic energy with reference values.
"""
import types

import numpy as np
import pytest

pytest.importorskip('pyscf')
pytest.importorskip('libsdp.bpsdp_jax')

from pyscf import gto, scf, ao2mo

from pdaggerq.numerical.codegen.auto_v2rdm import v2rdm_sdp
from libsdp.bpsdp_jax import bpsdp

H5_CHAIN = [['H', (0.0, 0.0, 0.0)], ['H', (0.0, 0.0, 1.0)], ['H', (0.0, 0.0, 2.0)],
            ['H', (0.0, 0.0, 3.0)], ['H', (0.0, 0.0, 4.0)]]

H5_CLUSTER = [['H', (0.0, 0.0, 0.0)], ['H', (0.0, 0.0, 1.0)], ['H', (0.0, 1.0, 0.0)],
              ['H', (0.0, 1.0, 1.0)], ['H', (1.0, 1.0, 1.0)]]

# tolerance on the electronic energy (E_h), set by the convergence of the SDP solver
TOLERANCE = 1e-6


def integrals(geometry):
    """
    one- and two-electron integrals (chemists' notation) for doublet H5 in the ROHF orbital basis

    :param geometry: list of [atom, (x, y, z)] in Angstrom
    :return nalpha, nbeta, nmo, oei, tei:
    """
    mol = gto.M(atom=geometry, basis='sto-3g', unit='Angstrom', spin=1, verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    C = mf.mo_coeff
    nmo = C.shape[1]
    oei = C.T @ mf.get_hcore() @ C
    tei = ao2mo.restore(1, ao2mo.kernel(mol, C), nmo)
    nalpha, nbeta = mol.nelec
    return nalpha, nbeta, nmo, oei, tei


def solve(sdp):
    """
    solve the SDP with the boundary-point algorithm and return the electronic energy
    """
    options = types.SimpleNamespace(
        sdp_error_convergence = 2e-5,
        sdp_objective_convergence = 2e-5,
        cg_convergence = 1e-12,
        dynamic_cg_convergence = True,
        cg_maxiter = 10000,
        mu_update_frequency = 500,
    )
    x = np.zeros(sdp.n_primal)
    x, y, z = bpsdp(x, sdp.b, sdp.c, sdp.block_dimensions, sdp.A_mat, options, 50000)

    # x is symmetric, so the folded c gives the same c.x as the symmetric one
    return sdp.c @ x


def test_dqg():
    nalpha, nbeta, nmo, oei, tei = integrals(H5_CHAIN)
    sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q1=True, q2=True, g2=True, constrain_spin=True,
                    fold_constraint_matrices=True)
    primal = solve(sdp)
    assert np.isclose(primal, -6.053046519244, rtol=0.0, atol=TOLERANCE)


#def test_t2():
#    nalpha, nbeta, nmo, oei, tei = integrals(H5_CHAIN)
#    sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q1=True, q2=True, g2=True, t2=True, constrain_spin=True,
#                    fold_constraint_matrices=True)
#    assert np.isclose(solve(sdp), -6.050112697969, rtol=0.0, atol=TOLERANCE)
#
#
#def test_3pos():
#    nalpha, nbeta, nmo, oei, tei = integrals(H5_CLUSTER)
#    sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q1=True, q2=True, g2=True, three_pos=True,
#                    constrain_spin=True, fold_constraint_matrices=True)
#    assert np.isclose(solve(sdp), -6.860438860712, rtol=0.0, atol=TOLERANCE)

if __name__ == "__main__":
    test_dqg()
