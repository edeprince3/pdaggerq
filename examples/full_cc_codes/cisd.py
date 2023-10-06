"""
A full working spin-orbital CISD code generated with pdaggerq

If you want to run the example here you should install pyscf openfermion and openfermion-pyscf
The actual CISD code (cisd_energy, cisd_singles_residual, cisd_doubles_residual, kernel)
do not depend on those packages but you gotta get integrals frome somehwere.

We also check the code by using pyscfs functionality for generating spin-orbital
t-amplitudes from RCCSD.  the main() function is fairly straightforward.
"""
import numpy as np
from numpy import einsum


def kernel(r1, r2, fock, g, o, v, eps, max_iter=100, stopping_eps=1.0E-8,
           diis_size=None, diis_start_cycle=4):

    # initialize diis if diis_size is not None
    # else normal scf iterate
    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        r1_dim = r1.size
        old_vec = np.hstack((r1.flatten(), r2.flatten()))

    n = np.newaxis
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)

    old_energy = cisd_energy(r1, r2, fock, g, o, v)

    for idx in range(max_iter):

        singles_res = cisd_singles_residual(r1, r2, fock, g, o, v) + fock_e_ai * r1
        doubles_res = cisd_doubles_residual(r1, r2, fock, g, o, v) + fock_e_abij * r2

        denom_abij = 1 / (old_energy -eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
        denom_ai = 1 / (old_energy -eps[v, n] + eps[n, o])

        new_singles = singles_res * denom_ai
        new_doubles = doubles_res * denom_abij

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:r1_dim].reshape(r1.shape)
            new_doubles = new_vectorized_iterate[r1_dim:].reshape(r2.shape)
            old_vec = new_vectorized_iterate

        current_energy = cisd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps :
            return new_singles, new_doubles
        else:
            r1 = new_singles
            r2 = new_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
    else:
        print("Did not converge")
        return new_singles, new_doubles


def main():
    from itertools import product
    import pyscf
    import openfermion as of
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    import numpy as np


    basis = 'cc-pvdz'
    mol = pyscf.M(
        atom='H 0 0 0; B 0 0 {}'.format(1.6),
        basis=basis)

    mf = mol.RHF().run()

    molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['B', (0, 0, 1.6)]],
                                basis=basis, charge=0, multiplicity=1)
    molecule = run_pyscf(molecule, run_ccsd=True)
    oei, tei = molecule.get_integrals()
    norbs = int(mf.mo_coeff.shape[1])
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

    # put in physics notation. OpenFermion stores <12|2'1'>
    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])

    g = gtei
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc

    r1f, r2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, eps,
                      diis_size=8, diis_start_cycle=4)

    print('    CISD correlation energy: %20.12f' % (cisd_energy(r1f, r2f, fock, g, o, v)))
    print('    * CISD total energy:     %20.12f' % (cisd_energy(r1f, r2f, fock, g, o, v) + hf_energy))


def cisd_energy(r1, r2, f, g, o, v):

    #    E r(a,i) = < 0 | i* a H (1 + r1 + r2) | 0> :
    
    #	  1.0000 f(i,i)
    #energy =  1.000000000000000 * einsum('ii', f[o, o])
    
    #	  1.0000 f(i,a)*r1(a,i)
    energy =  1.000000000000000 * einsum('ia,ai', f[o, v], r1)
    
    #	 -0.5000 <j,i||j,i>
    #energy += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    
    #	  0.2500 <j,i||a,b>*r2(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], r2)
    
    return energy


def cisd_singles_residual(r1, r2, f, g, o, v):

    #    E = < 0 | H (1 + r1 + r2) | 0> :
    
    #	  1.0000 f(a,i)
    singles_res =  1.000000000000000 * einsum('ai->ai', f[v, o])
    
    #	  1.0000 f(j,j)*r1(a,i)
    #singles_res +=  1.000000000000000 * einsum('jj,ai->ai', f[o, o], r1)
    
    #	 -1.0000 f(j,i)*r1(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj->ai', f[o, o], r1)
    
    #	  1.0000 f(a,b)*r1(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi->ai', f[v, v], r1)
    
    #	 -1.0000 f(j,b)*r2(b,a,i,j)
    singles_res += -1.000000000000000 * einsum('jb,baij->ai', f[o, v], r2)
    
    #	 -0.5000 <k,j||k,j>*r1(a,i)
    #singles_res += -0.500000000000000 * einsum('kjkj,ai->ai', g[o, o, o, o], r1)
    
    #	  1.0000 <j,a||b,i>*r1(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj->ai', g[o, v, v, o], r1)
    
    #	 -0.5000 <k,j||b,i>*r2(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj->ai', g[o, o, v, o], r2)
    
    #	 -0.5000 <j,a||b,c>*r2(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij->ai', g[o, v, v, v], r2)
    
    return singles_res


def cisd_doubles_residual(r1, r2, f, g, o, v):

    #    E r(a,b,i,j) = < 0 | i* a H (1 + r1 + r2) | 0> :
    
    #	 -1.0000 P(i,j)*P(a,b)f(a,j)*r1(b,i)
    contracted_intermediate = -1.000000000000000 * einsum('aj,bi->abij', f[v, o], r1)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 f(k,k)*r2(a,b,i,j)
    #doubles_res +=  1.000000000000000 * einsum('kk,abij->abij', f[o, o], r2)
    
    #	 -1.0000 P(i,j)f(k,j)*r2(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f[o, o], r2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f(a,c)*r2(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f[v, v], r2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 <l,k||l,k>*r2(a,b,i,j)
    #doubles_res += -0.500000000000000 * einsum('lklk,abij->abij', g[o, o, o, o], r2)
    
    #	  1.0000 <a,b||i,j>
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g[v, v, o, o])
    
    #	  1.0000 P(a,b)<k,a||i,j>*r1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kaij,bk->abij', g[o, v, o, o], r1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<a,b||c,j>*r1(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('abcj,ci->abij', g[v, v, v, o], r1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <l,k||i,j>*r2(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g[o, o, o, o], r2)
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>*r2(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g[o, v, v, o], r2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||c,d>*r2(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g[v, v, v, v], r2)
    
    return doubles_res

if __name__ == "__main__":
    main()


