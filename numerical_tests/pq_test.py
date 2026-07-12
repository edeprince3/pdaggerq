import numpy as np
import psi4    
import os
import subprocess
import sys
import importlib

from pdaggerq.numerical_utils.cc import cc
from pdaggerq.numerical_utils.eom_cc import eom_ccsd

# Setup paths
script_path = os.path.dirname(os.path.realpath(__file__))
gen_dir = os.path.join(script_path, "generated_equations")
print(gen_dir)
os.makedirs(gen_dir, exist_ok=True)

# Add the generation directory to sys.path so Python can import from it
if gen_dir not in sys.path:
    sys.path.insert(0, gen_dir)

def setup_psi4_test():

    # set up job for psi4
    mol = psi4.geometry("""
    0 1
    O
    H 1 1.0
    H 1 1.0 2 104.5
    #H            0.000000000000     0.000000000000     0.000000000000 
    #F            0.000000000000     0.000000000000     1.6
    symmetry c1
    """)

    # set options
    psi4.set_options({'basis': 'sto-3g',
                      'scf_type': 'pk',
                      'reference': 'uhf',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12})

    # compute the Hartree-Fock energy and wave function
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    return mol, wfn

def test_ccsd_codegen():

    # pq codegen 
    from pdaggerq.numerical_utils.autogen import cc_residual

    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}
    
    # Execute the code string in memory
    exec(cc_residual('cc_energy', ['t1', 't2'], [['1']], 'cc_energy'), globals(), local_namespace)
    exec(cc_residual('r1', ['t1', 't2'], [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(cc_residual('r2', ['t1', 't2'], [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)
    
    # Pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()
    mycc = cc(
        wfn, 
        mol, 
        nfzc=1, 
        cc_energy_func=local_namespace["cc_energy"],
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"]
    )

    en = mycc.t_solver()

    assert np.isclose(en, -75.019641774768, rtol=1e-8, atol=1e-8)

def test_uccsd_3_codegen():

    # pq codegen 
    from pdaggerq.numerical_utils.autogen import uccsd_energy
    from pdaggerq.numerical_utils.autogen import uccsd_singles_residual
    from pdaggerq.numerical_utils.autogen import uccsd_doubles_residual

    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}
    
    # Execute the code string in memory
    exec(uccsd_energy(3, 'cc_energy', 'cc_energy'), globals(), local_namespace)
    exec(uccsd_singles_residual(2, 'r1', [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(uccsd_doubles_residual(2, 'r2', [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)

    # pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()
    mycc = cc(
        wfn, 
        mol, 
        nfzc=1, 
        cc_energy_func=local_namespace["cc_energy"], 
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"]
    )

    en = mycc.t_solver()

    assert np.isclose(en, -75.020242934640, rtol=1e-8, atol=1e-8)

def test_uccsd_4_codegen():

    # pq codegen 
    from pdaggerq.numerical_utils.autogen import uccsd_energy
    from pdaggerq.numerical_utils.autogen import uccsd_singles_residual
    from pdaggerq.numerical_utils.autogen import uccsd_doubles_residual

    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}

    # Execute the code string in memory
    exec(uccsd_energy(4, 'cc_energy', 'cc_energy'), globals(), local_namespace)
    exec(uccsd_singles_residual(3, 'r1', [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(uccsd_doubles_residual(3, 'r2', [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)

    # pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()
    mycc = cc(
        wfn,
        mol,
        nfzc=1,
        cc_energy_func=local_namespace["cc_energy"], 
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"]
    )

    en = mycc.t_solver()

    assert np.isclose(en, -75.019695059108, rtol=1e-8, atol=1e-8)

def test_lambda_ccsd_codegen():
        
    # pq codegen 
    from pdaggerq.numerical_utils.autogen import cc_residual
    from pdaggerq.numerical_utils.autogen import lambda_cc_residual
    from pdaggerq.numerical_utils.autogen import lambda_cc_pseudoenergy
        
    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}
   
    # Execute the code string in memory
    exec(cc_residual('cc_energy', ['t1', 't2'], [['1']], 'cc_energy'), globals(), local_namespace)
    exec(cc_residual('r1', ['t1', 't2'], [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(cc_residual('r2', ['t1', 't2'], [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)

    exec(lambda_cc_pseudoenergy('cc_pseudoenergy', [['l1'], ['l2']], [['1']], 'cc_pseudoenergy'), globals(), local_namespace)
    exec(lambda_cc_residual('r1', ['t1', 't2'], [['l1'], ['l2']], 'e1(a,i)', 'l1_residual'), globals(), local_namespace)
    exec(lambda_cc_residual('r2', ['t1', 't2'], [['l1'], ['l2']], 'e2(a,b,j,i)', 'l2_residual'), globals(), local_namespace)

    # pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()

    mycc = cc(
        wfn, 
        mol,
        nfzc=1,
        cc_energy_func=local_namespace["cc_energy"],
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"],
        cc_pseudoenergy_func=local_namespace["cc_pseudoenergy"],
        l1_residual_func=local_namespace["l1_residual"],
        l2_residual_func=local_namespace["l2_residual"],
    )
    
    en = mycc.t_solver()

    assert np.isclose(en, -75.019641774768, rtol=1e-8, atol=1e-8)

    pseudoen = mycc.lambda_solver()

    assert np.isclose(pseudoen, -0.054046897553, rtol=1e-8, atol=1e-8)

# old workflow, writing functions to disk
#def test_ccsdt_codegen():
#
#    # pq codegen 
#    from pdaggerq.numerical_utils.autogen import cc_residual
#
#    cc_residual('cc_energy', ['t1', 't2', 't3'], [['1']], 'cc_energy')
#    cc_residual('r1', ['t1', 't2', 't3'], [['e1(i,a)']], 't1_residual')
#    cc_residual('r2', ['t1', 't2', 't3'], [['e2(i,j,b,a)']], 't2_residual')
#    cc_residual('r3', ['t1', 't2', 't3'], [['e3(i,j,k,c,b,a)']], 't3_residual')
#
#    # define the autogenerated module names we want to import
#    module_names = ["cc_energy", "t1_residual", "t2_residual", "t3_residual"]
#    local_namespace = {}
#
#    # import each pq-generated file
#    for name in module_names:
#        if name in sys.modules:
#            importlib.reload(sys.modules[name])
#        local_namespace[name] = importlib.import_module(name)
#
#    # pass pq-generated functions into the ccsd solver
#    mol, wfn = setup_psi4_test()
#    mycc = cc(
#        wfn,
#        mol,
#        nfzc=1,
#        cc_energy_func=local_namespace["cc_energy"].cc_energy,
#        t1_residual_func=local_namespace["t1_residual"].t1_residual,
#        t2_residual_func=local_namespace["t2_residual"].t2_residual,
#        t3_residual_func=local_namespace["t3_residual"].t3_residual
#    )
#
#    en = mycc.t_solver()
#
#    assert np.isclose(en, -75.019746392571, rtol=1e-8, atol=1e-8)

def test_ccsdt_codegen():

    # pq codegen 
    from pdaggerq.numerical_utils.autogen import cc_residual

    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}
    
    # Execute the code string in memory
    exec(cc_residual('cc_energy', ['t1', 't2', 't3'], [['1']], 'cc_energy'), globals(), local_namespace)
    exec(cc_residual('r1', ['t1', 't2', 't3'], [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(cc_residual('r2', ['t1', 't2', 't3'], [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)
    exec(cc_residual('r3', ['t1', 't2', 't3'], [['e3(i,j,k,c,b,a)']], 't3_residual'), globals(), local_namespace)
    
    # Pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()
    mycc = cc(
        wfn,
        mol, 
        nfzc=1,  
        cc_energy_func=local_namespace["cc_energy"],
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"],
        t3_residual_func=local_namespace["t3_residual"]
    )

    en = mycc.t_solver()

    assert np.isclose(en, -75.019746392571, rtol=1e-8, atol=1e-8)

def test_cc3_codegen():

    # pq codegen 
    from pdaggerq.numerical_utils.autogen import cc_residual
    from pdaggerq.numerical_utils.autogen import cc3_triples_residual

    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}

    # Execute the code string in memory
    exec(cc_residual('cc_energy', ['t1', 't2', 't3'], [['1']], 'cc_energy'), globals(), local_namespace)
    exec(cc_residual('r1', ['t1', 't2', 't3'], [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(cc_residual('r2', ['t1', 't2', 't3'], [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)
    exec(cc3_triples_residual('r3', [['e3(i,j,k,c,b,a)']], 't3_residual'), globals(), local_namespace)
    
    # Pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()
    mycc = cc(
        wfn,
        mol, 
        nfzc=1,  
        cc_energy_func=local_namespace["cc_energy"],
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"],
        t3_residual_func=local_namespace["t3_residual"]
    )

    en = mycc.t_solver()

    assert np.isclose(en, -75.019717612241, rtol=1e-8, atol=1e-8)

def test_eomccsd_codegen():

    # pq codegen 
    from pdaggerq.numerical_utils.autogen import cc_residual
    from pdaggerq.numerical_utils.autogen import eomcc_sigma

    # Create an empty dictionary to hold the pq-generated equations
    local_namespace = {}

    # Execute the code string in memory
    exec(cc_residual('cc_energy', ['t1', 't2'], [['1']], 'cc_energy'), globals(), local_namespace)
    exec(cc_residual('r1', ['t1', 't2'], [['e1(i,a)']], 't1_residual'), globals(), local_namespace)
    exec(cc_residual('r2', ['t1', 't2'], [['e2(i,j,b,a)']], 't2_residual'), globals(), local_namespace)

    exec(eomcc_sigma('sigma0', ['t1', 't2'], [['1']], [['r0'], ['r1'], ['r2']], 'right_sigma0'), globals(), local_namespace)
    exec(eomcc_sigma('sigma1', ['t1', 't2'], [['e1(i,a)']], [['r0'], ['r1'], ['r2']], 'right_sigma1'), globals(), local_namespace)
    exec(eomcc_sigma('sigma2', ['t1', 't2'], [['e2(i,j,b,a)']], [['r0'], ['r1'], ['r2']], 'right_sigma2'), globals(), local_namespace)

    exec(eomcc_sigma('sigma0', ['t1', 't2'], [['l0'], ['l1'], ['l2']], [['1']], 'left_sigma0'), globals(), local_namespace)
    exec(eomcc_sigma('sigma1', ['t1', 't2'], [['l0'], ['l1'], ['l2']], [['e1(a,i)']], 'left_sigma1'), globals(), local_namespace)
    exec(eomcc_sigma('sigma2', ['t1', 't2'], [['l0'], ['l1'], ['l2']], [['e2(a,b,j,i)']], 'left_sigma2'), globals(), local_namespace)

    # pass pq-generated functions into the ccsd solver
    mol, wfn = setup_psi4_test()
    mycc = cc(
        wfn, 
        mol, 
        nfzc=1, 
        cc_energy_func=local_namespace["cc_energy"],
        t1_residual_func=local_namespace["t1_residual"],
        t2_residual_func=local_namespace["t2_residual"]
    )

    en = mycc.t_solver()

    assert np.isclose(en, -75.019641774768, rtol=1e-8, atol=1e-8)

    eomcc = eom_ccsd(mycc, 
        right_sigma0_func=local_namespace["right_sigma0"],
        right_sigma1_func=local_namespace["right_sigma1"],
        right_sigma2_func=local_namespace["right_sigma2"],
        left_sigma0_func=local_namespace["left_sigma0"],
        left_sigma1_func=local_namespace["left_sigma1"],
        left_sigma2_func=local_namespace["left_sigma2"]
     )

    ref_energies = [0.000000000000,
        0.356359777102,
        0.412605450358,
        0.457799399030,
        0.457998414869,
    ]

    eomcc.right_solver()

    assert np.allclose(ref_energies, eomcc.eom_cc_energy, rtol=1e-8, atol=1e-8)

    eomcc.left_solver()

    assert np.allclose(ref_energies, eomcc.eom_cc_energy, rtol=1e-8, atol=1e-8)

def main():
    #test_ccsd_spin_orbital_codegen()
    #test_ccsd_codegen()
    #test_ccsdt_codegen()
    #test_cc3_codegen()
    #test_eomccsd_codegen()
    #test_lambda_ccsd_codegen()
    #test_uccsd_3_codegen()
    #test_uccsd_4_codegen()
    raise Exception("run with pytest")


if __name__ == "__main__":
    main()

