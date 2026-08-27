import subprocess
import sys

tests = [
    "ccsd", "ccsd_d1", "ccsd_d2", "ccsd_doubles", "ccsd_energy", "ccsd_singles",
    "ccsd_t", "eom_ccsd_sigma", "ea_eom_ccsd", "eom_ccsd_d1_by_hand", "eom_ccsd_d1",
    "eom_ccsd_hamiltonian", "eom_ccsd", "ip_eom_ccsd", "lambda_singles", "lambda_doubles",
    "ccsd_with_spin", "ucc3", "ucc4", "cid_d1", "cid_d2", "cisd_hamiltonian",
    "rdm_mappings", "extended_rpa", "ccsd_t", "cc3", "ccsdt", "ccsdt_with_spin",
    "active_space_CCSDt", "ea_eom_ccsdt", "ip_eom_ccsdt", "qed_ccsd_21", "qed_ccsd_22",
    "eom_qed_ccsd_21", "eom_qed_ccsd_21_1rdm", "eom_qed_ccsd_21_2rdm"
]

# Deduplicate tests while preserving list order
unique_tests = list(dict.fromkeys(tests))

for test in unique_tests:
    py_file = f"{test}.py"
    ref_file = f"{test}.ref"
    
    print(f"Executing {py_file} -> {ref_file}...")
    
    with open(ref_file, "w") as out:
        result = subprocess.run(
            [sys.executable, py_file],
            stdout=out,
            stderr=subprocess.STDOUT,  # Redirects stderr to the same .ref file
            text=True
        )
        
    if result.returncode != 0:
        print(f"  [ERROR] {py_file} failed with exit code {result.returncode}")
