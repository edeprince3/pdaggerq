# -*- coding: utf-8 -*-
from numpy.f2py.auxfuncs import throw_error

import pdaggerq
import subprocess
import pytest
import sys
import os
import re

# Tests
ccsd_tests     = ("ccsd", "ccsd_d1", "ccsd_d2", "ccsd_doubles", "ccsd_energy", "ccsd", "ccsd_singles", "ccsd_t",
                  "eom_ccsd_sigma", "ea_eom_ccsd", "eom_ccsd_d1_by_hand", "eom_ccsd_d1", "eom_ccsd_hamiltonian","eom_ccsd", "ip_eom_ccsd", 
                  "lambda_singles", "lambda_doubles", "ccsd_with_spin")
ci_tests       = ("cid_d1", "cid_d2", "cisd_hamiltonian")
other_tests    = ("rdm_mappings", "extended_rpa")
ccsdt_tests    = ("ccsd_t", "cc3", "ccsdt", "ccsdt_with_spin", "active_space_CCSDt", "ea_eom_ccsdt", "ip_eom_ccsdt", "ccsdt_with_spin")
qed_tests      = ("qed_ccsd_21", "qed_ccsd_22", "eom_qed_ccsd_21", "eom_qed_ccsd_21_1rdm", "eom_qed_ccsd_21_2rdm")

# Combine all tests
tests  = ("cisd", "ccsd", "ccsdt", "eom_ccsd")

@pytest.mark.parametrize("test_name", tests)
def test_script_output(test_name):
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Get paths to the codegen and code files
    codegen_path = f"{script_path}/../pq_graph/tests/{test_name}_codegen.py"
    code_path = f"{script_path}/../pq_graph/tests/{test_name}_code.py"

    # Run the codegen script
    print(f"Running test {test_name}")
    result = subprocess.run([str(sys.executable), codegen_path], capture_output=True, text=True)
    status = result.returncode
    if status != 0:
        raise AssertionError(f"Failure during execution:\n {result.stderr}")

    # now run the generated code
    result = subprocess.run([str(sys.executable), code_path], capture_output=True, text=True)
    status = result.returncode
    if status != 0:
        raise AssertionError(f"Failure during execution:\n {result.stderr}")

    # all good
    return

if __name__ == "__main__":
    for test_name in tests:
        test_script_output(test_name)
