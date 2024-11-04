# -*- coding: utf-8 -*-
from numpy.f2py.auxfuncs import throw_error

import pdaggerq
import subprocess
import pytest
import sys
import os
import re

# add numerical tests
tests  = (
    "cisd",
    "ccsd",
    "ccsd_with_spin",
    "lambda_ccsd",
    "eom_ccsd",
    "ccsdt",
    "cc3",
    "ccsdt_with_spin"
)

# get the path to the script
script_path = os.path.dirname(os.path.realpath(__file__))

# remove log files
os.system(f"rm numerical_test.log")

# remove the generated files
os.system(f"rm {script_path}/../pq_graph/tests/*_code.py")

@pytest.mark.parametrize("test_name", tests)
def test_script_output(test_name):

    # Get paths to the codegen and code files
    codegen_path = f"{script_path}/../pq_graph/tests/{test_name}_codegen.py"
    code_path = f"{script_path}/../pq_graph/tests/{test_name}_code.py"

    # Run the codegen script
    print(f"Running test {test_name}")
    result = subprocess.run([str(sys.executable), codegen_path], capture_output=True, text=True)
    status = result.returncode
    if status != 0:
        with open("numerical_test.log", "a") as file:
            file.write(f"Test {test_name} codegen\n")
            file.write(result.stderr)
        raise AssertionError(f"Failure during execution:\n {result.stderr}")

    # append stdout to log file
    with open("numerical_test.log", "a") as file:
        file.write(f"Test {test_name} codegen failed!!\n")
        file.write(result.stdout)

    # now run the generated code
    result = subprocess.run([str(sys.executable), code_path], capture_output=True, text=True)
    status = result.returncode
    if status != 0:
        with open("numerical_test.log", "a") as file:
            file.write(f"Test {test_name} code failed!!\n")
            file.write(result.stderr)
        raise AssertionError(f"Failure during execution:\n {result.stderr}")

    # append stdout to log file
    with open("numerical_test.log", "a") as file:
        file.write(f"Test {test_name} code\n")
        file.write(result.stdout)

    # all good
    return

if __name__ == "__main__":
    print("Please use pytest to run the tests")
    print("Syntax: python -m pytest numerical_test.py")
    sys.exit(1)
