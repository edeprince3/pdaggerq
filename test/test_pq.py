# -*- coding: utf-8 -*-
import pdaggerq
import subprocess
import pytest
import os
import re

# clear files in test_outputs if they exist
script_path = os.path.dirname(os.path.realpath(__file__))
os.system(f"rm {script_path}/test_outputs/*/*")

# Helper Functions
def read_file(file_path):
    with open(file_path) as file:
        return file.read()

def write_file(file_path, output, title=None):
    content = "\n".join([" ".join(line) for line in output])
    if title is not None:
        content = f"# {title}\n{content}"

    with open(file_path, "w") as file:
        file.write(content)

def process_output(output):
    processed = output.strip().split("\n")

    # use regex to find all items within ''
    pq_regex = re.compile(r"'(.*?)'")

    tmp = []
    for i, line in enumerate(processed):
        # skip lines that do not end with ] (i.e. lines that do not contain pq output)
        # note: this does not check the output of the einsum parser
        if not line.endswith("]"):
            continue

        # extract elements within ''
        elements = pq_regex.findall(line)

        tmp.append(elements)

    processed = tmp
    del tmp

    for line in processed:
        # Format floats to 6 decimal places (avoids mismatch due to rounding)
        for i, word in enumerate(line):
            try:
                line[i] = f"{float(word):.6f}"
            except ValueError:
                pass

        # sort the elements within each line alphabetically
        line.sort()

    # sort each line alphabetically
    processed.sort()

    return processed

def compare_outputs(test_name, script_path):
    # compute difference
    diff = subprocess.run(["diff", f"{script_path}/test_outputs/actual/{test_name}_result.out", f"{script_path}/test_outputs/expected/{test_name}_expected.out"], capture_output=True, text=True).stdout

    # ensure that the difference is empty
    if len(diff) > 0:
        with open(f"{script_path}/test_outputs/diff/{test_name}_diff.out", "w") as file:
            file.write(diff)

        print(f"Test {test_name} failed")
        assert len(diff) == 0

# Tests
ccsd_tests     = ("ccsd", "ccsd_d1", "ccsd_d2", "ccsd_doubles", "ccsd_energy", "ccsd", "ccsd_singles", "ccsd_t",
                  "eom_ccsd_sigma", "ea_eom_ccsd", "eom_ccsd_d1_by_hand", "eom_ccsd_d1", "eom_ccsd_hamiltonian","eom_ccsd", "ip_eom_ccsd", 
                  "lambda_singles", "lambda_doubles", "ccsd_with_spin")
ci_tests       = ("cid_d1", "cid_d2", "cisd_hamiltonian")
other_tests    = ("rdm_mappings", "extended_rpa")
ccsdt_tests    = ("ccsd_t", "cc3", "ccsdt", "ccsdt_with_spin", "ccsdt_with_spin", "ea_eom_ccsdt", "ip_eom_ccsdt", "ccsdt_with_spin")
code_gen_tests = ("ccsd_codegen", "lambda_singles_codegen", "lambda_doubles_codegen")

# Combine all tests
tests = ccsd_tests + ci_tests + other_tests + ccsdt_tests + code_gen_tests

@pytest.mark.parametrize("test_name", tests)
def test_script_output(test_name):
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Run the script
    print(f"Running test {test_name}")
    result = subprocess.run(["python", f"{script_path}/../examples/{test_name}.py"], capture_output=True, text=True)

    # Process outputs
    result_set   = process_output(result.stdout)
    expected_set = process_output(read_file(f"{script_path}/../examples/reference_outputs/{test_name}.ref"))

    # Write actual and expected output to files
    write_file(f"{script_path}/test_outputs/actual/{test_name}_result.out", result_set)
    write_file(f"{script_path}/test_outputs/expected/{test_name}_expected.out", expected_set)

    # Compare outputs
    compare_outputs(test_name, script_path)

if __name__ == "__main__":
    for test_name in tests:
        test_script_output(test_name)
