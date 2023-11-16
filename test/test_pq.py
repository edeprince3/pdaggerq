# -*- coding: utf-8 -*-
import pdaggerq
import subprocess
import pytest
import os

# Helper Functions
def read_file(file_path):
    with open(file_path) as file:
        return file.read()

def write_file(file_path, content, title=None):
    if title is not None:
        content = f"# {title}\n{content}"
    with open(file_path, "w") as file:
        file.write(content)

def process_output(output):
    processed = [line.split() for line in output.strip().split("\n") if isinstance(line, list)]
    for line in processed:
        # Format floats
        for i, word in enumerate(line):
            print(word, flush=True)
            try:
                line[i] = f"{float(word):.6f}"
            except ValueError:
                pass

        # sort the line
        line.sort()

    # sort the output
    processed.sort()

    return processed

def compare_outputs(result_set, expected_set, test_name, script_path):
    # compute difference
    diff = subprocess.run(["diff", f"{script_path}/test_outputs/actual/{test_name}_result.out", f"{script_path}/test_outputs/expected/{test_name}_expected.out"], capture_output=True, text=True).stdout

    # ensure that the difference is empty
    if len(diff) > 0:
        diff = diff.split("\n")
        diff = [line for line in diff if line.startswith("<") or line.startswith(">")]
        diff = "\n".join(diff)
        write_file(f"{script_path}/test_outputs/difference/{test_name}_diff.out", diff, title="Difference")

        print(f"Test {test_name} failed")
        assert len(diff) == 0

# Tests
tests=("ccsd", "eom_ccsd_sigma", "cc3", "rdm_mappings", "extended_rpa", "ccsd_with_spin", "cc3", 
       "ccsd_codegen", "ccsd_d1", "ccsd_d2", "ccsd_doubles", "ccsd_energy", "ccsd", "ccsd_singles", 
       "ccsd_t", "ccsdt", "cid_d1", "cid_d2", "cisd_hamiltonian", "ea_eom_ccsd", "ea_eom_ccsdt", 
       "eom_ccsd_d1_by_hand", "eom_ccsd_d1", "eom_ccsd_hamiltonian","eom_ccsd", "eom_ccsd_sigma", 
       "ip_eom_ccsd", "ip_eom_ccsdt", "lambda_doubles_codegen", "lambda_doubles", "lambda_singles_codegen", 
       "lambda_singles", "ccsdt_with_spin")

@pytest.mark.parametrize("test_name", tests)
def test_script_output(test_name):
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Run the script
    print(f"Running test {test_name}")
    result = subprocess.run(["python", f"{script_path}/../examples/{test_name}.py"], capture_output=True, text=True)

    # Process outputs
    result_set = process_output(result.stdout)
    expected_set = process_output(read_file(f"{script_path}/../examples/reference_outputs/{test_name}.ref"))

    # Write actual and expected output to files
    write_file(f"{script_path}/test_outputs/actual/{test_name}_result.out", result.stdout)
    write_file(f"{script_path}/test_outputs/expected/{test_name}_expected.out", read_file(f"{script_path}/reference_outputs/{test_name}.ref"))

    # Compare outputs
    compare_outputs(result_set, expected_set, test_name, script_path)

if __name__ == "__main__":
    for test_name in tests:
        test_script_output(test_name)
