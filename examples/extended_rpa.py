"""
Evaluate the double commutator associated with extended RPA equations
"""
from typing import List
import pdaggerq

from pdaggerq.parser import vacuum_normal_ordered_strings_to_tensor_terms
from pdaggerq.algebra import TensorTerm, Delta, Index, BaseTerm


def pq_commutator(ahat, left_op_string: List[str], right_op_string: List[str],
                  scaling_coeff=1.0):
    """
    Helper function to compute commutator

    Compute coeff * [left_op, right_op] commutator"""
    ahat.set_string(left_op_string + right_op_string)
    ahat.set_factor(1 * scaling_coeff)
    ahat.add_new_string()

    ahat.set_string(right_op_string + left_op_string)
    ahat.set_factor(-1 * scaling_coeff)
    ahat.add_new_string()
    return ahat


def pq_double_commutator(ahat, left_op_string: List[str],
                         center_op_string: List[str],
                         right_op_string: List[str],
                         scaling_coeff=1.0):
    """
    Helper function to compute double commutator

    [lO, [cO, rO]] =  lO cO rO - lO rO cO - cO rO lO + rO cO lO
    """
    ahat.set_string(left_op_string + center_op_string + right_op_string)
    ahat.set_factor(1. * scaling_coeff)
    ahat.add_new_string()

    ahat.set_string(left_op_string + right_op_string + center_op_string)
    ahat.set_factor(-1. * scaling_coeff)
    ahat.add_new_string()

    ahat.set_string(center_op_string + right_op_string + left_op_string)
    ahat.set_factor(-1. * scaling_coeff)
    ahat.add_new_string()

    ahat.set_string(right_op_string + center_op_string + left_op_string)
    ahat.set_factor(1. * scaling_coeff)
    ahat.add_new_string()

    return ahat


def erpa_terms_to_einsum(tensor_terms: List[TensorTerm],
                         constant_terms=['r', 's', 'p', 'q'],
                         contract_d2_with='k2'):
    """
    Generate terms einsum contractions for the ERPA matrix

    This is the simplest contraction generation and should only be used to
    check if other simplified codes are correct.  The deltafunctions are not
    removed and thus the user must specify an identity matrix called kd.  To

    get the erpa matrix reshape the 4-tensor with labels (pqrs) into a matrix
    with row indices rs and column indices pq.
    """
    k2_idx = [Index('i', 'all'), Index('j', 'all'), Index('k', 'all'), Index('l', 'all')]
    for tt in tensor_terms:
        # add the hamiltonian to contract with
        tt.base_terms += (BaseTerm(indices=tuple(k2_idx), name=contract_d2_with),)

        print("# ", tt)
        print(tt.einsum_string(update_val='erpa_val',
                               occupied=['i', 'j', 'k', 'l', 'r', 's', 'p', 'q'],
                               virtual=[],
                               output_variables=constant_terms,
                               optimize=False))
        print()


def main():
    # need cumulant decomposition on 3-RDM terms
    # to simplify to a 2-RDM + 1-RDM expression
    ahat = pdaggerq.pq_helper('true')

    # [r^s, [H, p^ q]]
    # (r^s [H, p^ q] - [H, p^q] r^s)
    # [H, p^ q] = H p^ q - p^ q H
    # [r^s, [H, p^ q]]  = r^ s H p^ q - r^ s p^ q H - H p^ q r^ s + p^ q H r^ s)

    # check another way
    # (r^s [H, p^ q] + [p^q, H] r^s)
    # r^ s  H p^ q - r^ s p^ q H +  p^ q H r^ s - H p^ q r^s

    # Hamiltonian indices will be i,j,k,l
    ahat.set_string(['r*', 's', 'i*','j*','k','l', 'p*', 'q'])
    ahat.add_new_string()

    ahat.set_string(['r*', 's', 'p*', 'q','i*','j*','k','l'])
    ahat.set_factor(-1.)
    ahat.add_new_string()

    ahat.set_string(['p*', 'q',  'i*','j*','k', 'l', 'r*', 's'])
    ahat.add_new_string()

    ahat.set_string(['i*','j*','k','l', 'p*', 'q', 'r*', 's'])
    ahat.set_factor(-1.)
    ahat.add_new_string()


    ahat.simplify()
    ahat.print()

    ahat.clear()


    ahat = pdaggerq.pq_helper('true')

    ahat = pq_double_commutator(ahat, ['r*', 's'], ['i*','j*','k','l'],  ['p*', 'q'])
    # print(ahat.strings())


    rpa_tensor_terms = vacuum_normal_ordered_strings_to_tensor_terms(ahat.strings())

    ahat.clear()
    print(rpa_tensor_terms)

    erpa_terms_to_einsum(tensor_terms=rpa_tensor_terms)



if __name__ == "__main__":
    main()