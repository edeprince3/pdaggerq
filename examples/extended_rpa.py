"""
Evaluate the double commutator associated with extended RPA equations
"""
from typing import List
import pdaggerq


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
    ahat.print()
    ahat.clear()


if __name__ == "__main__":
    main()