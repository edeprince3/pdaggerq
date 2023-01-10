"""
Evaluate the double commutator associated with extended RPA equations
"""
from typing import List
import pdaggerq

from pdaggerq.parser import vacuum_normal_ordered_strings_to_tensor_terms
from pdaggerq.algebra import TensorTerm, Delta, Index, BaseTerm


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
        tt.base_terms += (BaseTerm(indices=tuple(k2_idx), name=contract_d2_with, spin=''),)

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
    pq = pdaggerq.pq_helper('true')

    # [r^s, [H, p^ q]] = - [[H,p^ q],r^s]
    pq.add_double_commutator(-1.0,['e2(i,j,k,l)'],['e1(p,q)'],['e1(r,s)'])
    pq.simplify()
    pq.print(string_type = 'all')
    rpa_tensor_terms = vacuum_normal_ordered_strings_to_tensor_terms(pq.strings())
    pq.clear()
    print(rpa_tensor_terms)

    erpa_terms_to_einsum(tensor_terms=rpa_tensor_terms)


if __name__ == "__main__":
    main()
