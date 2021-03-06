#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from pdaggerq.algebra import (OneBody, TwoBody, T1amps, T2amps, T3amps, T4amps,
                              Index, TensorTerm, D1, Delta, Left0amps, Left1amps,
                              Left2amps, Left3amps, Left4amps, Right0amps,
                              Right1amps, Right2amps, Right3amps, Right4amps,
                              FockMat, BaseTerm, ContractionPermuter,
                              TensorTermAction, )
from pdaggerq.config import OCC_INDICES, VIRT_INDICES


def string_to_baseterm(term_string, occ_idx=OCC_INDICES, virt_idx=VIRT_INDICES):
    if "||" in term_string:
        index_string = term_string.replace('<', '').replace('>', '').replace(
            '||', ',')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return TwoBody(indices=tuple(g_idx))
    if "g(" in term_string:
        index_string = term_string.replace('g(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return TwoBody(indices=tuple(g_idx))
    elif 'h' in term_string:
        index_string = term_string.replace('h(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return OneBody(indices=tuple(g_idx))
    elif 'f(' in term_string:
        index_string = term_string.replace('f(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return FockMat(indices=tuple(g_idx))
    elif 't4' in term_string:
        index_string = term_string.replace('t4(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx))
    elif 't3' in term_string:
        index_string = term_string.replace('t3(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T3amps(indices=tuple(g_idx))
    elif 't2' in term_string:
        index_string = term_string.replace('t2(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T2amps(indices=tuple(g_idx))
    elif 't1' in term_string:
        index_string = term_string.replace('t1(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T1amps(indices=tuple(g_idx))
    elif 'd(' in term_string:
        index_string = term_string.replace('d(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Delta(indices=tuple(g_idx))
    elif 'l0' in term_string:
        return Left0amps()
    elif 'r0' in term_string:
        return Right0amps()
    elif 'l1' in term_string:
        index_string = term_string.replace('l1(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left1amps(indices=tuple(g_idx))
    elif 'l2' in term_string:
        index_string = term_string.replace('l2(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left2amps(indices=tuple(g_idx))
    elif 'l3' in term_string:
        index_string = term_string.replace('l3(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left3amps(indices=tuple(g_idx))
    elif 'l4' in term_string:
        index_string = term_string.replace('l4(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx))
    elif 'r1' in term_string:
        index_string = term_string.replace('r1(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right1amps(indices=tuple(g_idx))
    elif 'r2' in term_string:
        index_string = term_string.replace('r2(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right2amps(indices=tuple(g_idx))
    elif 'r3' in term_string:
        index_string = term_string.replace('r3(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right3amps(indices=tuple(g_idx))
    elif 'r4' in term_string:
        index_string = term_string.replace('r4(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx))
    elif 'P(' in term_string:
        index_string = term_string.replace('P(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return ContractionPermuter(indices=tuple(g_idx))
    else:
        raise TypeError("{} not recognized".format(term_string))


def contracted_strings_to_tensor_terms(pdaggerq_list_of_strings):
    """
    Take the output from pdaggerq.fully_contracted_strings() and generate
    TensorTerms

    :param pdaggerq_list_of_strings: List[List[str]] where the first item is
                                     always a float.
    :return: List of algebra.TensorTerms
    """
    tensor_terms = []
    for pq_string in pdaggerq_list_of_strings:
        coeff = float(pq_string[0])
        single_tensor_term = []
        actions = []
        for ts in pq_string[1:]:
            bs = string_to_baseterm(ts)
            if isinstance(bs, TensorTermAction):
                actions.append(bs)
            else:
                single_tensor_term.append(bs)
        tensor_terms.append(
            TensorTerm(base_terms=tuple(single_tensor_term), coefficient=coeff,
                       permutation_ops=actions)
        )
    return tensor_terms


def vacuum_normal_ordered_strings_to_tensor_terms(pdaggerq_list_of_strings):
    """
    Take the output of a normal ordering in pdaggerq and produce tensor terms

    Thus function enforces parsing normal ordered terms.  The name of the term
    parsed is equal to d{} half the length of the number of indices. d1 'i*','j'
    d2 'i*','j*','k','l', etc

    :param pdaggerq_list_of_strings: List[List[str]] where  first item is always
                                     a float.
    :return: List of algebra.TensorTerms
    """
    tensor_terms = []
    for pq_strings in pdaggerq_list_of_strings:
        coeff = float(pq_strings[0])
        delta_strings = list(
            filter(lambda xx: True if 'd(' in xx else False, pq_strings))
        delta_terms = []
        for d_str in delta_strings:
            delta_idx = d_str.replace('d(', '').replace(')', '').split(',')
            delta_idx = [Index(xx, 'all') for xx in delta_idx]
            delta_terms.append(Delta(indices=tuple(delta_idx)))
        rdm_strings = list(
            filter(lambda xx: False if 'd(' in xx else True, pq_strings[1:]))
        dagger_locations = [1 if '*' in xx else 0 for xx in rdm_strings]
        zero_location = dagger_locations.index(0)
        if not all(dagger_locations[:zero_location]):
            raise ValueError("Not in vacuum normal order")
        if any(dagger_locations[zero_location:]):
            raise ValueError("Not in vacuum normal order")
        rdm_idx = [xx.replace('*', '') if '*' in xx else xx for xx in
                   rdm_strings]
        g_idx = [Index(xx, 'all') for xx in rdm_idx]
        rdm_baseterm = BaseTerm(indices=tuple(g_idx),
                                name="d{}".format(len(g_idx) // 2))
        tensor_terms.append(
            TensorTerm(base_terms=tuple(delta_terms + [rdm_baseterm]),
                       coefficient=coeff))
    return tensor_terms


if __name__ == "__main__":
    # ahat_strings = [['-1.000000', 'j*', 'r*', 'k', 'q', 'd(i,s)', 'd(l,p)'],
    #                 ['+1.000000', 'j*', 'r*', 'l', 'q', 'd(i,s)', 'd(k,p)'],
    #                 ['+1.000000', 'i*', 'r*', 'k', 'q', 'd(j,s)', 'd(l,p)'],
    #                 ['-1.000000', 'i*', 'r*', 'l', 'q', 'd(j,s)', 'd(k,p)'],
    #                 ['+1.000000', 'j*', 'r*', 'k', 'l', 'd(p,s)', 'd(i,q)'],
    #                 ['-1.000000', 'i*', 'r*', 'k', 'l', 'd(p,s)', 'd(j,q)'],
    #                 ['-1.000000', 'p*', 'r*', 'k', 'l', 'd(i,q)', 'd(j,s)'],
    #                 ['+1.000000', 'p*', 'r*', 'k', 'l', 'd(i,s)', 'd(j,q)'],
    #                 ['-1.000000', 'i*', 'j*', 'k', 's', 'd(l,p)', 'd(q,r)'],
    #                 ['+1.000000', 'i*', 'j*', 'q', 's', 'd(l,p)', 'd(k,r)'],
    #                 ['+1.000000', 'i*', 'j*', 'l', 's', 'd(k,p)', 'd(q,r)'],
    #                 ['-1.000000', 'i*', 'j*', 'q', 's', 'd(k,p)', 'd(l,r)'],
    #                 ['-1.000000', 'j*', 'p*', 'k', 's', 'd(i,q)', 'd(l,r)'],
    #                 ['+1.000000', 'j*', 'p*', 'l', 's', 'd(i,q)', 'd(k,r)'],
    #                 ['+1.000000', 'i*', 'p*', 'k', 's', 'd(j,q)', 'd(l,r)'],
    #                 ['-1.000000', 'i*', 'p*', 'l', 's', 'd(j,q)', 'd(k,r)']]

    # vacuum_normal_ordered_strings_to_tensor_terms(ahat_strings)

    string_list = [['+0.250000', '<l,k||c,d>', 't4(c,d,a,b,i,j,l,k)']]
    contracted_strings_to_tensor_terms(string_list)
