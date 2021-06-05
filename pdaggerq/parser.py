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

from pdaggerq.algebra import (OneBody, TwoBody, T1amps, T2amps, Index,
                              TensorTerm, D1,
                              Delta, Left0amps, Left1amps, Left2amps,
                              Right0amps,
                              Right1amps, Right2amps, FockMat)
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
    elif 'd' in term_string:
        index_string = term_string.replace('d(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return D1(indices=tuple(g_idx))
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
        for ts in pq_string[1:]:
            bs = string_to_baseterm(ts)
            single_tensor_term.append(bs)
        tensor_terms.append(
            TensorTerm(base_terms=tuple(single_tensor_term), coefficient=coeff)
        )
    return tensor_terms