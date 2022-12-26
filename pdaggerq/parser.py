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
        tmp = index_string.split('_')
        spin = ''
        if len(tmp) > 1 :
            spin = '_' + tmp[1]
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx in tmp[0].split(',')]
        return TwoBody(indices=tuple(g_idx), spin=spin)
    if "g(" in term_string:
        index_string = term_string.replace('g(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return TwoBody(indices=tuple(g_idx))
    elif 'h(' in term_string:
        index_string = term_string.replace('h(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return OneBody(indices=tuple(g_idx))
    elif 'f(' in term_string:
        index_string = term_string.replace('f(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return FockMat(spin='', indices=tuple(g_idx))
    elif 'f_aa(' in term_string:
        index_string = term_string.replace('f_aa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return FockMat(indices=tuple(g_idx), spin='_aa')
    elif 'f_bb(' in term_string:
        index_string = term_string.replace('f_bb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return FockMat(indices=tuple(g_idx), spin='_bb')
    elif 't4(' in term_string:
        index_string = term_string.replace('t4(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx))
    elif 't4_aaaaaaaa(' in term_string:
        index_string = term_string.replace('t4_aaaaaaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx), spin='_aaaaaaaa')
    elif 't4_aaabaaab(' in term_string:
        index_string = term_string.replace('t4_aaabaaab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx), spin='_aaabaaab')
    elif 't4_aabbaabb(' in term_string:
        index_string = term_string.replace('t4_aabbaabb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx), spin='_aabbaabb')
    elif 't4_abbbabbb(' in term_string:
        index_string = term_string.replace('t4_abbbabbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx), spin='_abbbabbb')
    elif 't4_bbbbbbbb(' in term_string:
        index_string = term_string.replace('t4_bbbbbbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T4amps(indices=tuple(g_idx), spin='_bbbbbbbb')
    elif 't3(' in term_string:
        index_string = term_string.replace('t3(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T3amps(indices=tuple(g_idx))
    elif 't3_aaaaaa(' in term_string:
        index_string = term_string.replace('t3_aaaaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T3amps(indices=tuple(g_idx), spin='_aaaaaa')
    elif 't3_aabaab(' in term_string:
        index_string = term_string.replace('t3_aabaab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T3amps(indices=tuple(g_idx), spin='_aabaab')
    elif 't3_abbabb(' in term_string:
        index_string = term_string.replace('t3_abbabb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T3amps(indices=tuple(g_idx), spin='_abbabb')
    elif 't3_bbbbbb(' in term_string:
        index_string = term_string.replace('t3_bbbbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T3amps(indices=tuple(g_idx), spin='_bbbbbb')
    elif 't2(' in term_string:
        index_string = term_string.replace('t2(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T2amps(indices=tuple(g_idx))
    elif 't2_aaaa(' in term_string:
        index_string = term_string.replace('t2_aaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T2amps(indices=tuple(g_idx), spin='_aaaa')
    elif 't2_abab(' in term_string:
        index_string = term_string.replace('t2_abab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T2amps(indices=tuple(g_idx), spin='_abab')
    elif 't2_bbbb(' in term_string:
        index_string = term_string.replace('t2_bbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T2amps(indices=tuple(g_idx), spin='_bbbb')
    elif 't1(' in term_string:
        index_string = term_string.replace('t1(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T1amps(indices=tuple(g_idx))
    elif 't1_aa(' in term_string:
        index_string = term_string.replace('t1_aa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T1amps(indices=tuple(g_idx), spin='_aa')
    elif 't1_bb(' in term_string:
        index_string = term_string.replace('t1_bb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return T1amps(indices=tuple(g_idx), spin='_bb')
    elif 'd(' in term_string:
        index_string = term_string.replace('d(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Delta(indices=tuple(g_idx))
    elif 'l0' in term_string:
        return Left0amps()
    elif 'r0' in term_string:
        return Right0amps()
    elif 'l1(' in term_string:
        index_string = term_string.replace('l1(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left1amps(indices=tuple(g_idx))
    elif 'l1_aa(' in term_string:
        index_string = term_string.replace('l1_aa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left1amps(indices=tuple(g_idx), spin='_aa')
    elif 'l1_bb(' in term_string:
        index_string = term_string.replace('l1_bb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left1amps(indices=tuple(g_idx), spin='_bb')
    elif 'l2(' in term_string:
        index_string = term_string.replace('l2(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left2amps(indices=tuple(g_idx))
    elif 'l2_aaaa(' in term_string:
        index_string = term_string.replace('l2_aaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left2amps(indices=tuple(g_idx), spin='_aaaa')
    elif 'l2_abab(' in term_string:
        index_string = term_string.replace('l2_abab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left2amps(indices=tuple(g_idx), spin='_abab')
    elif 'l2_bbbb(' in term_string:
        index_string = term_string.replace('l2_bbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left2amps(indices=tuple(g_idx), spin='_bbbb')
    elif 'l3' in term_string:
        index_string = term_string.replace('l3(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left3amps(indices=tuple(g_idx))
    elif 'l4(' in term_string:
        index_string = term_string.replace('l4(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx))
    elif 'l4_aaaaaaaa(' in term_string:
        index_string = term_string.replace('l4_aaaaaaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx), spin='_aaaaaaaa')
    elif 'l4_aaabaaab(' in term_string:
        index_string = term_string.replace('l4_aaabaaab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx), spin='_aaabaaab')
    elif 'l4_aabbaabb(' in term_string:
        index_string = term_string.replace('l4_aabbaabb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx), spin='_aabbaabb')
    elif 'l4_abbbabbb(' in term_string:
        index_string = term_string.replace('l4_abbbabbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx), spin='_abbbabbb')
    elif 'l4_bbbbbbbb(' in term_string:
        index_string = term_string.replace('l4_bbbbbbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Left4amps(indices=tuple(g_idx), spin='_bbbbbbbb')
    elif 'r1(' in term_string:
        index_string = term_string.replace('r1(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right1amps(indices=tuple(g_idx))
    elif 'r1_aa(' in term_string:
        index_string = term_string.replace('r1_aa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right1amps(indices=tuple(g_idx), spin='_aa')
    elif 'r1_bb(' in term_string:
        index_string = term_string.replace('r1_bb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right1amps(indices=tuple(g_idx), spin='_bb')
    elif 'r2(' in term_string:
        index_string = term_string.replace('r2(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right2amps(indices=tuple(g_idx))
    elif 'r2_aaaa(' in term_string:
        index_string = term_string.replace('r2_aaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right2amps(indices=tuple(g_idx), spin='_aaaa')
    elif 'r2_abab(' in term_string:
        index_string = term_string.replace('r2_abab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right2amps(indices=tuple(g_idx), spin='_abab')
    elif 'r2_bbbb(' in term_string:
        index_string = term_string.replace('r2_bbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right2amps(indices=tuple(g_idx), spin='_bbbb')
    elif 'r3(' in term_string:
        index_string = term_string.replace('r3(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right3amps(indices=tuple(g_idx))
    elif 'r3_aaaaaa(' in term_string:
        index_string = term_string.replace('r3_aaaaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right3amps(indices=tuple(g_idx), spin='_aaaaaa')
    elif 'r3_aabaab(' in term_string:
        index_string = term_string.replace('r3_aabaab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right3amps(indices=tuple(g_idx), spin='_aabaab')
    elif 'r3_abbabb(' in term_string:
        index_string = term_string.replace('r3_abbabb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right3amps(indices=tuple(g_idx), spin='_abbabb')
    elif 'r3_bbbbbb(' in term_string:
        index_string = term_string.replace('r3_bbbbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right3amps(indices=tuple(g_idx), spin='_bbbbbb')
    elif 'r4(' in term_string:
        index_string = term_string.replace('r4(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx))
    elif 't4_aaaaaaaa(' in term_string:
        index_string = term_string.replace('t4_aaaaaaaa(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx), spin='_aaaaaaaa')
    elif 't4_aaabaaab(' in term_string:
        index_string = term_string.replace('t4_aaabaaab(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx), spin='_aaabaaab')
    elif 'r4_aabbaabb(' in term_string:
        index_string = term_string.replace('r4_aabbaabb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx), spin='_aabbaabb')
    elif 'r4_abbbabbb(' in term_string:
        index_string = term_string.replace('r4_abbbabbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx), spin='_abbbabbb')
    elif 'r4_bbbbbbbb(' in term_string:
        index_string = term_string.replace('r4_bbbbbbbb(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return Right4amps(indices=tuple(g_idx), spin='_bbbbbbbb')
    elif 'P(' in term_string:
        index_string = term_string.replace('P(', '').replace(')', '')
        g_idx = [Index(xx, 'occ') if xx in occ_idx else Index(xx, 'virt') for xx
                 in index_string.split(',')]
        return ContractionPermuter(indices=tuple(g_idx))
    else:
        raise TypeError("{} not recognized".format(term_string))


def contracted_strings_to_tensor_terms(pdaggerq_list_of_strings):
    """
    Take the output from pdaggerq.fully_contracted_strings() or
    pdaggerq.fully_contracted_strings_spin() and generate
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
        rdm_baseterm = BaseTerm(indices=tuple(g_idx), spin='',
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
