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

import re
from pdaggerq.algebra import (BaseTerm, Index, TensorTerm, TensorTermAction,
                              OneBody, TwoBody, FockMat, Delta, Dipole,
                              D1, D2, D3, D4, BosonDiagonal,
                              T0amps, T1amps, T2amps, T3amps, T4amps,
                              Left0amps, Left1amps, Left2amps, Left3amps, Left4amps,
                              Right0amps, Right1amps, Right2amps, Right3amps, Right4amps,
                              ContractionPermuter, ContractionPairPermuter2,
                              ContractionPairPermuter3, ContractionPairPermuter6)
from pdaggerq.config import OCC_INDICES, VIRT_INDICES


def string_to_baseterm(term_string, occ_idx=OCC_INDICES, virt_idx=VIRT_INDICES):

    # new operators should be added here
    tensor_map = {
        'h' : OneBody,
        'g' : TwoBody,
        'f' : FockMat,
        'd' : Delta,
        'd+' : Dipole,
        'd-' : Dipole,
        'w0' : BosonDiagonal,
        'd1' : D1,
        'd2' : D2,
        'd3' : D3,
        'd4' : D4,
        't0' : T0amps,
        't1' : T1amps,
        't2' : T2amps,
        't3' : T3amps,
        't4' : T4amps,
        'r0' : Right0amps,
        'r1' : Right1amps,
        'r2' : Right2amps,
        'r3' : Right3amps,
        'r4' : Right4amps,
        'l0' : Left0amps,
        'l1' : Left1amps,
        'l2' : Left2amps,
        'l3' : Left3amps,
        'l4' : Left4amps,
    }
    # tensor action has different constructor, so put new ones here
    tensor_action_map = {
        'p' : ContractionPermuter,
        'pp2' : ContractionPairPermuter2,
        'pp3' : ContractionPairPermuter3,
        'pp6' : ContractionPairPermuter6
    }

    # strip operator names, indices, # of boson, active-space label, and spin using regex

    # boson_pattern looks like '[...]_(n)p' where n is an integer
    boson_pattern = re.compile('_\d+p')

    # active_pattern looks like '[...]_[01]...[01]' without a p
    active_pattern = re.compile('_[01]+(?!p)')

    # spin_pattern for operator ranks looks like '[...]_[ab]...[ab]([...])'
    spin_pattern = re.compile('_[ab]+')

    # indices pattern looks like '[...]([a-zA-Z],...,[a-zA-Z])'
    # with ChatGPT's help:
    # (?<=\() : Positive lookbehind to match ( but not include it in the result.
    # [^)]+   : Match one or more characters that are not a closing parenthesis ).
    # (?=\))  : Positive lookahead to match ) but not include it in the result.
    idx_pattern  = re.compile('(?<=\()[^)]+(?=\))')

    # all operators will be of the form 'op_boson_spin_active([idx])' except ERIs
    # first, extract and strip the active space labels
    term_active = active_pattern.findall(term_string)
    if len(term_active) == 0:
        active = ''
    else:
        active = term_active[0]
        # remove '_active' to obtain 'op_boson_spin([idx])'
        term_string = term_string.replace(active, '')

    # next, extract and strip the spin
    term_spin = spin_pattern.findall(term_string)
    if len(term_spin)==0:
        spin = ''
    else:
        spin = term_spin[0]
        # remove '_spin' to obtain 'op_boson([idx])'
        term_string = term_string.replace(spin,'')

    # next, extract and strip the boson number
    term_boson = boson_pattern.findall(term_string)
    if len(term_boson) == 0:
        boson = ''
    else:
        boson = term_boson[0]
        # remove '_boson' to obtain 'op([idx])'
        term_string = term_string.replace(boson, '')


    if "||" in term_string:
        # special case for ERI, which is printed as <x,x||x,x>
        # strip '<' and '>', replace '||' by ',' to obtain 'x,x,x,x'
        index_string = term_string.replace('<', '').replace('>', '').replace('||', ',')
        tmp = index_string.split('_')
        idx = [Index(xx, 'occ') if xx in occ_idx
               else Index(xx, 'virt') for xx in tmp[0].split(',')]
        return TwoBody(indices=tuple(idx), spin=spin, active=active, boson=boson)

    else:
        # next, extract indices
        idx = idx_pattern.findall(term_string)
        if len(idx) == 0:
            idx = []
        else:
            idx = idx[0]
            # remove '([idx])' to obtain 'op'
            term_string = term_string.replace(f'({idx})','')
            idx = [Index(xx, 'occ') if xx in occ_idx
                   else Index(xx, 'virt') for xx in idx.split(',')]

        # check if operator is allowed
        # make operator label lowercase from this point on
        term_string = term_string.lower()
        if term_string in tensor_map.keys():
            return tensor_map[term_string](indices=tuple(idx), spin=spin, active=active, boson=boson)
        elif term_string in tensor_action_map.keys():
            return tensor_action_map[term_string](indices=tuple(idx), spin=spin)
        else:
            raise TypeError(f"Operator {term_string} not recognized")


def contracted_strings_to_tensor_terms(pdaggerq_list_of_strings):
    """
    Take the output from pdaggerq.fully_contracted_strings() or
    pdaggerq.fully_contracted_strings_with_spin() and generate
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
            filter(lambda xx: True if 'a(' in xx or 'a*(' in xx else False, pq_strings))
        dagger_locations = [1 if '*' in xx else 0 for xx in rdm_strings]
        zero_location = dagger_locations.index(0)
        if not all(dagger_locations[:zero_location]):
            raise ValueError("Not in vacuum normal order")
        if any(dagger_locations[zero_location:]):
            raise ValueError("Not in vacuum normal order")
        rdm_idx = [xx.replace('*', '') if '*' in xx else xx for xx in
                   rdm_strings]
        rdm_idx = [xx.replace('a(', '') if 'a(' in xx else xx for xx in
                   rdm_idx]
        rdm_idx = [xx.replace(')', '') if ')' in xx else xx for xx in
                   rdm_idx]
        g_idx = [Index(xx, 'all') for xx in rdm_idx]
        rdm_baseterm = BaseTerm(indices=tuple(g_idx), spin='', active='', boson='',
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
