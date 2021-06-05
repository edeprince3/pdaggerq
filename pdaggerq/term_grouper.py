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
"""
Group residual equation terms into those linearly dependent on diagonal fock
terms and those that are not.
"""
from collections import Counter
from pdaggerq.algebra import TensorTerm
from pdaggerq.config import OCC_INDICES


def check_if_fock_contributor(tensor_term: TensorTerm, cluster_amp_name: str,
                              one_body_op_name='h', two_body_op_name='g',
                              occupied_set=OCC_INDICES) -> bool:
    """
    MERGH

    This function is so lame...

    :param tensor_term:
    :param cluster_amp_name:
    :param occupied_set:
    :return:
    """
    if len(tensor_term.base_terms) != 2:
        return False
    else:
        if tensor_term.base_terms[0].name == cluster_amp_name and \
                tensor_term.base_terms[1].name != cluster_amp_name:
            if tensor_term.base_terms[1].name == one_body_op_name:
                return True
            elif tensor_term.base_terms[1].name == two_body_op_name:
                indices = Counter(
                    [xx.name for xx in tensor_term.base_terms[1].indices])
                if len(indices.keys()) == 3:
                    for key, val in indices.items():
                        if key in occupied_set and val == 2:
                            return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        elif tensor_term.base_terms[1].name == cluster_amp_name and \
                tensor_term.base_terms[0].name != cluster_amp_name:
            if tensor_term.base_terms[0].name == one_body_op_name:
                return True
            elif tensor_term.base_terms[0].name == two_body_op_name:
                indices = Counter(
                    [xx.name for xx in tensor_term.base_terms[0].indices])
                if len(indices.keys()) == 3:
                    for key, val in indices.items():
                        if key in occupied_set and val == 2:
                            return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        elif tensor_term.base_terms[1].name == cluster_amp_name and \
                tensor_term.base_terms[0] == cluster_amp_name:
            return False  # quadratic term
        else:
            return False


def remove_diagonal_fock(*, term_list, one_body_op_name='h', two_body_op_name='g',
                         linear_in_op_type='t2'):

    fock_contrib_terms = []
    residual_terms = []
    for tensor_term in term_list:
        if check_if_fock_contributor(tensor_term, linear_in_op_type,
                                     one_body_op_name=one_body_op_name,
                                     two_body_op_name=two_body_op_name):
            fock_contrib_terms.append(tensor_term)
        else:
            residual_terms.append(tensor_term)
    return fock_contrib_terms, residual_terms