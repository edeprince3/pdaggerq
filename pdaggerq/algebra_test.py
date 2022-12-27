#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
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

from pdaggerq.algebra import (BaseTerm, Index, TensorTerm, T1amps, T2amps,
                              TwoBody, OneBody)
import numpy as np


def test_index():
    i_idx = Index('i', 'occ')
    assert i_idx.support == 'occ'
    assert i_idx.name == 'i'

    i2_idx = Index('i', 'occ')
    assert i2_idx == i_idx

    j_idx = Index('j', 'occ')
    assert i_idx != j_idx

    iv_idx = Index('i', 'virt')
    assert i_idx != j_idx

    assert i_idx != iv_idx


def test_baseterm():
    term = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='h', spin='')
    assert term.indices[0] == Index('i', 'occ')
    assert term.indices[1] == Index('j', 'occ')

    term2 = BaseTerm(indices=(Index('k', 'occ'), Index('l', 'occ')),
                    name='t1', spin='')
    tensort = term2 * term
    assert isinstance(tensort, TensorTerm)

    tensort = term * term2
    assert isinstance(tensort, TensorTerm)

    assert term2 == term2
    assert term2 != term


def test_tensorterm():
    hij = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='h', spin='')
    t1ij = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='t', spin='')
    tensor_term = TensorTerm(base_terms=(hij, t1ij))
    assert tensor_term.coefficient == 1.0

    assert tensor_term.__repr__() == " 1.0000 h(i,j)*t(i,j)"

    hij = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='h', spin='_aa')
    t1ij = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='t', spin='_aa')
    tensor_term = TensorTerm(base_terms=(hij, t1ij))
    assert tensor_term.coefficient == 1.0

    assert tensor_term.__repr__() == " 1.0000 h_aa(i,j)*t_aa(i,j)"


def test_tensor_multiply():
    hij = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='h', spin='')
    t1ij = BaseTerm(indices=(Index('i', 'occ'), Index('j', 'occ')),
                    name='t', spin='')
    tensor_term = TensorTerm(base_terms=(hij, t1ij))
    test_tensor_term = tensor_term * 4
    assert isinstance(test_tensor_term, TensorTerm)
    assert np.isclose(test_tensor_term.coefficient, 4)

    test_tensor_term = 4 * tensor_term
    assert isinstance(test_tensor_term, TensorTerm)
    assert np.isclose(test_tensor_term.coefficient, 4)

    t1kl = BaseTerm(indices=(Index('k', 'occ'), Index('l', 'virt')),
                    name='t', spin='')
    test_tensor_term = tensor_term * t1kl
    assert test_tensor_term.base_terms[2] == t1kl

    test_tensor_term = t1kl * tensor_term
    assert test_tensor_term.base_terms[2] == t1kl


def test_preset_tensor_terms():
    i, j, a, b = Index('i', 'occ'), Index('j', 'occ'), Index('a', 'virt'), \
                 Index('b', 'virt'),
    test_term = T1amps(indices=(i, a))
    assert test_term.name == 't1'
    test_term = T2amps(indices=(i, j, b, a))
    assert test_term.name == 't2'
    test_term = TwoBody(indices=(i, j, b, a))
    assert test_term.name == 'g'
    test_term = OneBody(indices=(i, j))
    assert test_term.name == 'h'
