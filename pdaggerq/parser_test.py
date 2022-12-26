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

from pdaggerq.parser import contracted_strings_to_tensor_terms
from pdaggerq.algebra import BaseTerm, TensorTerm, Index, TwoBody, T2amps, T1amps


def test_parse_strings_to_tensor():
   energy_strings = [['+1.000000', 'f(i,i)'],
                     ['+1.000000', 'f(i,a)', 't1(a,i)'],
                     ['-0.500000', '<i,j||i,j>'],
                     ['-0.250000', '<i,j||a,b>', 't2(a,b,j,i)'],
                     ['+0.500000', '<i,j||a,b>', 't1(a,i)', 't1(b,j)']]

   energy_tensor_terms = contracted_strings_to_tensor_terms(energy_strings)
   i, j, a, b = Index('i', 'occ'), Index('j', 'occ'), Index('a', 'virt'), Index(
      'b', 'virt')
   h_ii = BaseTerm(indices=(i, i), name='f', spin='')
   f_ia = BaseTerm(indices=(i, a), name='f', spin='')
   t1 = T1amps(indices=(a, i))
   g_ijab = TwoBody(indices=(i, j, a, b), name='g')
   t2_abij = T2amps(indices=(a, b, j, i), name='t2')

   eterm1 = TensorTerm(base_terms=(h_ii,))
   eterm2 = TensorTerm(base_terms=(f_ia, t1,), coefficient=1)
   eterm3 = TensorTerm(base_terms=(g_ijab, t2_abij), coefficient=-0.25)
   assert eterm1.__repr__() == energy_tensor_terms[0].__repr__()
   assert eterm2.__repr__() == energy_tensor_terms[1].__repr__()
   assert eterm3.__repr__() == energy_tensor_terms[3].__repr__()