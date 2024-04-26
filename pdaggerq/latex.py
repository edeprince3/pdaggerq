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

from fractions import Fraction

import re

occ_idx = ['i', 'j', 'k', 'l', 'm', 'n', 'I', 'J', 'K', 'L', 'M', 'N']
def is_occ(idx):
    if idx in occ_idx:
        return True
    return False

def latex(pq, terms, input_string = '', kill_deltas = False, terms_per_line = 2):
    """
    generate latex equations from pq-generated terms

    :param pq: a pq_helper object
    :param terms: the terms for which we want latex
    :param input_string: the input latex string to which the current terms should be added
    :param kill_deltas: do kill delta functions involving occupied/virtual pairs?
    :param terms_per_line: how many terms before a line break?

    :return output_string: the output latex string
    """

    amplitude_types = []
    for i in range (1, 5):
        amplitude_types.append('t' + str(i) + '(')
        amplitude_types.append('r' + str(i) + '(')
        amplitude_types.append('s' + str(i) + '(')
        amplitude_types.append('l' + str(i) + '(')
        amplitude_types.append('m' + str(i) + '(')

    output_string = ''
    if len(terms) == 0 :
        output_string += '0'

    line_break = 0
    for count, term in enumerate(terms): 

        this_string = ''

        # sign        
        factor = float(term[0])
        if abs(factor) < 1.0:
            fraction = Fraction(abs(factor))
            sign = '+'
            if factor < 0.0:
                sign = '-'
            this_string += sign
            this_string += '\\frac{%i}{%i}' % (fraction.numerator, fraction.denominator)
        else :
            sign = '+'
            if factor < 0.0:
                sign = '-'
            if int(abs(factor)) == 1:
                this_string += sign 
            else:
                this_string += sign + '%i' % (int(abs(factor)))

        for i in range (1, len(term)): 

            # delta function
            if 'd(' in term[i]:

                # kill d(ov)
                if kill_deltas:
                    if is_occ(term[i][2]):
                        if not is_occ(term[i][4]):
                            this_string = ''
                            break
                    else:
                        if is_occ(term[i][4]):
                            this_string = ''
                            break

                tmp = term[i][2] + term[i][4]
                this_string += '\\delta_{%s}' % (tmp) 
            # permutation operator
            elif 'P(' in term[i]:
                this_string += term[i]
            # creator
            elif 'a(' in term[i]:
                this_string += '\\hat{a}_{' + term[i][2] + '}'
            # annihilator
            elif 'a*(' in term[i]:
                this_string += '\\hat{a}^\dagger_{' + term[i][3] + '}'
            # one-electron operator
            elif 'h(' in term[i]:
                this_string += term[i][0] + '^{' + term[i][2] + '}_{' + term[i][4] + '}'
            # two-electron operator
            elif 'g(' in term[i]:
                this_string += term[i][0] + '^{' + term[i][2] + term[i][4] + '}_{' + term[i][6] + term[i][8] + '}'
            # fock
            elif 'f(' in term[i]:
                this_string += term[i][0] + '^{' + term[i][2] + '}_{' + term[i][4] + '}'
            # eri
            elif '<' in term[i]:
                this_string += '\\langle ' + term[i][1] + term[i][3] + '||' + term[i][6] + term[i][8] + '\\rangle '
            # amplitudes
            elif term[i][0:3] in amplitude_types:
                #excitation order
                nbra = int(term[i][1])
                nket = int(term[i][1])

                # adjust number of bra/ket terms for IP, EA, etc.
                if term[i][0] == 'r' or term[i][0] == 's':
                    op_type = pq.get_right_operators_type()
                    if op_type == "IP":
                        nbra -= 1
                    elif op_type == "DIP":
                        nbra -= 2
                    elif op_type == "EA":
                        nket -= 1
                    elif op_type == "DEA":
                        nket -= 2
                elif term[i][0] == 'l' or term[i][0] == 'm':
                    op_type = pq.get_right_operators_type()
                    if op_type == "IP":
                        nket -= 1
                    elif op_type == "DIP":
                        nket -= 2
                    elif op_type == "EA":
                        nbra -= 1
                    elif op_type == "DEA":
                        nbra -= 2

                this_string += term[i][0] + '^{' 
                count = 3
                for bra_id in range (0, nbra):
                    this_string += term[i][count]
                    count += 2
                this_string += '}_{' 
                for ket_id in range (0, nket):
                    this_string += term[i][count]
                    count += 2
                this_string += '}'
            # D1
            elif 'D1(' in term[i]:
                this_string += '~{}^1' + term[i][0] + '^{' + term[i][3] + '}_{' + term[i][5] + '}'
            # D2
            elif 'D2(' in term[i]:
                this_string += '~{}^2' + term[i][0] + '^{' + term[i][3] + term[i][5] + '}_{' + term[i][7] + term[i][9] + '}'
            # t3/r3/l3
            elif 'D3(' in term[i]:
                this_string += '~{}^3' + term[i][0] + '^{' + term[i][3] + term[i][5] + term[i][7] + '}_{' + term[i][9] + term[i][11] + term[i][13] + '}'
            # t4/r4/l4
            elif 'D4(' in term[i]:
                this_string += '~{}^4' + term[i][0] + '^{' + term[i][3] + term[i][5] + term[i][7] + term[i][9] + '}_{' + term[i][11] + term[i][13] + term[i][15] + term[i][17] + '}'
            # creator / annihilator
            else: 
                tmp = '\\hat{a}'
                if '*' in term[i]:
                    tmp += '^\\dagger'
                this_string += tmp + '_{' + term[i][0] + '}'

        output_string += this_string

        if this_string != '':
            line_break += 1

        if count < len(terms) - 1 and line_break == terms_per_line:
            line_break = 0
            output_string += ' \\nonumber \\\\ &'
                  
    #output_string += '\\\\'
    return input_string + output_string 
