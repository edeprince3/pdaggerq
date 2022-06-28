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
"""
Map spatial orbitals to spin orbitals.
"""

import numpy as np

def spatial_to_spin_orbital_oei(h, n, no):
    """

    :param h: one-electron orbitals
    :param n: number of spatial orbitals
    :param no: number of (doubly) occupied orbitals
    :return:  spin-orbital one-electron integrals, sh
    """

    # build spin-orbital oeis
    sh = np.zeros((2*n,2*n))

    # index 1
    for i in range(0,n):
        ia = i
        ib = i
    
        # alpha occ do nothing
        if ( ia < no ):
            ia = i
        # alpha vir shift up by no
        else :
            ia += no
        # beta occ
        if ( ib < no ):
            ib += no
        else :
            ib += n
    
        # index 2
        for j in range(0,n):
            ja = j
            jb = j
    
            # alpha occ
            if ( ja < no ):
                ja = j
            # alpha vir
            else :
                ja += no
            # beta occ
            if ( jb < no ):
                jb += no
            # beta vir
            else :
                jb += n

            # Haa
            sh[ia,ja] = h[i,j]
            # Hbb
            sh[ib,jb] = h[i,j]
    
    return sh

def spatial_to_spin_orbital_tei(g, n, no):
    """

    :param g: two-electron integrals in chemists' notation
    :param n: number of spatial orbitals
    :param no: number of (doubly) occupied orbitals
    :return:  spin-orbital two-electron integrals, sg
    """

    # build spin-orbital teis
    sg = np.zeros((2*n,2*n,2*n,2*n))

    # index 1
    for i in range(0,n):
        ia = i
        ib = i
    
        # alpha occ do nothing
        if ( ia < no ):
            ia = i
        # alpha vir shift up by no
        else :
            ia += no
        # beta occ
        if ( ib < no ):
            ib += no
        else :
            ib += n
    
        # index 2
        for j in range(0,n):
            ja = j
            jb = j
    
            # alpha occ
            if ( ja < no ):
                ja = j
            # alpha vir
            else :
                ja += no
            # beta occ
            if ( jb < no ):
                jb += no
            # beta vir
            else :
                jb += n

            # index 3
            for k in range(0,n):
                ka = k
                kb = k
    
                # alpha occ
                if ( ka < no ):
                    ka = k
                # alpha vir
                else :
                    ka += no
                # beta occ
                if ( kb < no ):
                    kb += no
                # beta vir
                else :
                    kb += n
    
                # index 4
                for l in range(0,n):
                    la = l
                    lb = l
    
                    # alpha occ
                    if ( la < no ):
                        la = l
                    # alpha vir
                    else :
                        la += no
                    # beta occ
                    if ( lb < no ):
                        lb += no
                    # beta vir
                    else :
                        lb += n
                     
                    # (aa|aa)
                    sg[ia,ja,ka,la] = g[i,j,k,l]
                    # (aa|bb)
                    sg[ia,ja,kb,lb] = g[i,j,k,l]
                    # (bb|aa)
                    sg[ib,jb,ka,la] = g[i,j,k,l]
                    # (bb|bb)
                    sg[ib,jb,kb,lb] = g[i,j,k,l]
    
    return sg
