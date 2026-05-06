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

# TODO: we will need to change the parser to output
# einsum code that just lists the axis to contract over instead of indices
# this is because the einsum character alphabet is only 52 characters and
# total OCC_INDICES and VIRT_INDICES are 54!
import numpy as np

__version__ = "0.0.1"

OCC_INDICES = ["i", "j", "k", "l", "m", "n", "o", "t",
               "o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9",
               "o10", "o11", "o12", "o13", "o14", "o15", "o16", "o17", "o18",
               "o19"]

VIRT_INDICES = ["a", "b", "c", "d", "e", "f", "g", "h",
                "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19"]

# numpy einsum alphabet
EINSUM_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# set up base tensors for tensor contractions
sorbs = 12   # fake system
nocc = 6
nvirt = sorbs - nocc

# active-space, spin-orbital dimensions
nocc_act = 2 # act < ext like in real case
nvirt_act = 2 # act < ext like in real case
n_act = nocc_act + nvirt_act
nocc_ext = nocc - nocc_act
nvirt_ext = nvirt - nvirt_act
n_ext = nocc_ext + nvirt_ext

# spin-integrated integers
orbs = sorbs//2 # spin-integrated uses MO index instead of spinorb index
nocca = 4 # use different a and b so that bug can be caught
noccb = nocc-nocca
nvirta = orbs - nocca
nvirtb = orbs - noccb

# TODO: implement active-space + spin-traced dimensions and slices

# spin-orbital slices
o = slice(0, nocc, 1)
v = slice(nocc, sorbs, 1)

# spin-integrated slices
oa = slice(0, nocca, 1)
ob = slice(0, noccb, 1)
va = slice(nocca, orbs, 1)
vb = slice(noccb, orbs, 1)

# active-space slices
o0 = slice(0, nocc_ext, 1)
o1 = slice(0, nocc_act, 1)
v0 = slice(nocc_ext, n_ext, 1)
v1 = slice(nocc_act, n_act, 1)

# dictionary containing dimensions
spin_orbital_dims = {
    "occ": nocc,
    "virt": nvirt,
    "all": sorbs
}

active_dims = {
    "occ": {
        '0': nocc_ext,
        '1': nocc_act,
    },
    "virt": {
        '0': nvirt_ext,
        '1': nvirt_act,
    },
    "all": {
        '0': n_ext,
        '1': n_act,
    }
}

spin_traced_dims = {
    "occ": {
        'a': nocca,
        'b': noccb,
    },
    "virt": {
        'a': nvirta,
        'b': nvirtb,
    },
    "all": {
        'a': orbs,
        'b': orbs,
    }
}

'''
# integral objects, spin-orbitals
h = np.zeros((sorbs, sorbs))
f = np.zeros((sorbs, sorbs))
g = np.zeros((sorbs, sorbs, sorbs, sorbs))

# integral objects, spin-integrated
h_aa = np.zeros((sorbs, sorbs))
h_bb = np.zeros((sorbs, sorbs))
f_aa = np.zeros((sorbs, sorbs))
f_bb = np.zeros((sorbs, sorbs))
g_aaaa = np.zeros((sorbs, sorbs, sorbs, sorbs))
g_abab = np.zeros((sorbs, sorbs, sorbs, sorbs))
g_bbbb = np.zeros((sorbs, sorbs, sorbs, sorbs))

# cluster amplitudes, spin-orbitals
t1 = np.zeros((nvirt, nocc))
t2 = np.zeros((nvirt, nvirt, nocc, nocc))
t3 = np.zeros((nvirt, nvirt, nvirt, nocc, nocc, nocc))
t4 = np.zeros((nvirt, nvirt, nvirt, nvirt, nocc, nocc, nocc, nocc))

# cluster amplitudes, spin-integrated
t1_aa = np.zeros((nvirta, nocca))
t1_bb = np.zeros((nvirtb, noccb))
t2_aaaa = np.zeros((nvirta, nvirta, nocca, nocca))
t2_abab = np.zeros((nvirta, nvirtb, nocca, noccb))
t2_bbbb = np.zeros((nvirtb, nvirtb, noccb, noccb))
t3_aaaaaa = np.zeros((nvirta, nvirta, nvirta, nocca, nocca, nocca))
t3_aabaab = np.zeros((nvirta, nvirta, nvirtb, nocca, nocca, noccb))
t3_abbabb = np.zeros((nvirta, nvirtb, nvirtb, nocca, noccb, noccb))
t3_bbbbbb = np.zeros((nvirtb, nvirtb, nvirtb, noccb, noccb, noccb))
t4_aaaaaaaa = np.zeros((nvirta, nvirta, nvirta, nvirta, nocca, nocca, nocca, nocca))
t4_aaabaaab = np.zeros((nvirta, nvirta, nvirta, nvirtb, nocca, nocca, nocca, noccb))
t4_aabbaabb = np.zeros((nvirta, nvirta, nvirtb, nvirtb, nocca, nocca, noccb, noccb))
t4_abbbabbb = np.zeros((nvirta, nvirtb, nvirtb, nvirtb, nocca, noccb, noccb, noccb))
t4_bbbbbbbb = np.zeros((nvirtb, nvirtb, nvirtb, nvirtb, noccb, noccb, noccb, noccb))

# left EOM amplitudes, spin-orbitals
# l0 is also used in spin-integrated case
l0 = 0.0
l1 = np.zeros((nocc, nvirt))
l2 = np.zeros((nocc, nocc, nvirt, nvirt))
l3 = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt))
l4 = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt))

# left EOM amplitudes, spin-integrated
l1_aa = np.zeros((nocca, nvirta))
l1_bb = np.zeros((noccb, nvirtb))
l2_aaaa = np.zeros((nocca, nocca, nvirta, nvirta))
l2_abab = np.zeros((nocca, noccb, nvirta, nvirtb))
l2_bbbb = np.zeros((noccb, noccb, nvirtb, nvirtb))
l3_aaaaaa = np.zeros((nocca, nocca, nocca, nvirta, nvirta, nvirta))
l3_aabaab = np.zeros((nocca, nocca, noccb, nvirta, nvirta, nvirtb))
l3_abbabb = np.zeros((nocca, noccb, noccb, nvirta, nvirtb, nvirtb))
l3_bbbbbb = np.zeros((noccb, noccb, noccb, nvirtb, nvirtb, nvirtb))
l4_aaaaaaaa = np.zeros((nocca, nocca, nocca, nocca, nvirta, nvirta, nvirta, nvirta))
l4_aaabaaab = np.zeros((nocca, nocca, nocca, noccb, nvirta, nvirta, nvirta, nvirtb))
l4_aabbaabb = np.zeros((nocca, nocca, noccb, noccb, nvirta, nvirta, nvirtb, nvirtb))
l4_abbbabbb = np.zeros((nocca, noccb, noccb, noccb, nvirta, nvirtb, nvirtb, nvirtb))
l4_bbbbbbbb = np.zeros((noccb, noccb, noccb, noccb, nvirtb, nvirtb, nvirtb, nvirtb))

# right EOM amplitudes
# r0 is also used in spin-integrated case
r0 = 0.0
r1 = np.zeros((nvirt, nocc))
r2 = np.zeros((nvirt, nvirt, nocc, nocc))
r3 = np.zeros((nvirt, nvirt, nvirt, nocc, nocc, nocc))
r4 = np.zeros((nvirt, nvirt, nvirt, nvirt, nocc, nocc, nocc, nocc))

# right EOM amplitudes, spin-integrated
r1_aa = np.zeros((nvirta, nocca))
r1_bb = np.zeros((nvirtb, noccb))
r2_aaaa = np.zeros((nvirta, nvirta, nocca, nocca))
r2_bbbb = np.zeros((nvirta, nvirtb, nocca, noccb))
r2_abab = np.zeros((nvirtb, nvirtb, noccb, noccb))
r3_aaaaaa = np.zeros((nvirta, nvirta, nvirta, nocca, nocca, nocca))
r3_aabaab = np.zeros((nvirta, nvirta, nvirtb, nocca, nocca, noccb))
r3_abbabb = np.zeros((nvirta, nvirtb, nvirtb, nocca, noccb, noccb))
r3_bbbbbb = np.zeros((nvirtb, nvirtb, nvirtb, noccb, noccb, noccb))
r4_aaaaaaaa = np.zeros((nvirta, nvirta, nvirta, nvirta, nocca, nocca, nocca, nocca))
r4_aaabaaab = np.zeros((nvirta, nvirta, nvirta, nvirtb, nocca, nocca, nocca, noccb))
r4_aabbaabb = np.zeros((nvirta, nvirta, nvirtb, nvirtb, nocca, nocca, noccb, noccb))
r4_abbbabbb = np.zeros((nvirta, nvirtb, nvirtb, nvirtb, nocca, noccb, noccb, noccb))
r4_bbbbbbbb = np.zeros((nvirtb, nvirtb, nvirtb, nvirtb, noccb, noccb, noccb, noccb))

# delta = identity matrix
kd = np.zeros((sorbs, sorbs))
kd_aa = np.zeros((sorbs, sorbs))
kd_bb = np.zeros((sorbs, sorbs))

# RDMs, spin-orbitals
d1 = np.zeros((sorbs, sorbs))
d2 = np.zeros((sorbs, sorbs, sorbs, sorbs))
d3 = np.zeros((sorbs, sorbs, sorbs, sorbs, sorbs, sorbs))
d4 = np.zeros((sorbs, sorbs, sorbs, sorbs, sorbs, sorbs, sorbs, sorbs))

# RDMs, spin-integrated
d1_aa = np.zeros((orbs, orbs))
d1_bb = np.zeros((orbs, orbs))
d2_aaaa = np.zeros((orbs, orbs, orbs, orbs))
d2_abab = np.zeros((orbs, orbs, orbs, orbs))
d2_bbbb = np.zeros((orbs, orbs, orbs, orbs))
d3_aaaaaa = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs))
d3_aabaab = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs))
d3_abbabb = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs))
d3_bbbbbb = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs))
d4_aaaaaaaa = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs, orbs, orbs))
d4_aaabaaab = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs, orbs, orbs))
d4_aabbaabb = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs, orbs, orbs))
d4_abbbabbb = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs, orbs, orbs))
d4_bbbbbbbb = np.zeros((orbs, orbs, orbs, orbs, orbs, orbs, orbs, orbs))
'''