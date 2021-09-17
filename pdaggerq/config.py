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
sorbs = 8   # fake system
nocc = 4
nvirt = sorbs - nocc
o = slice(0, nocc, 1)
v = slice(nocc, sorbs, 1)

h = np.zeros((sorbs, sorbs))
f = np.zeros((sorbs, sorbs))
g = np.zeros((sorbs, sorbs, sorbs, sorbs))

t1 = np.zeros((nvirt, nocc))
t2 = np.zeros((nvirt, nvirt, nocc, nocc))
t3 = np.zeros((nvirt, nvirt, nvirt, nocc, nocc, nocc))
t4 = np.zeros((nvirt, nvirt, nvirt, nvirt, nocc, nocc, nocc, nocc))

l1 = np.zeros((nocc, nvirt))
l2 = np.zeros((nocc, nocc, nvirt, nvirt))
l3 = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt))
l4 = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt))

r1 = np.zeros((nocc, nvirt))
r2 = np.zeros((nocc, nocc, nvirt, nvirt))
r3 = np.zeros((nocc, nocc, nocc, nvirt, nvirt, nvirt))
r4 = np.zeros((nocc, nocc, nocc, nocc, nvirt, nvirt, nvirt, nvirt))

kd = np.zeros((sorbs, sorbs))
