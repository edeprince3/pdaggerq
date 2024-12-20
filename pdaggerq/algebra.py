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

from typing import Tuple
import copy
import numpy as np

# This imports fake values of all these quantities (zero tensors) so
# we can build the optimal tensor contraction orderings
# NOTE: THESE ARE PROTECTED VARIABLE NAMES FOR THIS MODULE
from pdaggerq.config import *

# these are integrals, RDMs, etc. that require explicit slicing in einsum
tensors_with_slices = ['h', 'g', 'f', 'kd', 'd1', 'd2', 'd3', 'd4']

class Index:

    def __init__(self, name: str, support: str):
        """
        Generate an index that acts on a particular space

        Later we might  want to generate the axis over which this index
        is used to generate einsum code.  See numpypsi4 for this.

        :param name: how the index shows up
        :param support: where the index ranges. Options: occ,virt,all
        """
        self.name = name
        self.support = support

    def __repr__(self):
        return "{}".format(self.name)

    def __eq__(self, other):
        if not isinstance(other, Index):
            raise TypeError("Can't compare non Index object to Index")
        return other.name == self.name and other.support == self.support

    def __ne__(self, other):
        return not self.__eq__(other)


class BaseTerm:
    """
    Base object for building named abstract tensors

    These objects one can ONLY be composed by multiplication with other BaseTerms
    to produce new TensorTerms.  They can also be checked for equality.

    :param indices: Tuple[Index, ...], tuple of indices for the tensor
    :param str name: name of the tensor
    :param str boson: boson (in cavity qed, photon) number in the format of `_np` where `n`
                      is an integer. If purely fermionic/electronic then leave as an empty
                      string ``. Always start string with an underscore unless empty.
    :param str spin: spin sector associated with tensor. If spin orbital then leave as
                     an empty string ``.  the alpha-alpha block is `_aa`, beta-beta block
                     is `_bb`. etc.  Always start string with an underscore unless empty.
    """

    def __init__(self, *, indices: Tuple[Index, ...], name: str, boson: str, spin: str, active: str):
        self.name = name
        self.boson = boson
        self.spin = spin
        self.active = active
        self.indices = indices
        self.varname = name+boson+spin+active

    def __repr__(self):
        return ("{}".format(self.name) +
                "{}".format(self.boson) +
                "{}".format(self.spin) +
                "{}".format(self.active) +
                "(" + ",".join(repr(xx) for xx in self.indices) + ")")

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __mul__(self, other):
        # what about numpy floats  and such?
        if isinstance(other, BaseTerm):
            return TensorTerm((copy.deepcopy(self), other))
        elif isinstance(other, TensorTerm):
            self_copy = copy.deepcopy(self)
            return other.__mul__(self_copy)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        name_equality = other.name == self.name
        if len(self.indices) == len(other.indices):
            index_equal = all([self.indices[xx] == other.indices[xx] for xx in
                               range(len(self.indices))])
            return name_equality and index_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def createArray(self):
        # slicing of non-amplitude tensors is done at einsum_string(), not here
        idx_all = True if self.name in tensors_with_slices else False
        arrayDims = []
        for xx,index in enumerate(self.indices):
            # TODO: add case where active *AND* spin can be toggled on together
            if (self.active != '') and (self.spin == ''):
                if idx_all:
                    arrayDims.append(active_dims["all"][self.active[xx+1]])
                else:
                    arrayDims.append(active_dims[index.support][self.active[xx+1]])
            elif (self.spin != '') and (self.active == ''):
                if idx_all:
                    arrayDims.append(spin_traced_dims["all"][self.spin[xx+1]])
                else:
                    arrayDims.append(spin_traced_dims[index.support][self.spin[xx+1]])
            else:
                if idx_all:
                    arrayDims.append(spin_orbital_dims["all"])
                else:
                    arrayDims.append(spin_orbital_dims[index.support])

        globals()[self.varname] = np.zeros(tuple(arrayDims))
        return

    def cleanArray(self):
        if self.varname in globals():
            del globals()[self.varname]
        else:
            pass
        return


class TensorTermAction:
    """
    Object for representing a type of transform on TensorTerm.

    An example of Transformation can be duplicate or permute indices,

    permute action will be used in conjunction with einsum contraction to
    minimize contraction work.
    """

    def __init__(self, *, indices: Tuple[Index, ...], name: str, spin: str):
        self.name = name
        self.spin = spin
        self.indices = indices

    def __repr__(self):
        return "{}".format(self.name) + "{}".format(self.spin) + "(" + ",".join(
            repr(xx) for xx in self.indices) + ")"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        name_equality = other.name == self.name
        if len(self.indices) == len(other.indices):
            index_equal = all([self.indices[xx] == other.indices[xx] for xx in
                               range(len(self.indices))])
            return name_equality and index_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class TensorTerm:
    """
    collection  of BaseTerms that can be translated to a einsnum contraction
    """

    def __init__(self, base_terms: Tuple[BaseTerm, ...], coefficient=1.0,
                 permutation_ops=None):
        self.base_terms = base_terms
        self.coefficient = coefficient
        if permutation_ops is not None:
            if len(permutation_ops) == 0:
                self.actions = None
            else:
                self.actions = permutation_ops
        else:
            self.actions = None

    def __repr__(self):
        if self.actions is None:
            return "{: 5.4f} ".format(self.coefficient) + "*".join(
                xx.__repr__() for xx in self.base_terms)
        else:
            return "{: 5.4f} ".format(self.coefficient) + "*".join(
                xx.__repr__() for xx in self.actions) + "*".join(
                xx.__repr__() for xx in self.base_terms)

    def __mul__(self, other):
        self_copy = copy.deepcopy(self)
        if isinstance(other, (int, float, complex)):
            self_copy.coefficient *= other
        elif isinstance(other, BaseTerm):
            self_copy.base_terms = self_copy.base_terms + (other,)
        return self_copy

    def __rmul__(self, other):
        return self.__mul__(other)

    def einsum_string(self, update_val,
                      output_variables=None,
                      occupied=['i', 'j', 'k', 'l', 'm', 'n', 'I', 'J', 'K', 'L', 'M', 'N'],
                      virtual=['a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F'],
                      occ_char=None,
                      virt_char=None,
                      optimize=True):
        einsum_out_strings = ""
        einsum_tensors = []
        tensor_out_idx = []
        einsum_strings = []
        if occ_char is None:
            # in our code this will be a slice. o = slice(None, nocc)
            occ_char = 'o'
        if virt_char is None:
            virt_char = 'v'  # v = slice(nocc, None)


        # these are amplitudes, having fixed shape and do not require explicit slicing
        tensors_amps  = ['t'+str(i) for i in range(1,5)]
        tensors_amps += ['r'+str(i) for i in range(0,5)]
        tensors_amps += ['l'+str(i) for i in range(0,5)]

        for bt in self.base_terms:
            tensor_index_ranges = [] # 'o', 'v', or ':'
            # parse indices and process them
            string_indices = [xx.name for xx in bt.indices]
            for idx_loc,idx_type in enumerate(string_indices):
                if bt.name in tensors_with_slices:
                    # Parse slice type (o or v or :)
                    idx_str = ''
                    if idx_type in occupied:
                        idx_str += occ_char
                    elif idx_type in virtual:
                        idx_str += virt_char
                    else:
                        idx_str += ':'

                    # add spin label (a/b) to o/v slices, but not : slice
                    if bt.spin != '' and idx_str!= ':':
                        idx_str += bt.spin[idx_loc+1]

                    # add active-space label (0/1) to o/v slices, but not : slice
                    if bt.active != '' and idx_str!= ':':
                        idx_str += bt.active[idx_loc+1]

                    tensor_index_ranges.append(idx_str)

                # add current index to einsum output indices
                if output_variables is not None:
                    if idx_type in output_variables:
                        tensor_out_idx.append(idx_type)

            if bt.name in tensors_amps:
                # T, L, and R amplitudes have fixed shape even with spin, so just append their content
                einsum_tensors.append(bt.name + bt.boson + bt.spin + bt.active)
            else:
                # append the appropriate slice for other tensors (e.g. h[o,o] or d2[:,v,:,v])
                einsum_tensors.append(
                    bt.name + bt.boson + bt.spin + bt.active + "[" + ", ".join(tensor_index_ranges) + "]")

            # this is a list of indices for einsum inputs (e.g., 'ijef', 'efab')
            einsum_strings.append("".join(string_indices))

        if tensor_out_idx:
            out_tensor_ordered = list(filter(None, [
                xx if xx in tensor_out_idx else None for xx in
                output_variables]))
            einsum_out_strings += "->{}".format("".join(out_tensor_ordered))

        teinsum_string = "= {: 5.15f} * einsum(\'".format(self.coefficient)

        if len(einsum_strings) > 2 and optimize:
            # construct arrays on the fly
            for bt in self.base_terms:
                bt.createArray()
            einsum_path_string = "np.einsum_path(\'".format(self.coefficient)
            einsum_path_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ", optimize=\'optimal\')"
            # TODO: this still introduce a bug when optimizing expressions with r1/l1 in it
            # Alternatively use "optimize=True" everywhere, but it will be recomputing the
            # optimal path in every iteration, making it more expensive in the long run.
            einsum_optimal_path = eval(einsum_path_string)
            # print(einsum_optimal_path[1])
            teinsum_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ", optimize={})".format(
                einsum_optimal_path[0])
            # clean up arrays
            for bt in self.base_terms:
                bt.cleanArray()
        else:
            teinsum_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ")"
        if update_val is not None and self.actions is None:
            teinsum_string = update_val + " " + '+' + teinsum_string

        # TODO: need logic for paired permutations PP2(ia,jb), PP3(ia,jb,kc), and PP6(ia,jb,kc)

        elif update_val is not None and self.actions is not None:
            # now generate logic for single permutation
            original_out = list(filter(None,
                                       [xx if xx in tensor_out_idx else None for
                                        xx in output_variables]))
            outstrings = [[+1, original_out]]

            for act in self.actions:
                # check if we have a permutation
                if not isinstance(act, ContractionPermuter):
                    raise NotImplementedError(
                        "currently only permutations are implemented")

                # get all sets of exchanged indices
                exchanged_indices = [xx.name for xx in act.indices]

                permuted_outstrings = []
                for perm in outstrings:
                    # Generate new permuted outstring by
                    # a) make copy of current permutation
                    # b) find position of indices in string
                    # c) swap positions of indices
                    tmp_outsrings = copy.deepcopy(perm[1])
                    ii, jj = tmp_outsrings.index(
                        exchanged_indices[0]), tmp_outsrings.index(
                        exchanged_indices[1])
                    tmp_outsrings[ii], tmp_outsrings[jj] = tmp_outsrings[jj], \
                                                           tmp_outsrings[ii]

                    # store permuted set with a -1 phase
                    permuted_outstrings.append([perm[0] * -1, tmp_outsrings])

                outstrings = outstrings + permuted_outstrings

            # now that we've generated all the permutations
            # generate the reshapped summands.
            teinsum_string = 'contracted_intermediate' + " " + teinsum_string + "\n"
            teinsum_string += update_val
            teinsum_string += " += "
            update_val_line = []
            for ots in outstrings:
                new_string = ""
                new_string += '{: 5.5f} * '.format(ots[0])
                if tuple(ots[1]) == tuple(original_out):
                    new_string += 'contracted_intermediate'
                else:
                    new_string += 'einsum(\'{}->{}\', contracted_intermediate) '.format(
                        ''.join(original_out), "".join(ots[1]))
                update_val_line.append(new_string)
            teinsum_string += " + ".join(update_val_line)
        return teinsum_string

class Right0amps(BaseTerm):

    def __init__(self, *, indices=(), name='r0', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Right1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r1', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Right2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r2', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Right3amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r3', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Right4amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='r4', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Left0amps(BaseTerm):

    def __init__(self, *, indices=(), name='l0', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Left1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l1', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Left2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l2', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Left3amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l3', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class Left4amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='l4', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class D1(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='d1', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class D2(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='d2', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class D3(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='d3', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class D4(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='d4', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class T1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t1', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class T2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t2', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class T3amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t3', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class T4amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='t4', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class OneBody(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='h', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class FockMat(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='f', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)


class TwoBody(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='g', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

    def __repr__(self):
        return "<{},{}||{},{}>{}".format(self.indices[0], self.indices[1],
                                       self.indices[2], self.indices[3], self.spin)

class Delta(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='kd', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

    def __repr__(self):
        return "d({},{})".format(self.indices[0], self.indices[1])

class Dipole(BaseTerm):

    def __init__(self, *, indices=Tuple[Index, ...], name='dipole', spin='', active='', boson=''):
        super().__init__(indices=indices, name=name, spin=spin, active=active, boson=boson)

class ContractionPermuter(TensorTermAction):

    def __init__(self, *, spin='', indices=Tuple[Index, ...], name='P'):
        super().__init__(indices=indices, name=name, spin=spin)

class ContractionPairPermuter6(TensorTermAction):

    def __init__(self, *, spin='', indices=Tuple[Index, ...], name='PP6'):
        super().__init__(indices=indices, name=name, spin=spin)

class ContractionPairPermuter2(TensorTermAction):

    def __init__(self, *, spin='', indices=Tuple[Index, ...], name='PP2'):
        super().__init__(indices=indices, name=name, spin=spin)

class ContractionPairPermuter3(TensorTermAction):

    def __init__(self, *, spin='', indices=Tuple[Index, ...], name='PP3'):
        super().__init__(indices=indices, name=name, spin=spin)
