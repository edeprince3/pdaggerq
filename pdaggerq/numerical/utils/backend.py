#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2026 A. Eugene DePrince III
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
Backend dispatch for the CC/EOMCC/CC-response solvers, so they can be handed either a psi4
wave function + molecule, or a converged pyscf mean-field object + Mole, without hardcoding
which package's integrals module or molecule API to use.
 
Detection is duck-typed off the object's defining module (psi4_integrals.py / pyscf_integrals.py
are imported lazily, only once the backend is known -- so a caller with only one of psi4/pyscf
installed never hits an ImportError from the other), so this module itself has no hard
dependency on either package.
"""
 
 
def detect_backend(wfn_or_mf):
    """
    figure out whether an object came from psi4 or pyscf, by the module its class is defined in
 
    :param wfn_or_mf: a psi4 wave function, or a converged pyscf mean-field object
    :return: 'psi4' or 'pyscf'
    """
 
    module_name = type(wfn_or_mf).__module__
 
    if module_name.startswith('psi4'):
        return 'psi4'
    elif module_name.startswith('pyscf'):
        return 'pyscf'
 
    raise TypeError(
        "could not determine backend (psi4 or pyscf) for object of type "
        f"'{type(wfn_or_mf).__module__}.{type(wfn_or_mf).__name__}'; expected a psi4 wave "
        "function or a pyscf mean-field object")
 
 
def get_integrals_module(backend):
    """
    lazily import and return the integrals module for a backend, so importing this module (or
    a solver that uses it) never requires both psi4 and pyscf to be installed
 
    :param backend: 'psi4' or 'pyscf' (see detect_backend)
    :return: the pdaggerq.numerical.utils.psi4_integrals or .pyscf_integrals module
    """
 
    if backend == 'psi4':
        from pdaggerq.numerical.utils import psi4_integrals
        return psi4_integrals
    elif backend == 'pyscf':
        from pdaggerq.numerical.utils import pyscf_integrals
        return pyscf_integrals
 
    raise ValueError(f"unknown backend '{backend}' (expected 'psi4' or 'pyscf')")
 
 
def nuclear_repulsion_energy(mol, backend):
    """
    nuclear repulsion energy, for either a psi4 molecule or a pyscf Mole
 
    :param mol: a psi4 molecule object, or a pyscf Mole object
    :param backend: 'psi4' or 'pyscf' (see detect_backend)
    :return: the nuclear repulsion energy
    """
 
    if backend == 'psi4':
        return mol.nuclear_repulsion_energy()
    elif backend == 'pyscf':
        return mol.energy_nuc()
 
    raise ValueError(f"unknown backend '{backend}' (expected 'psi4' or 'pyscf')")
 
 
def nuclear_dipole_components(mol, backend):
    """
    nuclear contribution to the dipole moment, (x, y, z), in atomic units, for either a psi4
    molecule or a pyscf Mole. Both packages express molecular geometry in Bohr by default, so
    no unit conversion is needed to match the (also atomic-units) electronic dipole integrals.
 
    :param mol: a psi4 molecule object, or a pyscf Mole object
    :param backend: 'psi4' or 'pyscf' (see detect_backend)
    :return: (nuc_dip_x, nuc_dip_y, nuc_dip_z)
    """
 
    if backend == 'psi4':
        nuc_dip_x = 0.0
        nuc_dip_y = 0.0
        nuc_dip_z = 0.0
        for i in range(mol.natom()):
            nuc_dip_x += mol.Z(i) * mol.x(i)
            nuc_dip_y += mol.Z(i) * mol.y(i)
            nuc_dip_z += mol.Z(i) * mol.z(i)
        return nuc_dip_x, nuc_dip_y, nuc_dip_z
 
    elif backend == 'pyscf':
        import numpy as np
        charges = mol.atom_charges()
        coords = mol.atom_coords()  # Bohr by default
        nuc_dip = np.einsum('i,ix->x', charges, coords)
        return float(nuc_dip[0]), float(nuc_dip[1]), float(nuc_dip[2])
 
    raise ValueError(f"unknown backend '{backend}' (expected 'psi4' or 'pyscf')")
