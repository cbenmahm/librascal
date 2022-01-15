"""Generic calculator-style interface for MD"""
import ase.io
from ase.units import kB

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from copy import deepcopy
import json

from ..utils import BaseIO, load_obj, to_dict, from_dict
from ..neighbourlist.structure_manager import AtomsList, unpack_ase


class GenericMDCalculator:

    """Generic MD driver for a librascal model

    Initialize with model JSON and a structure template, and calculate
    energies and forces based on position/cell updates _assuming the
    order and identity of atoms does not change_.
    """

    matrix_indices_in_voigt_notation = (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    )

    def __init__(
        self, model_json, is_periodic, structure_template=None, atomic_numbers=None
    ):
        """Initialize a model and structure template

        Parameters
        ----------
        model_json  Filename for a JSON file defining the potential
        is_periodic Specify whether the simulation is periodic or not
                    This helps avoid confusion if a geometry's "periodic"
                    flags have been set improperly, which can happen e.g.
                    if ASE cannot read the cell information in a file.  If
                    using a structure template and this is set to True,
                    will raise an error unless at least one of the PBC
                    flags in the structure template is on.  If set to
                    False, will raise an error if all PBC flags are not
                    off.  Set to None to skip PBC checking.  If not using a
                    structure template, this setting will determine the PBC
                    flags of the created atomic structure.
        structure_template
                    Filename for an ASE-compatible Atoms object, used
                    only to initialize atom types and numbers
        atomic_numbers
                    List of atom types (atomic numbers) to initialize
                    the atomic structure in case no structure template
                    is given
        """
        super(GenericMDCalculator, self).__init__()
        self.model_filename = model_json
        self.model = load_obj(model_json)
        self.representation = self.model.get_representation_calculator()
        self.manager = None
        # Structure initialization
        self.is_periodic = is_periodic
        if structure_template is not None:
            self.template_filename = structure_template
            self.atoms = ase.io.read(structure_template, 0)
            if (is_periodic is not None) and (
                is_periodic != np.any(self.atoms.get_pbc())
            ):
                raise ValueError(
                    "Structure template PBC flags: "
                    + str(self.atoms.get_pbc())
                    + " incompatible with 'is_periodic' setting"
                )
        elif atomic_numbers is not None:
            self.atoms = ase.Atoms(numbers=atomic_numbers, pbc=is_periodic)
        else:
            raise ValueError(
                "Must specify one of 'structure_template' or 'atomic_numbers'"
            )

    def calculate(self, positions, cell_matrix):
        """Calculate energies and forces from position/cell update

        positions   Atomic positions (Nx3 matrix)
        cell_matrix Unit cell (in ASE format, cell vectors as rows)
                    (set to zero for non-periodic simulations)

        The units of positions and cell are determined by the model JSON
        file; for now, only Å is supported.  Energies, forces, and
        stresses are returned in the same units (eV and Å supported).

        Returns a tuple of energy, forces, and stress - forces are
        returned as an Nx3 array and stresses are returned as a 3x3 array

        Stress convention: The stresses have units eV/Å^3
        (volume-normalized) and are defined as the gradients of the
        energy with respect to the cell parameters.
        """
        # Quick consistency checks
        if positions.shape != (len(self.atoms), 3):
            raise ValueError(
                "Improper shape of positions (is the number of atoms consistent?)"
            )
        if cell_matrix.shape != (3, 3):
            raise ValueError("Improper shape of cell info (expected 3x3 matrix)")

        # Update ASE Atoms object (we only use ASE to handle any
        # re-wrapping of the atoms that needs to take place)
        self.atoms.set_cell(cell_matrix)
        self.atoms.set_positions(positions)

        # Convert from ASE to librascal
        if self.manager is None:
            #  happens at the begining of the MD run
            at = self.atoms.copy()
            at.wrap(eps=1e-11)
            self.manager = [at]
        elif isinstance(self.manager, AtomsList):
            structure = unpack_ase(self.atoms, wrap_pos=True)
            structure.pop("center_atoms_mask")
            self.manager[0].update(**structure)

        # Compute representations and evaluate model
        self.manager = self.representation.transform(self.manager)
        energy = self.model.predict(self.manager)
        forces = self.model.predict_forces(self.manager)
        stress_voigt = self.model.predict_stress(self.manager)
        stress_matrix = np.zeros((3, 3))
        stress_matrix[tuple(zip(*self.matrix_indices_in_voigt_notation))] = stress_voigt
        # Symmetrize the stress matrix (replicate upper-diagonal entries)
        stress_matrix += np.triu(stress_matrix, k=1).T
        #stress_matrix[np.diag_indices_from(stress_matrix)] *= 0.5
        return energy, forces, stress_matrix

class FiniteTCalculator(GenericMDCalculator):
    
    def __init__(self, model_json, is_periodic, xdos, temperature, structure_template, is_volume=None, nelectrons=None, atomic_numbers=None):
        super().__init__(model_json, is_periodic, structure_template=structure_template, atomic_numbers=atomic_numbers)
        
        try:
            self.xdos = np.load(xdos)
        except:
            print("cannot load the energy axis, please make sure it is *.npy")
        self.temperature = float(temperature)
        self.beta = 1. / (self.temperature * kB)
        self.beta_0 = 1. / (200 * kB) 
        self.natoms = len(self.atoms)
        if nelectrons == None:
            raise ValueError(
                "please enter the number of valence electrons per atom"
            )
        self.nelectrons = float(nelectrons)
        self.nelectrons = self.nelectrons * self.natoms
        if is_volume == None:
            raise ValueError(
                    "plase provide if the model is volume scaled or not"
                    )
        self.is_volume = bool(is_volume)

        # Duplicate the weights and the self_contributions of model so it can be restored at the end of the force calculation
        self.unmodified_weights = deepcopy(self.model.weights)
        self.unmodified_self_contributions = deepcopy(self.model.self_contributions)


    def calculate(self, positions, cell_matrix):
        """Calculate energies and forces from position/cell update

        positions   Atomic positions (Nx3 matrix)
        cell_matrix Unit cell (in ASE format, cell vectors as rows)
                    (set to zero for non-periodic simulations)

        The units of positions and cell are determined by the model JSON
        file; for now, only Å is supported.  Energies, forces, and
        stresses are returned in the same units (eV and Å supported).

        Returns a tuple of energy, forces, and stress - forces are
        returned as an Nx3 array and stresses are returned as a 3x3 array

        Stress convention: The stresses have units eV/Å^3
        (volume-normalized) and are defined as the gradients of the
        energy with respect to the cell parameters.
        """
        # Quick consistency checks
        if positions.shape != (len(self.atoms), 3):
            raise ValueError(
                "Improper shape of positions (is the number of atoms consistent?)"
            )
        if cell_matrix.shape != (3, 3):
            raise ValueError("Improper shape of cell info (expected 3x3 matrix)")

        # Update ASE Atoms object (we only use ASE to handle any
        # re-wrapping of the atoms that needs to take place)
        self.atoms.set_cell(cell_matrix)
        self.atoms.set_positions(positions)

        # Convert from ASE to librascal
        if self.manager is None:
            #  happens at the begining of the MD run
            at = self.atoms.copy()
            at.wrap(eps=1e-12)
            self.manager = [at]
        elif isinstance(self.manager, AtomsList):
            structure = unpack_ase(self.atoms, wrap_pos=True)
            structure.pop("center_atoms_mask")
            self.manager[0].update(**structure)
        if self.is_volume:
            volume = at.get_volume()
            self.model.weights = self.model.weights * volume
        # Compute representations and evaluate model
        self.manager = self.representation.transform(self.manager)

        # Predict the DOS of the frame
        pred_dos = self.model.predict(self.manager)[0]
        self.dos_pred = pred_dos 
        # Compute the Fermi level (T>0) and the Fermi energy (T=0)
        # TODO: implement the T=0 case using the polylog functions
        # for now use T=200 as T=0 (the difference is negligigble)

        mu_T = getmu(pred_dos, self.beta, self.xdos, n=self.nelectrons)
        mu_0 = getmu(pred_dos, self.beta_0, self.xdos, n=self.nelectrons)

        # Compute the derivative of the FD occupation with the Fermi level
        deriv_fd_T = derivative_fd_fermi(self.xdos, mu_T, self.beta)
        deriv_fd_0 = derivative_fd_fermi(self.xdos, mu_0, self.beta_0)

        # Compute the FD occupations
        fd_T = fd_distribution(self.xdos, mu_T, self.beta)
        fd_0 = fd_distribution(self.xdos, mu_0, self.beta_0)

        # Compute the "shift" terms appearing in band enegy and entropy grads
        shift_T = get_shift(pred_dos, self.xdos, deriv_fd_T)
        shift_0 = get_shift(pred_dos, self.xdos, deriv_fd_0)

        # Compute the weights for the band energy at T>0 and T=0
        weights_band_T = trapezoid(self.xdos * fd_T * self.model.weights,
                                     self.xdos, axis=1)
        weights_band_0 = trapezoid(self.xdos * fd_0 * self.model.weights,
                                     self.xdos, axis=1) 

        # Compute the weights for the band energy forces at T>0 and T=0
        f_weights_band_T = trapezoid((self.xdos - shift_T) * fd_T * self.model.weights,
                                     self.xdos, axis=1)
        f_weights_band_0 = trapezoid((self.xdos - shift_0) * fd_0 * self.model.weights,
                                     self.xdos, axis=1)

        # Compute the weights for the entropy contribution at T>0 and T=0
        s_T, x_mask_T = get_entropy(fd_T)
        weights_entr_T = trapezoid(s_T * self.model.weights[:, x_mask_T], self.xdos[x_mask_T], axis=1)
        weights_entr_T *= (-kB)

        s_0, x_mask_0 = get_entropy(fd_0)
        weights_entr_0 = trapezoid(s_0 * self.model.weights[:, x_mask_0], self.xdos[x_mask_0], axis=1)
        weights_entr_0 *= (-kB)

        # Compute the weights for the entropy forces at T>0 and T=0
        f_weights_entr_T = weights_entr_T
        f_weights_entr_T += (kB * self.beta * (mu_T - shift_T) * trapezoid(fd_T * self.model.weights, self.xdos, axis=1))

        f_weights_entr_0 = weights_entr_0
        f_weights_entr_0 += (kB * self.beta_0 * (mu_0 - shift_0) * trapezoid(fd_0 * self.model.weights, self.xdos, axis=1))

        # Compute all the contributions to the energy, the force and the stress
        # Also compute all the self contribution terms
        self.model.is_scalar = True

        # band energy at T>0
        dos_self_contributions = self.unmodified_self_contributions
        for key in self.model.self_contributions.keys():
            self.model.self_contributions[key] = trapezoid(self.xdos * fd_T * self.unmodified_self_contributions[key],#dos_self_contributions[key],
                    self.xdos)
        self.model.weights = weights_band_T
        band_T = self.model.predict(self.manager)

        self.model.weights = f_weights_band_T
        band_T_forces = self.model.predict_forces(self.manager)
        band_T_stress_v = self.model.predict_stress(self.manager)
        band_T_stress = extract_stress_matrix(band_T_stress_v)

        # band energy at T=0
        for key in self.model.self_contributions.keys():
            self.model.self_contributions[key] = trapezoid(self.xdos * fd_0 *self.unmodified_self_contributions[key],# dos_self_contributions[key],
                    self.xdos)
        self.model.weights = weights_band_0
        band_0 = self.model.predict(self.manager)

        self.model.weights = f_weights_band_0
        band_0_forces = self.model.predict_forces(self.manager)
        band_0_stress_v = self.model.predict_stress(self.manager)
        band_0_stress = extract_stress_matrix(band_0_stress_v)

        # entropy contribution energy at T>0
        for key in self.model.self_contributions.keys():
            self.model.self_contributions[key] = trapezoid(self.unmodified_self_contributions[key][x_mask_T] * s_T,
                                                           self.xdos[x_mask_T])
            self.model.self_contributions[key] *= (-kB)
            self.model.self_contributions[key] *= (-self.temperature)
        self.model.weights = -self.temperature * weights_entr_T
        entr_T = self.model.predict(self.manager)

        self.model.weights = -self.temperature * f_weights_entr_T
        entr_T_forces = self.model.predict_forces(self.manager)
        entr_T_stress_v = self.model.predict_stress(self.manager)
        entr_T_stress = extract_stress_matrix(entr_T_stress_v)

        # entropy contribution energy at T=0
        # TODO remove this after using the polylog functions
        for key in self.model.self_contributions.keys():
            self.model.self_contributions[key] = trapezoid(self.unmodified_self_contributions[key][x_mask_0] * s_0,
                                                           self.xdos[x_mask_0])
            self.model.self_contributions[key] *= (-kB)
            self.model.self_contributions[key] *= (-200)
        self.model.weights = -200 * weights_entr_0
        entr_0 = self.model.predict(self.manager)

        self.model.weights = -200 * f_weights_entr_0
        entr_0_forces = self.model.predict_forces(self.manager)
        entr_0_stress_v = self.model.predict_stress(self.manager)
        entr_0_stress = extract_stress_matrix(entr_0_stress_v)

        energy = band_T - band_0 + (entr_T - entr_0)
        force = band_T_forces - band_0_forces + (entr_T_forces - entr_0_forces)
        stress = band_T_stress - band_0_stress + (entr_T_stress - entr_0_stress)

        res = {
            "band_energy(T>0)": band_T,
            "band_energy(T=0)": band_0,
            "entropy_T": entr_T,
            "entropy_0": entr_0,
            #"f_band_energy(T>0)": band_T_forces,
            #"f_band_energy(T=0)": band_0_forces,
            #"f_entropy(T>0)": entr_T_forces - entr_0_forces,
        }

        # Restore the model
        self.model.weights = deepcopy(self.unmodified_weights)
        self.model.self_contributions = deepcopy(self.unmodified_self_contributions)
        self.model.is_scalar = False
        #self.model = load_obj(self.model_filename)
        #extras = json.dumps(res)
        return energy, force, -stress, "" 

# Some helper functions for the finite temperature calculator
def fd_distribution(x, mu, beta):
    """Fermi-Dirac distribution"""
    y = (x-mu)*beta
    ey = np.exp(-np.abs(y))
    if hasattr(x,"__iter__"):
        negs = (y<0)
        pos = (y>=0)
        try:
            y[negs] = 1 / (1+ey[negs])        
            y[pos] = ey[pos] / (1+ey[pos])
        except:
            print (x, negs, pos)
            raise
        return y
    else:
        if y<0: return 1/(1+ey)
        else: return ey/(1+ey)

def derivative_fd_fermi(x, mu, beta):
    """the derivative of the Fermi-Dirac distribution wrt
    the Fermi energy (or chemical potential)
    For now, only cases of T>10K are handled by using np.float128"""
    y = (x-mu)*beta
    y = y.astype(np.float128)
    ey = np.exp(y)
    return beta * ey * fd_distribution(x, mu, beta)**2

def nelec(dos, mu, beta, xdos):
    """ computes the number of electrons covered in the DOS """
    return trapezoid(dos * fd_distribution(xdos, mu, beta), xdos)

def getmu(dos, beta, xdos, n=2.):
    """ computes the Fermi energy of structures based on the DOS """
    return brentq(lambda x: nelec(dos ,x ,beta, xdos)-n, xdos.min(), xdos.max())

def get_shift(dos, xdos, deriv_fd):
    return trapezoid(xdos * dos * deriv_fd, xdos) / trapezoid(dos * deriv_fd, xdos)

def get_entropy(f):
    """ computes the f*log(f) term in the expession of the entropy and return the integrand 
    and a mask to determine the valid energy interval"""
    entr = f * np.log(f) + (1. - f) * np.log(1. - f)
    valid = np.logical_not(np.isnan(entr))
    return entr[valid], valid

def extract_stress_matrix(stress_voigt):
    """logic extracted from parent class"""

    matrix_indices_in_voigt_notation = (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    )
    stress_matrix = np.zeros((3, 3))
    stress_matrix[tuple(zip(*matrix_indices_in_voigt_notation))] = stress_voigt
    # Symmetrize the stress matrix (replicate upper-diagonal entries)
    stress_matrix += np.triu(stress_matrix).T
    stress_matrix[np.diag_indices_from(stress_matrix)] *= 0.5
    return stress_matrix
