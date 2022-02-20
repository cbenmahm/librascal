"""Generic calculator-style interface for MD"""
from re import I
import ase.io
from ase.units import kB
from matplotlib.pyplot import get

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
    
    def __init__(self, model_json, is_periodic, xdos, temperature, structure_template, nelectrons=None, atomic_numbers=None, contribution="all"):
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
        self.xdos = np.load(xdos)
        if contribution not in ["all", "band_T", "band_0", "entr_T", "entr_0"]:
            raise ValueError(
                "please provide the correct contribution, choose between: all, band_T, band_0, entr_T and entr_0"
            )
        self.contribution = contribution

        # Duplicate the weights and the self_contributions of model so it can be restored at the end of the force calculation
        self.model.unmodified_weights = deepcopy(self.model.weights)
        self.model.unmodified_self_contributions = deepcopy(self.model.self_contributions)


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
        # Compute representations and evaluate model
        self.manager = self.representation.transform(self.manager)
        
        # Predict the DOS of the frame
        self.dos_pred = self.model.predict(self.manager)[0]
        if self.contribution == "band_0":
            energy, force, stress = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta_0, self.nelectrons, self.xdos)
            reset_model(self.model) 
            return -energy, -force, -stress, ""
        
        elif self.contribution == "band_T":
            energy, force, stress = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta, self.nelectrons, self.xdos)
            reset_model(self.model) 
            return energy, force, stress, ""
        
        elif self.contribution == "entr_0":
            energy, force, stress = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta_0, 200, self.nelectrons, self.xdos)
            reset_model(self.model) 
            return -energy, -force, -stress, ""
        
        elif self.contribution == "entr_T":
            energy, force, stress = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta, self.temperature, self.nelectrons, self.xdos)
            reset_model(self.model) 
            return energy, force, stress, ""
        
        else:
            energy_band_0, force_band_0, stress_band_0 = get_band_contribution(self.model, self.manager,pred_dos, self.beta_0, self.nelectrons, self.xdos)
            reset_model(self.model) 
            
            energy_band_T, force_band_T, stress_band_T = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta, self.nelectrons, self.xdos)
            reset_model(self.model) 
            
            energy_entr_0, force_entr_0, stress_entr_0 = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta_0, 200, self.nelectrons, self.xdos)
            reset_model(self.model) 
            
            energy_entr_T, force_entr_T, stress_entr_T = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta, self.temperature, self.nelectrons, self.xdos)
            reset_model(self.model) 
            
            energy = energy_band_T - energy_band_0 + energy_entr_T - energy_entr_0
            force = force_band_T - force_band_0 + force_entr_T - force_entr_0
            stress = stress_band_T - stress_band_0 + stress_entr_T - stress_entr_0

            return energy, force, stress, ""

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

def get_band_contribution(model, manager, dos, beta, nelectrons, xdos):
    
    mu = getmu(dos, beta, xdos, n=nelectrons)
    
    deriv_fd = derivative_fd_fermi(xdos, mu, beta)
    fd = fd_distribution(xdos, mu, beta)

    shift = get_shift(dos, xdos, deriv_fd)

    weights = trapezoid(xdos * fd * model.unmodified_weights,
                                     xdos, axis=1)
    f_weights = trapezoid((xdos - shift) * fd * model.unmodified_weights,
                                     xdos, axis=1)
    
    model.is_scalar = True

    for key in model.self_contributions.keys():
            model.self_contributions[key] = trapezoid(xdos * fd * model.unmodified_self_contributions[key],
                    xdos)
    
    model.weights = weights
    energy = model.predict(manager)

    model.weights = f_weights
    force = model.predict_forces(manager)

    stress = model.predict_stress(manager)
    stress = extract_stress_matrix(stress)

    return energy, force, stress

def get_entropy_contribution(model, manager, dos, beta, temperature, nelectrons, xdos):

    mu = getmu(dos, beta, xdos, n=nelectrons)
    
    deriv_fd = derivative_fd_fermi(xdos, mu, beta)
    fd = fd_distribution(xdos, mu, beta)

    shift = get_shift(dos, xdos, deriv_fd)

    s, x_mask = get_entropy(fd)

    model.is_scalar = True

    for key in model.self_contributions.keys():
            model.self_contributions[key] = trapezoid(model.unmodified_self_contributions[key][x_mask] * s,
                                                           xdos[x_mask])
            model.self_contributions[key] *= (-kB)
            model.self_contributions[key] *= (-temperature)
    
    weights = trapezoid(s * model.unmodified_weights[:, x_mask], xdos[x_mask], axis=1)
    weights *= (-kB)
    model.weights = -temperature * weights
    energy = model.predict(manager)

    f_weights = weights
    f_weights += (kB * beta * (mu - shift) * trapezoid(fd * model.unmodified_weights, xdos, axis=1))
    model.weights = -temperature * f_weights
    force = model.predict_forces(manager)

    stress = model.predict_stress(manager)
    stress = extract_stress_matrix(stress)

    return energy, force, stress

def reset_model(model):
        model.weights = deepcopy(model.unmodified_weights)
        model.self_contributions = deepcopy(model.unmodified_self_contributions)
        model.is_scalar = False
