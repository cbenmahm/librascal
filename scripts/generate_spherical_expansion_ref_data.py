"""Generate reference data for the librascal spherical expansion"""
import sys
import os

sys.path.insert(0, "../build/")
from ase.io import read
import numpy as np
import argparse
import ase
import json
from rascal.representations import SphericalExpansion
from rascal.utils import ostream_redirect
import rascal
import rascal.lib as lrl

root = os.path.abspath("../")
rascal_reference_path = os.path.join(root, "reference_data/")
inputs_path = os.path.join(rascal_reference_path, "inputs")
read_inputs_path = os.path.join("reference_data/", "inputs")
dump_path = os.path.join("reference_data/", "tests_only")

###############################################################################
###############################################################################


def get_soap_vectors(hypers, frames):
    with ostream_redirect():
        sph_expn = SphericalExpansion(**hypers)
        expansions = sph_expn.transform(frames)
        soap_vectors = expansions.get_features(sph_expn)
    return soap_vectors


###############################################################################

# dump spherical expansion


def dump_reference_json():
    import ubjson
    from copy import copy
    from itertools import product

    sys.path.insert(0, os.path.join(root, "build/"))
    sys.path.insert(0, os.path.join(root, "tests/"))

    cutoffs = [2, 3]
    gaussian_sigmas = [0.2, 0.5]
    max_radials = [4]
    max_angulars = [3]
    cutoff_smooth_widths = [1.0]
    radial_basis = ["GTO", "DVR"]
    cutoff_function_types = ["ShiftedCosine", "RadialScaling"]
    fns = [
        os.path.join(inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json"),
        os.path.join(inputs_path, "small_molecule.json"),
    ]
    fns_to_write = [
        os.path.join(read_inputs_path, "CaCrP2O7_mvc-11955_symmetrized.json"),
        os.path.join(read_inputs_path, "small_molecule.json"),
    ]
    np.random.seed(0)
    datasets_species = [1, 6, 7, 8, 15, 20, 24]
    projection_matrices = {
        str(sp): np.random.rand(
            max_angulars[0] + 1, max_radials[0], max_radials[0]
        ).tolist()
        for sp in datasets_species
    }
    optimization = [
        dict(),
        dict(Spline=dict(accuracy=1e-8)),
        dict(
            Spline=dict(accuracy=1e-8),
            RadialDimReduction=dict(projection_matrices=projection_matrices),
        ),
    ]
    data = dict(
        filenames=fns_to_write,
        cutoffs=cutoffs,
        gaussian_sigmas=gaussian_sigmas,
        max_radials=max_radials,
        rep_info=[],
    )

    for fn in fns:
        for cutoff in cutoffs:
            data["rep_info"].append([])
            for (
                gaussian_sigma,
                max_radial,
                max_angular,
                cutoff_smooth_width,
                rad_basis,
                cutoff_function_type,
                opt_args,
            ) in product(
                gaussian_sigmas,
                max_radials,
                max_angulars,
                cutoff_smooth_widths,
                radial_basis,
                cutoff_function_types,
                optimization,
            ):
                frames = [read(fn)]
                if cutoff_function_type == "RadialScaling":
                    cutoff_function_parameters = dict(
                        rate=1, scale=cutoff * 0.5, exponent=3
                    )
                else:
                    cutoff_function_parameters = dict()

                hypers = {
                    "interaction_cutoff": cutoff,
                    "cutoff_smooth_width": cutoff_smooth_width,
                    "max_radial": max_radial,
                    "max_angular": max_angular,
                    "gaussian_sigma_type": "Constant",
                    "cutoff_function_type": cutoff_function_type,
                    "cutoff_function_parameters": cutoff_function_parameters,
                    "gaussian_sigma_constant": gaussian_sigma,
                    "radial_basis": rad_basis,
                    "optimization": opt_args,
                }

                sph_expn = SphericalExpansion(**hypers)
                expansions = sph_expn.transform(frames)
                x = expansions.get_features(sph_expn)
                x[np.abs(x) < 1e-300] = 0.0
                data["rep_info"][-1].append(
                    dict(feature_matrix=x.tolist(), hypers=copy(sph_expn.hypers))
                )

    with open(
        os.path.join(root, dump_path, "spherical_expansion_reference.ubjson"),
        "wb",
    ) as f:
        ubjson.dump(data, f)


###############################################################################
###############################################################################


def main(json_dump, save_kernel):

    test_hypers = {
        "interaction_cutoff": 4.0,
        "cutoff_smooth_width": 0.0,
        "max_radial": 8,
        "max_angular": 6,
        "gaussian_sigma_type": "Constant",
        "gaussian_sigma_constant": 0.3,
    }

    nmax = test_hypers["max_radial"]
    lmax = test_hypers["max_angular"]
    nstr = "5"  # number of structures

    frames = read(os.path.join(inputs_path, "small_molecules-20.json"), ":" + str(nstr))
    species = set([atom for frame in frames for atom in frame.get_atomic_numbers()])
    nspecies = len(species)
    ncen = np.cumsum([len(frame) for frame in frames])[-1]

    x = get_soap_vectors(test_hypers, frames)
    if save_kernel is True:
        np.save(os.path.join(dump_path, "spherical_expansion_example.npy"), x)

    # --------------------------dump json reference data--------------------------#

    if json_dump == True:
        dump_reference_json()


###############################################################################
###############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json_dump", action="store_true", help="Switch for dumping json"
    )
    parser.add_argument(
        "-save_kernel", action="store_true", help="Switch for dumping json"
    )
    args = parser.parse_args()
    main(args.json_dump, args.save_kernel)
