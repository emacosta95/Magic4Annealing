"""
Max-magic landscape on the plane through 3 of the 10 optima of
ParametersLZR_T={T}_N={N}.npz (build/FrustatedRing/parameter_generation_LZR.py).
Usage: T N COMBO, COMBO = index into combinations(range(10), 3).
Magic in the spin basis for N <= 9, free fermions above; 1 worker (JAX memory).
"""

import os
import sys
from itertools import combinations

import numpy as np

from src.landscape_utils import (
    RingLandscape,
    bounds_region,
    max_magic_landscape,
    plot_max_magic_landscape,
)
from src.sparse_grape_method import lzs_bounds

T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; magic in the spin basis for N <= 9, free fermions above
J, JL, JR = 1.0, 0.5, 0.45

# ── time evolution parameters ─────────────────────────────────────────────────
# propagation steps per unit of time, as in the optimization of the
# parameters (build/FrustatedRing/parameter_generation_LZR.py)
steps_per_unit = 100
# time between the states the maximum magic is taken over
magic_dt = 0.1
n_samples = 1000  # Majorana samples per SRE estimate (free fermions, N > 9)

# ── optimization parameters ───────────────────────────────────────────────────
number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
resolution = 50
# processes the points of the plane are spread over. 1: the magic uses JAX,
# ~300 MB and ~1.5 cores per process; several of them filled the memory of a
# laptop. Raise it on the cluster to finish one landscape sooner
n_workers = 1

# theta -> maximum magic (src/landscape_utils.py)
landscape = RingLandscape(
    N,
    T,
    steps_per_unit=steps_per_unit,
    number_parameters=number_parameters,
    J=J,
    JL=JL,
    JR=JR,
    magic_dt=magic_dt,
    n_samples=n_samples,
    magic=True,
)


def max_magic(theta):
    # module level: the worker processes of the scan call it. They import
    # this file again (everything above), not what follows the guard
    return landscape.max_magic(theta)


if __name__ == "__main__":
    filename = f"../../generated/FrustatedRing/ParametersLZR_T={T}_N={N}.npz"
    data = np.load(filename)
    seed_combinations = np.array(list(combinations(range(10), 3)))
    chosen_seeds = seed_combinations[int(sys.argv[3])]

    theta1 = data["theta_list"][chosen_seeds[0]]
    theta2 = data["theta_list"][chosen_seeds[1]]
    theta3 = data["theta_list"][chosen_seeds[2]]

    A, B, max_magic_values, coords = max_magic_landscape(
        theta1,
        theta2,
        theta3,
        max_magic,
        resolution=resolution,
        n_workers=n_workers,
    )
    # part of the plane within the bounds of the optimization (bounds_opt)
    region = bounds_region(
        theta1, theta2, theta3, A, B, lzs_bounds(number_parameters)
    )

    path = f"../../images/FrustatedRing/MaxMagicLandscapeLZR_T={T}_N={N}"
    if not os.path.exists(path):
        os.makedirs(path)

    filename_img_max_magic = f"{path}/{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"

    energies = {
        "theta1": landscape.energy(theta1),
        "theta2": landscape.energy(theta2),
        "theta3": landscape.energy(theta3),
    }

    plot_max_magic_landscape(
        A,
        B,
        max_magic_values,
        coords,
        energies,
        title=f"Max magic landscape T={T} N={N}",
        save_path=filename_img_max_magic,
        region=region,
    )
