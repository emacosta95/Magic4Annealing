"""
Max-entanglement landscape on the plane through 3 of the 10 optima of
ParametersLZR_T={T}_N={N}.npz (build/FrustatedRing/parameter_generation_LZR.py).
Usage: T N COMBO, COMBO = index into combinations(range(10), 3).
Free fermions, parallel over the points of the plane.
"""

import os
import sys
from itertools import combinations

import numpy as np

from src.landscape_utils import (
    RingLandscape,
    bounds_region,
    max_entanglement_landscape,
    plot_max_entanglement_landscape,
)
from src.parallel import available_cpus
from src.sparse_grape_method import lzs_bounds

T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; free fermions, no 2^N limit
J, JL, JR = 1.0, 0.5, 0.45

# ── time evolution parameters ─────────────────────────────────────────────────
# propagation steps per unit of time, as in the optimization of the
# parameters (build/FrustatedRing/parameter_generation_LZR.py)
steps_per_unit = 100

# ── optimization parameters ───────────────────────────────────────────────────
number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
resolution = 50
# processes the points of the plane are spread over (all the cores of the job)
n_workers = available_cpus()

# theta -> maximum half-chain entanglement, in free fermions
# (src/landscape_utils.py)
landscape = RingLandscape(
    N,
    T,
    steps_per_unit=steps_per_unit,
    number_parameters=number_parameters,
    J=J,
    JL=JL,
    JR=JR,
)


def max_entanglement(theta):
    # module level: the worker processes of the scan call it. They import
    # this file again (everything above), not what follows the guard
    return landscape.max_entanglement(theta)


if __name__ == "__main__":
    filename = f"../../generated/FrustatedRing/ParametersLZR/ParametersLZR_T={T}_N={N}.npz"
    data = np.load(filename)
    seed_combinations = np.array(list(combinations(range(10), 3)))
    chosen_seeds = seed_combinations[int(sys.argv[3])]

    theta1 = data["theta_list"][chosen_seeds[0]]
    theta2 = data["theta_list"][chosen_seeds[1]]
    theta3 = data["theta_list"][chosen_seeds[2]]

    A, B, max_entanglement_values, coords = max_entanglement_landscape(
        theta1,
        theta2,
        theta3,
        max_entanglement,
        resolution=resolution,
        n_workers=n_workers,
    )
    # part of the plane within the bounds of the optimization (bounds_opt)
    region = bounds_region(
        theta1, theta2, theta3, A, B, lzs_bounds(number_parameters)
    )

    path = f"../../images/FrustatedRing/LandscapesLZR/MaxEntanglement/T={T}_N={N}"
    if not os.path.exists(path):
        os.makedirs(path)

    filename_img_max_entanglement = f"{path}/{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"

    energies = {
        "theta1": landscape.energy(theta1),
        "theta2": landscape.energy(theta2),
        "theta3": landscape.energy(theta3),
    }

    plot_max_entanglement_landscape(
        A,
        B,
        max_entanglement_values,
        coords,
        energies,
        title=f"Max entanglement landscape T={T} N={N}",
        save_path=filename_img_max_entanglement,
        region=region,
    )
