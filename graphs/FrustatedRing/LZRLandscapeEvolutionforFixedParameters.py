"""
Evolution with T of the landscape around 3 fixed vectors: the plane is the
one through 3 optima found at T, and energy, max magic and max entanglement
are evaluated on it at Ti = T + 10 (k - 2), k = 0..4 (one job per k).
Usage: T N SEED1 SEED2 SEED3 k. Input: ParametersLZR_T={T}_N={N}.npz
(build/FrustatedRing/LandscapesLZR/parameter_generation_LZR.py).
"""

import os
import sys

import numpy as np

from src.landscape_utils import (
    RingLandscape,
    bounds_region,
    plot_energy_landscape,
    plot_max_entanglement_landscape,
    plot_max_magic_landscape,
    scan_plane,
)
from src.sparse_grape_method import lzs_bounds

T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; magic in the spin basis for N <= 9, free fermions above

seed1 = int(sys.argv[3])

seed2 = int(sys.argv[4])

seed3 = int(sys.argv[5])
k = int(sys.argv[6])

chosen_seeds = [seed1, seed2, seed3]

Tlist = [T + i * 10 for i in range(-2, 3)]
Ti = Tlist[k]
J, JL, JR = 1.0, 0.5, 0.45

# ── time evolution parameters ─────────────────────────────────────────────────
# propagation steps per unit of time, as in the optimization of the
# parameters (build/FrustatedRing/LandscapesLZR/parameter_generation_LZR.py)
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

# theta -> (final energy, maximum magic, maximum entanglement) for a total
# time Ti; the parameters are the ones optimized for T
# (src/landscape_utils.py)
landscape = RingLandscape(
    N,
    Ti,
    steps_per_unit=steps_per_unit,
    number_parameters=number_parameters,
    J=J,
    JL=JL,
    JR=JR,
    magic_dt=magic_dt,
    n_samples=n_samples,
    magic=True,
)


def observables(theta):
    # module level: the worker processes of the scan call it. They import
    # this file again (everything above), not what follows the guard
    return landscape.observables(theta)


if __name__ == "__main__":
    filename = f"../../generated/FrustatedRing/ParametersLZR/ParametersLZR_T={T}_N={N}.npz"
    data = np.load(filename)

    theta1 = data["theta_list"][chosen_seeds[0]]
    theta2 = data["theta_list"][chosen_seeds[1]]
    theta3 = data["theta_list"][chosen_seeds[2]]

    # the three observables of every point from the same evolution
    A, B, values, coords = scan_plane(
        theta1,
        theta2,
        theta3,
        observables,
        resolution=resolution,
        n_workers=n_workers,
    )
    E, max_magic, max_entanglement = values[..., 0], values[..., 1], values[..., 2]
    # part of the plane within the bounds of the optimization (bounds_opt)
    region = bounds_region(
        theta1, theta2, theta3, A, B, lzs_bounds(number_parameters)
    )

    path = f"../../images/FrustatedRing/LandscapesLZR/Evolution/T={T}_N={N}_seeds={chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}"
    if not os.path.exists(path):
        os.makedirs(path)

    filename_img_energy = f"{path}/EnergyLandscape_Ti={Ti}.png"
    filename_img_max_entanglement = f"{path}/MaxEntanglement_Ti={Ti}.png"
    filename_img_max_magic = f"{path}/MaxMagic_Ti={Ti}.png"

    energies = {
        "theta1": landscape.energy(theta1),
        "theta2": landscape.energy(theta2),
        "theta3": landscape.energy(theta3),
    }

    # fixed color scales: the same for every Ti
    plot_energy_landscape(
        A,
        B,
        E,
        coords,
        energies,
        title=f"Energy landscape T={Ti} N={N}",
        save_path=filename_img_energy,
        vmin=-4.55,
        vmax=-3.896,
        region=region,
    )

    plot_max_magic_landscape(
        A,
        B,
        max_magic,
        coords,
        energies,
        title=f"Max magic landscape T={Ti} N={N}",
        save_path=filename_img_max_magic,
        vmin=1.63,
        vmax=2.55,
        region=region,
    )

    plot_max_entanglement_landscape(
        A,
        B,
        max_entanglement,
        coords,
        energies,
        title=f"Max entanglement landscape T={Ti} N={N}",
        save_path=filename_img_max_entanglement,
        vmin=0.49,
        vmax=1.1,
        region=region,
    )
