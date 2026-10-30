"""
General landscape: energy, max magic and max entanglement (one evolution
per point) on the plane through 3 of the optima of
ParametersLZR_T={T}_N={N}.npz (build/FrustatedRing/LandscapesLZR/parameter_generation_LZR.py).
T, N and the 3 seeds are set at the top.
"""

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

T = 120

N = 7  # odd; magic in the spin basis for N <= 9, free fermions above
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
resolution = 25
# processes the points of the plane are spread over. 1: the magic uses JAX,
# ~300 MB and ~1.5 cores per process; several of them filled the memory of a
# laptop. Raise it on the cluster to finish one landscape sooner
n_workers = 1

# theta -> (final energy, maximum magic, maximum entanglement)
# (src/landscape_utils.py)
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


def observables(theta):
    # module level: the worker processes of the scan call it. They import
    # this file again (everything above), not what follows the guard
    return landscape.observables(theta)


if __name__ == "__main__":
    filename = f"../../generated/FrustatedRing/ParametersLZR/ParametersLZR_T={T}_N={N}.npz"
    data = np.load(filename)
    chosen_seeds = [7, 8, 9]

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

    filename_img_energy = f"../../images/FrustatedRing/FinalEnergyLandscapeLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"
    filename_img_max_entanglement = f"../../images/FrustatedRing/MaxEntanglementLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"
    filename_img_max_magic = f"../../images/FrustatedRing/MaxMagicLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"

    energies = {
        "theta1": landscape.energy(theta1),
        "theta2": landscape.energy(theta2),
        "theta3": landscape.energy(theta3),
    }

    plot_energy_landscape(
        A,
        B,
        E,
        coords,
        energies,
        title=f"Energy landscape T={T} N={N}",
        save_path=filename_img_energy,
        region=region,
    )

    plot_max_magic_landscape(
        A,
        B,
        max_magic,
        coords,
        energies,
        title=f"Max magic landscape T={T} N={N}",
        save_path=filename_img_max_magic,
        region=region,
    )

    plot_max_entanglement_landscape(
        A,
        B,
        max_entanglement,
        coords,
        energies,
        title=f"Max entanglement landscape T={T} N={N}",
        save_path=filename_img_max_entanglement,
        region=region,
    )
