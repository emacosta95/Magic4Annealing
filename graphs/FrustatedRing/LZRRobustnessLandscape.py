import os
import sys
from itertools import combinations

import numpy as np

from src.fidelity_robustness import AXES
from src.landscape_utils import (
    RingLandscape,
    bounds_region,
    plot_landscape,
    scan_plane,
)
from src.parallel import available_cpus
from src.sparse_grape_method import lzs_bounds

# Landscape of the fidelity robustness against the global fields
# V_a = lambda sum_i sigma^a_i (a = x, y, z, spin-basis axes: x is the driver
# axis, z the target one): G_aa of src/fidelity_robustness.py, i.e. the
# variance Var(int_0^T V_a(t) dt) in the Heisenberg picture, with
# 1 - F ~ lambda^2 G_aa, F the fidelity with the unperturbed final state.
# Spin basis, full 2^N space (sigma^y and sigma^z leave the Z2 +1 sector and
# are not Gaussian in free fermions). One figure per axis.

T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; full 2^N space, the augmented vector has 4 * 2^N entries
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

# theta -> (G_xx, G_yy, G_zz) (src/landscape_utils.py)
landscape = RingLandscape(
    N,
    T,
    steps_per_unit=steps_per_unit,
    number_parameters=number_parameters,
    J=J,
    JL=JL,
    JR=JR,
    robustness=True,
)


def robustness(theta):
    # module level: the worker processes of the scan call it. They import
    # this file again (everything above), not what follows the guard
    return landscape.robustness(theta)


if __name__ == "__main__":
    filename = f"../../generated/FrustatedRing/ParametersLZR_T={T}_N={N}.npz"
    data = np.load(filename)
    seed_combinations = np.array(list(combinations(range(10), 3)))
    chosen_seeds = seed_combinations[int(sys.argv[3])]

    theta1 = data["theta_list"][chosen_seeds[0]]
    theta2 = data["theta_list"][chosen_seeds[1]]
    theta3 = data["theta_list"][chosen_seeds[2]]

    # G[..., a] = G_aa, the three axes from the same augmented evolution
    A, B, G, coords = scan_plane(
        theta1,
        theta2,
        theta3,
        robustness,
        resolution=resolution,
        n_workers=n_workers,
    )
    # part of the plane within the bounds of the optimization (bounds_opt)
    region = bounds_region(
        theta1, theta2, theta3, A, B, lzs_bounds(number_parameters)
    )

    energies = {
        "theta1": landscape.energy(theta1),
        "theta2": landscape.energy(theta2),
        "theta3": landscape.energy(theta3),
    }

    for a, axis in enumerate(AXES):
        path = f"../../images/FrustatedRing/RobustnessLandscapeLZR{axis.upper()}_T={T}_N={N}"
        if not os.path.exists(path):
            os.makedirs(path)

        # G_aa spans orders of magnitude over the plane: log color scale
        # (1 - F ~ lambda^2 G_aa, see the top of the file)
        plot_landscape(
            A,
            B,
            G[..., a],
            coords,
            energies,
            label=rf"$G_{{{axis}{axis}}}$",
            title=rf"Robustness landscape, $V = \lambda\, \Sigma_i\, \sigma^{axis}_i$, T={T} N={N}",
            save_path=f"{path}/{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png",
            region=region,
            log=True,
        )
