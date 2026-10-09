import os

import matplotlib.pyplot as plt
import numpy as np

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.fidelity_robustness import AXES, FidelityRobustness
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.time_grid import schedules_from_saved

tag = "_bounded"  # "_NoGrad", "_step_test", "_bounded", "_no_random" or ""
tag_T = ""  # "_more_T" or ""
N = 7
T = 100

# range of the global field lambda, V = lambda * sum_i sigma^a_i, for each axis a
lambda_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)}
n_lambdas = 1001

data = np.load(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR"
    + tag
    + tag_T
    + ".npz"
)
# s(t) at the cell midpoints of the optimized schedule (old and new files)
saved = schedules_from_saved(data, prefix=f"T={T}_")
schedule_ctrl = saved["schedule_ctrl"]

J, JL, JR = 1.0, 0.5, 0.45

jij, hz = frustrated_ring_jij_hz(N, J, JL, JR)

nqubits = N
# full 2^N space: sigma^y and sigma^z fields leave the Z2 +1 sector
target_hamiltonian = get_longitudinal_hamiltonian(jij, hz)
driver_hamiltonian = get_driver_hamiltonian(nqubits=nqubits)

dim = 2**nqubits
psi_init_full = np.ones(dim, dtype=complex) / np.sqrt(dim)

schedules = {
    "LZR schedule": schedule_ctrl,
    # s = t / T on the same control grid
    "Linear schedule": saved["times_ctrl"] / T,
}

# each fidelity is relative to the unperturbed final state of its own schedule
robustness = {}
for name, s_ctrl in schedules.items():
    print(name)
    robustness[name] = FidelityRobustness(
        nqubits,
        driver_hamiltonian,
        target_hamiltonian,
        psi_init_full,
        s_ctrl,
        T,
        verbose=True,
    )
    robustness[name].compute()

name_tag = (tag + tag_T).lstrip("_")
filename = (name_tag + "_" if name_tag else "") + f"N={N}_T={T}.png"

for a, axis in enumerate(AXES):
    lambda_min, lambda_max = lambda_range[axis]
    lambdas = np.linspace(lambda_min, lambda_max, n_lambdas)

    # lam =[x_0..x_{N-1}, y_0..y_{N-1}, z_0..z_{N-1}]: lambda on the N
    # entries of this axis, 0 on the other 2N
    lam = np.zeros((n_lambdas, 3, nqubits))
    lam[:, a, :] = lambdas[:, None]

    plt.figure(figsize=(7, 5))
    bottom = 1.0
    for k, name in enumerate(schedules):
        fidelity = robustness[name].fidelity(lam)

        # scale of validity of the second order along this direction
        g_axis = robustness[name].infidelity(lam[-1]) / lambdas[-1] ** 2
        lambda_star = 1 / np.sqrt(g_axis) if g_axis > 0 else np.inf
        print(
            f"{name}, sum_i sigma^{axis}_i: Re G = {g_axis:.6g}, "
            f"lambda* = {lambda_star:.4g} (second order off by > ~1 % for "
            f"|lambda| > {lambda_star / 3:.4g})"
        )

        plt.plot(
            lambdas,
            fidelity,
            "-",
            linewidth=1.5,
            color=f"C{k}",
            label=rf"{name}, $\lambda^*$ = {lambda_star:.2g}",
        )
        # band where the second order is reliable, in the color of its curve
        if np.isfinite(lambda_star):
            plt.axvspan(
                max(-lambda_star / 3, lambda_min),
                min(lambda_star / 3, lambda_max),
                color=f"C{k}",
                alpha=0.2,
            )
        bottom = min(bottom, fidelity.min())
    # the second-order fidelity is not bounded below: the axis stops at 0
    bottom = max(bottom, 0.0)
    margin = 0.03 * (1 - bottom)
    plt.ylim(bottom - margin, 1 + margin)
    plt.xlim(lambda_min, lambda_max)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Fidelity (second order)")
    plt.title(
        rf"$V = \lambda \sum_i \sigma^{axis}_i$,  N={N}, T={T}, LZR{tag + tag_T}"
    )
    plt.legend(title=r"shaded: $|\lambda| \leq \lambda^*/3$")
    plt.grid()

    path = f"../../images/FrustatedRing/FidelityRobustnessGlobal{axis.upper()}/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"{path}{filename}", dpi=200, bbox_inches="tight")
    print(f"Guardado: {path}{filename}")

plt.show()
