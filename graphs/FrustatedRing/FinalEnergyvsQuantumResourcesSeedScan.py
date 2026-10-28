"""
Same plots as FinalEnergyvsQuantumResources.py, but at a fixed T: each point
is one optimization from a different random initial point (seed).
Input: QuantumResourcesSeedScan_N={N}_T={T}_LZR{tag}.npz
(build/FrustatedRing/run_LZR_seed_scan_FF_bounded.py + merge_LZR_seed_scan.py).
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

tag = "_FF_bounded"
N = 13
T = 200
# "magic" or "magic_filtered" (the latter only in the FF files)
magic_key = "magic"

data = np.load(
    f"../../generated/FrustatedRing/QuantumResourcesSeedScan_N={N}_T={T}_LZR"
    + tag
    + ".npz"
)

# only the keys that are plotted are read from the file
patron = re.compile(r"^seed=(\d+)_evo_energy$")
seeds = sorted(int(m.group(1)) for m in map(patron.match, data.files) if m)


def plateau_heights(theta):
    """Plateau heights s_1..s_M of the LZS ansatz, decoded from theta as in
    SparseGRAPEModel (raw values with bounds_opt, sigmoid otherwise). The
    bounded runs are the ones with "bounded" in the tag."""
    theta = np.ravel(theta)
    M = (len(theta) - 1) // 3
    raw = theta[2 * M + 1 :]
    return raw if "bounded" in tag else 1.0 / (1.0 + np.exp(-raw))


# position of the avoided crossing: minimum of the gap of H(s), computed by
# the build in the job of seed 0 (sampling it along the schedules is not
# reliable at a fixed T: fast ramps can skip the crossing between two points)
s_star = float(data["seed=0_s_star"][0])
print(f"s* = {s_star:.4f} (gap min = {float(data['seed=0_gap_min'][0]):.4g})")

energy_error = []
max_magic = []
max_entanglement = []
plateaus = []
max_plateau = []
plateau_difference = []
crossing_passes = []
max_p1 = []
seeds_open = []
for seed in seeds:
    # error with respect to the instantaneous ground state at the final time
    energy_error.append(
        data[f"seed={seed}_evo_energy"][-1] - data[f"seed={seed}_e0"][-1]
    )
    max_magic.append(np.max(data[f"seed={seed}_{magic_key}"]))
    max_entanglement.append(np.max(data[f"seed={seed}_entanglement"]))

    s_plateaus = plateau_heights(data[f"seed={seed}_theta"])
    plateaus.append(s_plateaus)
    max_plateau.append(np.max(s_plateaus))
    plateau_difference.append(s_plateaus[-1] - s_plateaus[0])

    # times s(t) goes through s*: sign changes of s - s* (the exact zeros
    # are dropped, so a plateau sitting on s* is not counted twice)
    side = np.sign(data[f"seed={seed}_schedule"] - s_star)
    side = side[side != 0]
    crossing_passes.append(np.count_nonzero(np.diff(side)))

    max_p1.append(np.max(data[f"seed={seed}_p1"]))
    if not np.isclose(data[f"seed={seed}_schedule"][-1], 1.0):
        seeds_open.append(seed)

if seeds_open:
    print(
        f"Aviso: el schedule no termina en 1 para seed = {seeds_open}; ahí e0[-1] "
        "no es la energía fundamental del Hamiltoniano problema."
    )

error_arr = np.array(energy_error)
passes_arr = np.array(crossing_passes)
# the seed best_of_seeds keeps in run_LZR_FF_bounded.py (lowest final energy,
# i.e. lowest error): the point of this T in the QuantumResourcesvsT file
best = int(np.argmin(error_arr))
print(f"Best seed: {seeds[best]} (error = {error_arr[best]:.4g})")

# fixed colours for the number of passes, the same as in
# FinalEnergyvsQuantumResources.py
pass_colors = {1: "tab:blue", 3: "tab:orange", 0: "tab:green"}
other_colors = iter(["tab:red", "tab:purple", "tab:brown", "tab:pink"])
pass_values = sorted(set(crossing_passes))
colors = {n: pass_colors.get(n) or next(other_colors) for n in pass_values}


def pass_label(n):
    return f"{n} pass" + ("" if n == 1 else "es")


def scatter_by_passes(x):
    """Scatter of the error against x, one colour per number of passes, and
    the best seed marked with a star."""
    x = np.asarray(x)
    for n in pass_values:
        mask = passes_arr == n
        plt.scatter(
            x[mask],
            error_arr[mask],
            color=colors[n],
            s=22,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=pass_label(n),
        )
    plt.scatter(
        x[best],
        error_arr[best],
        marker="*",
        color=colors[passes_arr[best]],
        s=160,
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
        label=f"best seed ({seeds[best]})",
    )
    plt.legend(title=rf"Passes through $s^*={s_star:.3f}$")


path = "../../images/FrustatedRing/FinalEnergyvsQuantumResourcesSeedScan/"
if not os.path.exists(path):
    os.makedirs(path)
name_tag = tag.lstrip("_")
suffix = (name_tag + "_" if name_tag else "") + f"N={N}_T={T}.png"
title = f"N={N}, T={T}, LZR{tag}, {len(seeds)} seeds"

resources = {
    "MaxMagic": (max_magic, "Maximum Magic"),
    "MaxEntanglement": (max_entanglement, "Maximum Entanglement"),
    "MaxPlateau": (max_plateau, r"Maximum plateau height $\max(s_1, s_2)$"),
    "PlateauDifference": (plateau_difference, r"Plateau height difference $s_2 - s_1$"),
    "MaxP1": (max_p1, r"Maximum first excited state population $\max_t\, p_1$"),
}
for name, (resource, xlabel) in resources.items():
    plt.figure(figsize=(7, 5))
    scatter_by_passes(resource)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Error in Energy at final time")
    plt.title(title)
    plt.grid()

    filename = f"{path}{name}_{suffix}"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Guardado: {filename}")

# the number of passes is an integer: strip plot, one column per value with a
# small horizontal jitter so that the points do not pile up
rng = np.random.default_rng(0)
position = np.array([pass_values.index(n) for n in crossing_passes], dtype=float)
plt.figure(figsize=(7, 5))
scatter_by_passes(position + rng.uniform(-0.15, 0.15, size=len(position)))
plt.xticks(range(len(pass_values)), [pass_label(n) for n in pass_values])
plt.xlim(-0.5, len(pass_values) - 0.5)
# the columns fill the whole axes: legend outside, to the right
plt.legend(
    title=rf"Passes through $s^*={s_star:.3f}$",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)
plt.yscale("log")
plt.xlabel(rf"Passes through the avoided crossing $s^*={s_star:.3f}$")
plt.ylabel("Error in Energy at final time")
plt.title(title)
plt.grid(axis="y")

filename = f"{path}CrossingPasses_{suffix}"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

# map of the plateau heights (s_1, s_2), coloured by the final energy error:
# above the diagonal the schedule is monotonic, below it goes back
s_1, s_2 = np.array(plateaus).T
plt.figure(figsize=(6.5, 5.5))
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, zorder=1)
plt.axvline(s_star, color="gray", linestyle=":", linewidth=1, zorder=1)
plt.axhline(s_star, color="gray", linestyle=":", linewidth=1, zorder=1)
plt.scatter(
    s_1,
    s_2,
    c=energy_error,
    cmap="viridis_r",
    norm=LogNorm(),
    s=22,
    edgecolors="white",
    linewidths=0.4,
    zorder=3,
)
plt.colorbar(label="Error in Energy at final time")
plt.scatter(
    s_1[best],
    s_2[best],
    marker="*",
    facecolors="none",
    s=200,
    edgecolors="black",
    linewidths=1.0,
    zorder=4,
    label=f"best seed ({seeds[best]})",
)
plt.legend(loc="upper left", framealpha=0.4)
# small margin: the bounded optimizer often leaves a height exactly at 0 or 1
plt.xlim(-0.03, 1.03)
plt.ylim(-0.03, 1.03)
plt.gca().set_aspect("equal")
plt.xlabel(r"First plateau height $s_1$")
plt.ylabel(r"Second plateau height $s_2$")
plt.title(title + rf", dotted: $s^*={s_star:.3f}$")
plt.grid()

filename = f"{path}PlateauMap_{suffix}"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

plt.show()
