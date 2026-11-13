"""
Same plots as FinalEnergyvsQuantumResources.py, but at a fixed T: each point
is one optimization from a different random initial point (seed). Also s_1,
s_2 against the speeds |ds/dt| of the three ramps (S1vsRampSlopes,
S2vsRampSlopes), the
maximum magic against s_1, s_2 (MaxMagicvsPlateaus) and against the time at
which it is reached (MaxMagicvsTime), and the final error and the maximum
magic against the ramp speeds (ErrorvsRampSlopes, MaxMagicvsRampSlopes).
Input: QuantumResourcesSeedScan_N={N}_T={T}_LZR{tag}.npz
(build/FrustatedRing/QuantumResourcesSeedScan/run_LZR_seed_scan_FF_bounded.py + merge_LZR_seed_scan.py).
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
    f"../../generated/FrustatedRing/QuantumResourcesSeedScan/QuantumResourcesSeedScan_N={N}_T={T}_LZR"
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


def segment_durations(theta):
    """Durations of the 2M+1 segments of the LZS ansatz (ramp, plateau, ...,
    ramp), decoded from theta as in SparseGRAPEModel (raw values with
    bounds_opt, softplus otherwise) and rescaled so that they add up to T."""
    theta = np.ravel(theta)
    M = (len(theta) - 1) // 3
    raw = theta[: 2 * M + 1]
    durations = raw if "bounded" in tag else np.logaddexp(0.0, raw)
    return durations / durations.sum() * T


# position of the avoided crossing: minimum of the gap of H(s), computed by
# the build in the job of seed 0 (sampling it along the schedules is not
# reliable at a fixed T: fast ramps can skip the crossing between two points)
s_star = float(data["seed=0_s_star"][0])
print(f"s* = {s_star:.4f} (gap min = {float(data['seed=0_gap_min'][0]):.4g})")

energy_error = []
max_magic = []
max_magic_time = []
max_entanglement = []
plateaus = []
durations = []
max_plateau = []
plateau_difference = []
crossing_passes = []
max_p1 = []
max_excited = []
seeds_open = []
for seed in seeds:
    # error with respect to the instantaneous ground state at the final time
    energy_error.append(
        data[f"seed={seed}_evo_energy"][-1] - data[f"seed={seed}_e0"][-1]
    )
    magic_t = data[f"seed={seed}_{magic_key}"]
    max_magic.append(np.max(magic_t))
    # magic is computed at the times time_sub (subsampled state grid)
    max_magic_time.append(data[f"seed={seed}_time_sub"][np.argmax(magic_t)] / T)
    max_entanglement.append(np.max(data[f"seed={seed}_entanglement"]))

    s_plateaus = plateau_heights(data[f"seed={seed}_theta"])
    plateaus.append(s_plateaus)
    durations.append(segment_durations(data[f"seed={seed}_theta"]))
    max_plateau.append(np.max(s_plateaus))
    plateau_difference.append(s_plateaus[-1] - s_plateaus[0])

    # times s(t) goes through s*: sign changes of s - s* (the exact zeros
    # are dropped, so a plateau sitting on s* is not counted twice)
    side = np.sign(data[f"seed={seed}_schedule"] - s_star)
    side = side[side != 0]
    crossing_passes.append(np.count_nonzero(np.diff(side)))

    max_p1.append(np.max(data[f"seed={seed}_p1"]))
    # population out of the ground state (p0: lowest level, even sector)
    max_excited.append(np.max(1.0 - data[f"seed={seed}_p0"]))
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


path = f"../../images/FrustatedRing/FinalEnergyvsQuantumResourcesSeedScan/N={N}_T={T}_LZR{tag}/"
if not os.path.exists(path):
    os.makedirs(path)
title = f"N={N}, T={T}, LZR{tag}, {len(seeds)} seeds"

resources = {
    "MaxMagic": (max_magic, "Maximum Magic"),
    "MaxEntanglement": (max_entanglement, "Maximum Entanglement"),
    "MaxPlateau": (max_plateau, r"Maximum plateau height $\max(s_1, s_2)$"),
    "PlateauDifference": (plateau_difference, r"Plateau height difference $s_2 - s_1$"),
    "MaxP1": (max_p1, r"Maximum first excited state population $\max_t\, p_1$"),
    "MaxExcited": (max_excited, r"Maximum excited population $\max_t\,(1 - p_0)$"),
}
for name, (resource, xlabel) in resources.items():
    plt.figure(figsize=(7, 5))
    scatter_by_passes(resource)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Error in Energy at final time")
    plt.title(title)
    plt.grid()

    filename = f"{path}{name}.png"
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

filename = f"{path}CrossingPasses.png"
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

filename = f"{path}PlateauMap.png"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

# speeds |ds/dt| of the ramps into the first plateau (0 -> s_1), into the
# second (s_1 -> s_2) and out of it (s_2 -> 1), in physical time as in
# FinalEnergyvsQuantumResources.py so that both can be compared. Linear
# axis: many ramps have zero speed (s_2 = 1 or s_1 = s_2)
durations_arr = np.array(durations)
ramps = [
    (s_1 / durations_arr[:, 0], r"Speed of the ramp $0 \to s_1$, $|ds/dt|$"),
    (np.abs(s_2 - s_1) / durations_arr[:, -3], r"Speed of the ramp $s_1 \to s_2$, $|ds/dt|$"),
    ((1.0 - s_2) / durations_arr[:, -1], r"Speed of the ramp $s_2 \to 1$, $|ds/dt|$"),
]
# the ramp s_1 -> s_2 goes back (s_2 < s_1) or forward: marker
back = s_2 < s_1
markers = [(back, "o", r"$s_2 < s_1$ (goes back)"), (~back, "^", r"$s_2 > s_1$")]


def plot_against_ramps(y, ylabel, c, cmap, cnorm, clabel):
    """One panel per ramp, y against its speed, coloured by c, with the
    direction of the ramp s_1 -> s_2 in the marker and the best seed marked
    with a star. Returns the figure and the axes."""
    fig, axes = plt.subplots(1, len(ramps), figsize=(16, 5), sharey=True)
    for ax, (x, xlabel) in zip(axes, ramps):
        for direction, marker, _ in markers:
            points = ax.scatter(
                x[direction],
                y[direction],
                c=c[direction],
                cmap=cmap,
                norm=cnorm,
                marker=marker,
                s=22,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        ax.scatter(
            x[best],
            y[best],
            marker="*",
            facecolors="none",
            s=200,
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )
        ax.set_xlabel(xlabel)
        ax.grid()
    axes[0].set_ylabel(ylabel)
    # grey markers in the legend: the colour of the points is c. Below the
    # panels, so that it covers no points
    fig.legend(
        handles=[
            plt.Line2D([], [], marker=marker, color="gray", linestyle="", label=label)
            for _, marker, label in markers
        ]
        + [
            plt.Line2D(
                [], [], marker="*", color="black", markerfacecolor="none", markersize=12,
                linestyle="", label=f"best seed ({seeds[best]})",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.45, 0.02),
        ncol=3,
    )
    fig.colorbar(points, ax=axes, label=clabel)
    return fig, axes


# each plateau height against the ramp speeds, coloured by the final energy
# error, one figure per plateau: checks whether changes in s_1, s_2 come with
# changes in how fast the ramps around them cross s*
norm = LogNorm(vmin=error_arr.min(), vmax=error_arr.max())
for name, s_plateau, ylabel in [
    ("S1", s_1, r"First plateau height $s_1$"),
    ("S2", s_2, r"Second plateau height $s_2$"),
]:
    fig, axes = plot_against_ramps(
        s_plateau, ylabel, error_arr, "viridis_r", norm,
        "Error in Energy at final time",
    )
    for ax in axes:
        ax.axhline(s_star, color="gray", linestyle=":", linewidth=1, zorder=1)
    axes[0].set_ylim(-0.03, 1.03)
    fig.suptitle(title + rf", dotted: $s^*={s_star:.3f}$")

    filename = f"{path}{name}vsRampSlopes.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Guardado: {filename}")

# maximum magic against the plateau heights and against the time at which it
# is reached, coloured by the final energy error: which schedule variables
# tell the groups of the MaxMagic plot apart
max_magic_arr = np.array(max_magic)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (s_plateau, xlabel) in zip(
    axes, [(s_1, r"First plateau height $s_1$"), (s_2, r"Second plateau height $s_2$")]
):
    ax.axvline(s_star, color="gray", linestyle=":", linewidth=1, zorder=1)
    points = ax.scatter(
        s_plateau,
        max_magic_arr,
        c=energy_error,
        cmap="viridis_r",
        norm=norm,
        s=22,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax.scatter(
        s_plateau[best],
        max_magic_arr[best],
        marker="*",
        facecolors="none",
        s=200,
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
        label=f"best seed ({seeds[best]})",
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(xlabel)
    ax.grid()
axes[0].set_ylabel("Maximum Magic")
axes[0].legend(loc="upper left", framealpha=0.4)
fig.colorbar(points, ax=axes, label="Error in Energy at final time")
fig.suptitle(title + rf", dotted: $s^*={s_star:.3f}$")

filename = f"{path}MaxMagicvsPlateaus.png"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

plt.figure(figsize=(7, 5))
plt.scatter(
    max_magic_time,
    max_magic_arr,
    c=energy_error,
    cmap="viridis_r",
    norm=norm,
    s=22,
    edgecolors="white",
    linewidths=0.4,
    zorder=3,
)
plt.colorbar(label="Error in Energy at final time")
plt.scatter(
    max_magic_time[best],
    max_magic_arr[best],
    marker="*",
    facecolors="none",
    s=200,
    edgecolors="black",
    linewidths=1.0,
    zorder=4,
    label=f"best seed ({seeds[best]})",
)
plt.legend(loc="upper left", framealpha=0.4)
plt.xlim(0, 1)
plt.xlabel(r"Time of the maximum magic $t/T$")
plt.ylabel("Maximum Magic")
plt.title(title)
plt.grid()

filename = f"{path}MaxMagicvsTime.png"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

# final energy error against the ramp speeds, coloured by s_2: whether the
# precision depends on how fast the ramps go
fig, axes = plot_against_ramps(
    error_arr, "Error in Energy at final time", s_2, "viridis", plt.Normalize(0.0, 1.0),
    r"Second plateau height $s_2$",
)
axes[0].set_yscale("log")
fig.suptitle(title)

filename = f"{path}ErrorvsRampSlopes.png"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

# maximum magic against the ramp speeds, coloured by the final energy error:
# whether the speeds tell the groups of the MaxMagic plot apart
fig, axes = plot_against_ramps(
    max_magic_arr, "Maximum Magic", error_arr, "viridis_r", norm,
    "Error in Energy at final time",
)
fig.suptitle(title)

filename = f"{path}MaxMagicvsRampSlopes.png"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

plt.show()
