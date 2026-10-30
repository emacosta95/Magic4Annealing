"""
Final energy error vs resources over T (one point per T): scatter against
max magic, max entanglement, plateau heights and max p1; passes through s*
(CrossingPasses), the (s_1, s_2) plateau map (PlateauMap) and s_2 against
the slopes of the ramps before and after it (RampSlopes).
Input: merged QuantumResourcesvsT_N={N}_LZR{tag}.npz. Set tag, N at the top.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

tag = "_FF_bounded"  # "_FF", "_bounded", "_bounded_SA" or ""
tag_T = ""  # "_more_T" or ""
N = 7
# "magic" or "magic_filtered" (the latter only in the FF files)
magic_key = "magic"

data = np.load(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR"
    + tag
    + tag_T
    + ".npz"
)

# only the keys that are plotted are read from the file
patron = re.compile(r"^T=(\d+)_evo_energy$")
Tlist = sorted(int(m.group(1)) for m in map(patron.match, data.files) if m)


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
    ramp) in units of T, decoded from theta as in SparseGRAPEModel (raw
    values with bounds_opt, softplus otherwise)."""
    theta = np.ravel(theta)
    M = (len(theta) - 1) // 3
    raw = theta[: 2 * M + 1]
    durations = raw if "bounded" in tag else np.logaddexp(0.0, raw)
    return durations / durations.sum()


# position of the avoided crossing: minimum of the gap over the (s, gap)
# samples of all the runs together, since s* does not depend on T
s_samples = np.concatenate([data[f"T={T}_schedule"] for T in Tlist])
gap_samples = np.concatenate([data[f"T={T}_gap"] for T in Tlist])
s_star = s_samples[np.argmin(gap_samples)]
print(f"s* = {s_star:.4f} (gap min = {gap_samples.min():.4g})")

energy_error = []
max_magic = []
max_entanglement = []
plateaus = []
durations = []
max_plateau = []
plateau_difference = []
crossing_passes = []
max_p1 = []
T_open = []
for T in Tlist:
    # error with respect to the instantaneous ground state at the final time
    energy_error.append(data[f"T={T}_evo_energy"][-1] - data[f"T={T}_e0"][-1])
    max_magic.append(np.max(data[f"T={T}_{magic_key}"]))
    max_entanglement.append(np.max(data[f"T={T}_entanglement"]))

    s_plateaus = plateau_heights(data[f"T={T}_theta"])
    plateaus.append(s_plateaus)
    durations.append(segment_durations(data[f"T={T}_theta"]))
    max_plateau.append(np.max(s_plateaus))
    plateau_difference.append(s_plateaus[-1] - s_plateaus[0])

    # times s(t) goes through s*: sign changes of s - s* (the exact zeros
    # are dropped, so a plateau sitting on s* is not counted twice)
    side = np.sign(data[f"T={T}_schedule"] - s_star)
    side = side[side != 0]
    crossing_passes.append(np.count_nonzero(np.diff(side)))

    max_p1.append(np.max(data[f"T={T}_p1"]))
    if not np.isclose(data[f"T={T}_schedule"][-1], 1.0):
        T_open.append(T)

if T_open:
    print(
        f"Aviso: el schedule no termina en 1 para T = {T_open}; ahí e0[-1] no es "
        "la energía fundamental del Hamiltoniano problema."
    )

path = "../../images/FrustatedRing/FinalEnergyvsQuantumResources/"
if not os.path.exists(path):
    os.makedirs(path)
name_tag = (tag + tag_T).lstrip("_")
suffix = (name_tag + "_" if name_tag else "") + f"N={N}.png"

resources = {
    "MaxMagic": (max_magic, "Maximum Magic"),
    "MaxEntanglement": (max_entanglement, "Maximum Entanglement"),
    "MaxPlateau": (max_plateau, r"Maximum plateau height $\max(s_1, s_2)$"),
    "PlateauDifference": (plateau_difference, r"Plateau height difference $s_2 - s_1$"),
    "MaxP1": (max_p1, r"Maximum first excited state population $\max_t\, p_1$"),
}
for name, (resource, xlabel) in resources.items():
    plt.figure(figsize=(7, 5))
    plt.scatter(
        resource,
        energy_error,
        c=Tlist,
        cmap="viridis",
        s=22,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    plt.colorbar(label="T")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Error in Energy at final time")
    plt.title(f"N={N}, LZR{tag + tag_T}")
    plt.grid()

    filename = f"{path}{name}_{suffix}"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Guardado: {filename}")

# the number of passes is an integer, so it goes in the colour (one category
# per value) and T on the x axis; fixed colours so that every N matches
pass_colors = {1: "tab:blue", 3: "tab:orange", 0: "tab:green"}
other_colors = iter(["tab:red", "tab:purple", "tab:brown", "tab:pink"])
Tarr = np.array(Tlist)
error_arr = np.array(energy_error)
passes_arr = np.array(crossing_passes)
plt.figure(figsize=(7, 5))
for n in sorted(set(crossing_passes)):
    mask = passes_arr == n
    plt.scatter(
        Tarr[mask],
        error_arr[mask],
        color=pass_colors.get(n) or next(other_colors),
        s=22,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        label=f"{n} pass" + ("" if n == 1 else "es"),
    )
plt.legend(title=rf"Passes through $s^*={s_star:.3f}$")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Error in Energy at final time")
plt.title(f"N={N}, LZR{tag + tag_T}")
plt.grid()

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
# small margin: the bounded optimizer often leaves a height exactly at 0 or 1
plt.xlim(-0.03, 1.03)
plt.ylim(-0.03, 1.03)
plt.gca().set_aspect("equal")
plt.xlabel(r"First plateau height $s_1$")
plt.ylabel(r"Second plateau height $s_2$")
plt.title(f"N={N}, LZR{tag + tag_T}" + rf", dotted: $s^*={s_star:.3f}$")
plt.grid()

filename = f"{path}PlateauMap_{suffix}"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

# second plateau height against the slopes ds/d(t/T) of the ramps into it
# (s_1 -> s_2, negative when the schedule goes back) and out of it (s_2 -> 1),
# coloured by the final energy error, as in the seed scan at fixed T
durations_arr = np.array(durations)
with np.errstate(divide="ignore"):
    ramps = [
        ((s_2 - s_1) / durations_arr[:, -3], r"Slope of the ramp $s_1 \to s_2$, $ds/d(t/T)$"),
        ((1.0 - s_2) / durations_arr[:, -1], r"Slope of the ramp $s_2 \to 1$, $ds/d(t/T)$"),
    ]
# the shortest T give almost instantaneous ramps (slopes up to ~1e10, or a
# jump of zero duration): they are left out so that the rest can be seen
max_slope = 20
steep = (np.abs(ramps[0][0]) > max_slope) | (np.abs(ramps[1][0]) > max_slope)
if steep.any():
    print(f"Aviso: |pendiente| > {max_slope} para T = {Tarr[steep].tolist()}; no se dibujan.")
norm = LogNorm(vmin=error_arr[~steep].min(), vmax=error_arr[~steep].max())
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (x, xlabel) in zip(axes, ramps):
    ax.axhline(s_star, color="gray", linestyle=":", linewidth=1, zorder=1)
    points = ax.scatter(
        x[~steep],
        s_2[~steep],
        c=error_arr[~steep],
        cmap="viridis_r",
        norm=norm,
        s=22,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax.set_xlabel(xlabel)
    ax.grid()
axes[0].set_ylabel(r"Second plateau height $s_2$")
axes[0].set_ylim(-0.03, 1.03)
fig.colorbar(points, ax=axes, label="Error in Energy at final time")
fig.suptitle(f"N={N}, LZR{tag + tag_T}" + rf", dotted: $s^*={s_star:.3f}$")

filename = f"{path}RampSlopes_{suffix}"
plt.savefig(filename, dpi=200, bbox_inches="tight")
print(f"Guardado: {filename}")

plt.show()
