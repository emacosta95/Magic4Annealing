import os
import re

import numpy as np
from matplotlib import animation


def load_data(archivo_salida):
    """
    Lee el .npz combinado y devuelve un diccionario:
    { T (int): {nombre_variable: array, ...}, ... }
    """
    data = np.load(archivo_salida)

    patron = re.compile(r"^T=(\d+)_(.+)$")
    resultado = {}

    for clave in data.files:
        match = patron.match(clave)
        if not match:
            print(
                f"Aviso: clave '{clave}' no coincide con el patrón esperado, se omite."
            )
            continue

        T = int(match.group(1))
        nombre_variable = match.group(2)

        if T not in resultado:
            resultado[T] = {}

        resultado[T][nombre_variable] = data[clave]

    return resultado


typeLZR = ""  # "_NoGrad" or ""

data_linear = load_data("../../generated/FrustatedRing/QuantumResourcesvsT_linear.npz")
data_LZR = load_data(
    "../../generated/FrustatedRing/QuantumResourcesvsT_LZR" + typeLZR + ".npz"
)

Tlist_linear = sorted(data_linear.keys())
Tlist_LZR = sorted(data_LZR.keys())

min_gap_linear = []
max_entanglement_linear = []
max_magic_linear = []
max_entanglement_gs_level = []
max_magic_gs_level = []
entanglement_integral_linear = []
entanglement_integral_gs_level = []
magic_integral_linear = []
magic_integral_gs_level = []
for T in Tlist_linear:
    min_gap_linear.append(np.min(data_linear[T]["gap"]))
    max_entanglement_linear.append(np.max(data_linear[T]["entanglement"]))
    max_magic_linear.append(np.max(data_linear[T]["magic"]))
    max_entanglement_gs_level.append(np.max(data_linear[T]["entanglement_gs_level"]))
    max_magic_gs_level.append(np.max(data_linear[T]["magic_gs_level"]))
    entanglement_integral_linear.append(
        (1 / T)
        * np.trapezoid(data_linear[T]["entanglement"], x=data_linear[T]["time_sub"])
    )
    entanglement_integral_gs_level.append(
        (1 / T)
        * np.trapezoid(
            data_linear[T]["entanglement_gs_level"], x=data_linear[T]["time_sub"]
        )
    )
    magic_integral_linear.append(
        (1 / T) * np.trapezoid(data_linear[T]["magic"], x=data_linear[T]["time_sub"])
    )
    magic_integral_gs_level.append(
        (1 / T)
        * np.trapezoid(data_linear[T]["magic_gs_level"], x=data_linear[T]["time_sub"])
    )


min_gap_LZR = []
max_entanglement_LZR = []
max_magic_LZR = []
entanglement_integral_LZR = []
magic_integral_LZR = []
for T in Tlist_LZR:
    min_gap_LZR.append(np.min(data_LZR[T]["gap"]))
    max_entanglement_LZR.append(np.max(data_LZR[T]["entanglement"]))
    max_magic_LZR.append(np.max(data_LZR[T]["magic"]))
    entanglement_integral_LZR.append(
        (1 / T) * np.trapezoid(data_LZR[T]["entanglement"], x=data_LZR[T]["time_sub"])
    )
    magic_integral_LZR.append(
        (1 / T) * np.trapezoid(data_LZR[T]["magic"], x=data_LZR[T]["time_sub"])
    )


plt.figure(figsize=(7, 5))
plt.plot(
    Tlist_linear,
    min_gap_linear,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule",
)
plt.plot(
    Tlist_LZR, min_gap_LZR, ".-", linewidth=1, markersize=4.5, label="LZR schedule"
)
plt.xlabel("T")
plt.ylabel("Minimum Gap")
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(
    Tlist_linear,
    max_entanglement_linear,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule",
)
plt.plot(
    Tlist_LZR,
    max_entanglement_LZR,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="LZR schedule",
)
plt.plot(
    Tlist_linear,
    max_entanglement_gs_level,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule (ground state level)",
)
plt.xlabel("T")
plt.ylabel("Maximum Entanglement")
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(
    Tlist_linear,
    max_magic_linear,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule",
)
plt.plot(
    Tlist_LZR, max_magic_LZR, ".-", linewidth=1, markersize=4.5, label="LZR schedule"
)
plt.plot(
    Tlist_linear,
    max_magic_gs_level,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule (ground state level)",
)
plt.xlabel("T")
plt.ylabel("Maximum Magic")
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(
    Tlist_linear,
    entanglement_integral_linear,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule",
)
plt.plot(
    Tlist_LZR,
    entanglement_integral_LZR,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="LZR schedule",
)
plt.plot(
    Tlist_linear,
    entanglement_integral_gs_level,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule (ground state level)",
)
plt.xlabel("T")
plt.ylabel("Entanglement Integral")
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(
    Tlist_linear,
    magic_integral_linear,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule",
)
plt.plot(
    Tlist_LZR,
    magic_integral_LZR,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="LZR schedule",
)
plt.plot(
    Tlist_linear,
    magic_integral_gs_level,
    ".-",
    linewidth=1,
    markersize=4.5,
    label="Linear schedule (ground state level)",
)
plt.xlabel("T")
plt.ylabel("Magic Integral")
plt.legend()
plt.show()


N_linear = len(Tlist_linear)
N_LZR = len(Tlist_LZR)

Tlist_linear_reduced = [Tlist_linear[i] for i in range(0, N_linear, 20)]
Tlist_LZR_reduced = [Tlist_LZR[i] for i in range(0, N_LZR, 20)]


sequence_linear = list(range(len(Tlist_linear_reduced)))
sequence_LZR = list(range(len(Tlist_LZR_reduced)))

fig_gif, ax_gif = plt.subplots(figsize=(7, 5))


def animate_probs_linear(i):
    ax_gif.clear()

    idx = sequence_linear[i]
    T = Tlist_linear_reduced[idx]

    times = data_linear[T]["times"]
    p0 = data_linear[T]["p0"]
    p1 = data_linear[T]["p1"]
    ax_gif.plot(times, p0, ".-", linewidth=1, markersize=4.5, label="p0")
    ax_gif.plot(times, p1, ".-", linewidth=1, markersize=4.5, label="p1")

    ax_gif.set_title(f"Probabilities (T = {T})")
    ax_gif.set_xlabel(r"$t$")
    ax_gif.set_ylabel("Probabilities")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(
    fig_gif, animate_probs_linear, frames=len(sequence_linear), interval=1200
)
path = "/home/bsc/bsc504472/repos/Magic4Annealing/images/FrustatedRing/"
if not os.path.exists(path):
    os.makedirs(path)
ani.save(
    f"{path}Probabilites_linear.gif",
    writer="pillow",
    fps=0.3,
)


def animate_probs_LZR(i):
    ax_gif.clear()

    idx = sequence_LZR[i]
    T = Tlist_LZR_reduced[idx]

    times = data_LZR[T]["times"]
    p0 = data_LZR[T]["p0"]
    p1 = data_LZR[T]["p1"]
    ax_gif.plot(times, p0, ".-", linewidth=1, markersize=4.5, label="p0")
    ax_gif.plot(times, p1, ".-", linewidth=1, markersize=4.5, label="p1")

    ax_gif.set_title(f"Probabilities (T = {T})")
    ax_gif.set_xlabel(r"$t$")
    ax_gif.set_ylabel("Probabilities")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(
    fig_gif, animate_probs_LZR, frames=len(sequence_LZR), interval=1200
)
path = "/home/bsc/bsc504472/repos/Magic4Annealing/images/FrustatedRing/"
if not os.path.exists(path):
    os.makedirs(path)
ani.save(
    f"{path}Probabilites_LZR" + typeLZR + ".gif",
    writer="pillow",
    fps=0.3,
)


def animate_schedule_linear(i):
    ax_gif.clear()

    idx = sequence_linear[i]
    T = Tlist_linear_reduced[idx]

    times = data_linear[T]["times"]
    schedule = data_linear[T]["schedule"]
    ax_gif.plot(times, schedule, ".-", linewidth=1, markersize=4.5, label="Schedule")
    ax_gif.plot(
        times, 1 - schedule, ".-", linewidth=1, markersize=4.5, label="1 - Schedule"
    )

    ax_gif.set_title(f"Schedule (T = {T})")
    ax_gif.set_xlabel(r"$t$")
    ax_gif.set_ylabel("Schedule")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(
    fig_gif, animate_schedule_linear, frames=len(sequence_linear), interval=1200
)
path = "/home/bsc/bsc504472/repos/Magic4Annealing/images/FrustatedRing/"
if not os.path.exists(path):
    os.makedirs(path)
ani.save(
    f"{path}Schedule_linear.gif",
    writer="pillow",
    fps=0.3,
)


def animate_schedule_LZR(i):
    ax_gif.clear()

    idx = sequence_LZR[i]
    T = Tlist_LZR_reduced[idx]

    times = data_LZR[T]["times"]
    schedule = data_LZR[T]["schedule"]
    ax_gif.plot(times, schedule, ".-", linewidth=1, markersize=4.5, label="Schedule")
    ax_gif.plot(
        times, 1 - schedule, ".-", linewidth=1, markersize=4.5, label="1 - Schedule"
    )

    ax_gif.set_title(f"Schedule (T = {T})")
    ax_gif.set_xlabel(r"$t$")
    ax_gif.set_ylabel("Schedule")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(
    fig_gif, animate_schedule_LZR, frames=len(sequence_LZR), interval=1200
)
path = "/home/bsc/bsc504472/repos/Magic4Annealing/images/FrustatedRing/"
if not os.path.exists(path):
    os.makedirs(path)
ani.save(
    f"{path}Schedule_LZR" + typeLZR + ".gif",
    writer="pillow",
    fps=0.3,
)


def animate_energies_linear(i):
    ax_gif.clear()

    idx = sequence_linear[i]
    T = Tlist_linear_reduced[idx]

    times = data_linear[T]["times"]
    evo_energy = data_linear[T]["evo_energy"]
    e0 = data_linear[T]["e0"]
    e1 = data_linear[T]["e1"]

    ax_gif.plot(times, e0, "-", linewidth=1.5, markersize=4.5, label="E0")
    ax_gif.plot(times, e1, "-", linewidth=1.5, markersize=4.5, label="E1")
    ax_gif.plot(
        times,
        evo_energy,
        "--",
        linewidth=1.5,
        markersize=4.5,
        label="Evolution Energy",
        color="black",
    )
    ax_gif.set_title(f"Energy (T = {T})")
    ax_gif.set_xlabel(r"$t$")
    ax_gif.set_ylabel("Energy")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(
    fig_gif, animate_energies_linear, frames=len(sequence_linear), interval=1200
)
path = "/home/bsc/bsc504472/repos/Magic4Annealing/images/FrustatedRing/"
if not os.path.exists(path):
    os.makedirs(path)
ani.save(
    f"{path}Energies_linear.gif",
    writer="pillow",
    fps=0.3,
)


def animate_energies_LZR(i):
    ax_gif.clear()

    idx = sequence_LZR[i]
    T = Tlist_LZR_reduced[idx]

    times = data_LZR[T]["times"]
    evo_energy = data_LZR[T]["evo_energy"]
    e0 = data_LZR[T]["e0"]
    e1 = data_LZR[T]["e1"]

    ax_gif.plot(times, e0, "-", linewidth=1.5, markersize=4.5, label="E0")
    ax_gif.plot(times, e1, "-", linewidth=1.5, markersize=4.5, label="E1")
    ax_gif.plot(
        times,
        evo_energy,
        "--",
        linewidth=1.5,
        markersize=4.5,
        label="Evolution Energy",
        color="black",
    )
    ax_gif.set_title(f"Energy (T = {T})")
    ax_gif.set_xlabel(r"$t$")
    ax_gif.set_ylabel("Energy")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(
    fig_gif, animate_energies_LZR, frames=len(sequence_LZR), interval=1200
)
path = "/home/bsc/bsc504472/repos/Magic4Annealing/images/FrustatedRing/"
if not os.path.exists(path):
    os.makedirs(path)
ani.save(
    f"{path}Energies_LZR" + typeLZR + ".gif",
    writer="pillow",
    fps=0.3,
)
