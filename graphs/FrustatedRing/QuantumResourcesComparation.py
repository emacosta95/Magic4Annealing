import re

import matplotlib.pyplot as plt
import numpy as np


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

        Ti = int(match.group(1))
        nombre_variable = match.group(2)

        if Ti not in resultado:
            resultado[Ti] = {}

        resultado[Ti][nombre_variable] = data[clave]

    return resultado


graph_type = "probs"  # "max_entang", "max_magic", "final_energy", "entang_int", "magic_int", "min_gap", "probs", "schedules", "spectrum", "entang_evo", "magic_evo"
graph_type2 = "magic_evo"
graph_type3 = "entang_evo"

nlevels = 5  # number of energy levels to plot in the spectrum

T = 120
T2 = T

tag = "_bounded_SA"  # "_NoGrad", "_step_test", "_bounded", "_no_random" or ""
tag_T = ""  # "_more_T" or ""
N = 7

tag2 = "_bounded"  # "_NoGrad", "_step_test", "_bounded", "_no_random" or ""
tag_T2 = ""
N2 = N

data_linear = load_data(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_linear" + tag_T + ".npz"
)
data_LZR = load_data(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR"
    + tag
    + tag_T
    + ".npz"
)
data_linear2 = load_data(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N2}_linear" + tag_T2 + ".npz"
)
data_LZR2 = load_data(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N2}_LZR"
    + tag2
    + tag_T2
    + ".npz"
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
for Ti in Tlist_linear:
    min_gap_linear.append(np.min(data_linear[Ti]["gap"]))
    max_entanglement_linear.append(np.max(data_linear[Ti]["entanglement"]))
    max_magic_linear.append(np.max(data_linear[Ti]["magic"]))
    max_entanglement_gs_level.append(np.max(data_linear[Ti]["entanglement_gs_level"]))
    max_magic_gs_level.append(np.max(data_linear[Ti]["magic_gs_level"]))
    entanglement_integral_linear.append(
        (1 / Ti)
        * np.trapezoid(data_linear[Ti]["entanglement"], x=data_linear[Ti]["time_sub"])
    )
    entanglement_integral_gs_level.append(
        (1 / Ti)
        * np.trapezoid(
            data_linear[Ti]["entanglement_gs_level"], x=data_linear[Ti]["time_sub"]
        )
    )
    magic_integral_linear.append(
        (1 / Ti) * np.trapezoid(data_linear[Ti]["magic"], x=data_linear[Ti]["time_sub"])
    )
    magic_integral_gs_level.append(
        (1 / Ti)
        * np.trapezoid(data_linear[Ti]["magic_gs_level"], x=data_linear[Ti]["time_sub"])
    )


min_gap_LZR = []
max_entanglement_LZR = []
max_magic_LZR = []
entanglement_integral_LZR = []
magic_integral_LZR = []
for Ti in Tlist_LZR:
    min_gap_LZR.append(np.min(data_LZR[Ti]["gap"]))
    max_entanglement_LZR.append(np.max(data_LZR[Ti]["entanglement"]))
    max_magic_LZR.append(np.max(data_LZR[Ti]["magic"]))
    entanglement_integral_LZR.append(
        (1 / Ti)
        * np.trapezoid(data_LZR[Ti]["entanglement"], x=data_LZR[Ti]["time_sub"])
    )
    magic_integral_LZR.append(
        (1 / Ti) * np.trapezoid(data_LZR[Ti]["magic"], x=data_LZR[Ti]["time_sub"])
    )

Tlist_linear2 = sorted(data_linear2.keys())
Tlist_LZR2 = sorted(data_LZR2.keys())

min_gap_linear2 = []
max_entanglement_linear2 = []
max_magic_linear2 = []
max_entanglement_gs_level2 = []
max_magic_gs_level2 = []
entanglement_integral_linear2 = []
entanglement_integral_gs_level2 = []
magic_integral_linear2 = []
magic_integral_gs_level2 = []
for Ti in Tlist_linear2:
    min_gap_linear2.append(np.min(data_linear2[Ti]["gap"]))
    max_entanglement_linear2.append(np.max(data_linear2[Ti]["entanglement"]))
    max_magic_linear2.append(np.max(data_linear2[Ti]["magic"]))
    max_entanglement_gs_level2.append(np.max(data_linear2[Ti]["entanglement_gs_level"]))
    max_magic_gs_level2.append(np.max(data_linear2[Ti]["magic_gs_level"]))
    entanglement_integral_linear2.append(
        (1 / Ti)
        * np.trapezoid(data_linear2[Ti]["entanglement"], x=data_linear2[Ti]["time_sub"])
    )
    entanglement_integral_gs_level2.append(
        (1 / Ti)
        * np.trapezoid(
            data_linear2[Ti]["entanglement_gs_level"], x=data_linear2[Ti]["time_sub"]
        )
    )
    magic_integral_linear2.append(
        (1 / Ti)
        * np.trapezoid(data_linear2[Ti]["magic"], x=data_linear2[Ti]["time_sub"])
    )
    magic_integral_gs_level2.append(
        (1 / Ti)
        * np.trapezoid(
            data_linear2[Ti]["magic_gs_level"], x=data_linear2[Ti]["time_sub"]
        )
    )


min_gap_LZR2 = []
max_entanglement_LZR2 = []
max_magic_LZR2 = []
entanglement_integral_LZR2 = []
magic_integral_LZR2 = []
for Ti in Tlist_LZR2:
    min_gap_LZR2.append(np.min(data_LZR2[Ti]["gap"]))
    max_entanglement_LZR2.append(np.max(data_LZR2[Ti]["entanglement"]))
    max_magic_LZR2.append(np.max(data_LZR2[Ti]["magic"]))
    entanglement_integral_LZR2.append(
        (1 / Ti)
        * np.trapezoid(data_LZR2[Ti]["entanglement"], x=data_LZR2[Ti]["time_sub"])
    )
    magic_integral_LZR2.append(
        (1 / Ti) * np.trapezoid(data_LZR2[Ti]["magic"], x=data_LZR2[Ti]["time_sub"])
    )

final_energies = [data_LZR[Ti]["evo_energy"][-1] for Ti in Tlist_LZR]
final_energies2 = [data_LZR2[Ti]["evo_energy"][-1] for Ti in Tlist_LZR2]


def comparation_plot(graph_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if graph_type == "min_gap":
        ax1.plot(
            Tlist_linear,
            min_gap_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax1.plot(
            Tlist_LZR,
            min_gap_LZR,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax1.set_xlabel("T")
        ax1.set_ylabel("Minimum Gap")
        ax1.legend()

        ax2.plot(
            Tlist_linear2,
            min_gap_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax2.plot(
            Tlist_LZR2,
            min_gap_LZR2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax2.set_xlabel("T")
        ax2.legend()
        fig.suptitle("Minimum Gap vs T")
    elif graph_type == "final_energy":
        ax1.plot(
            Tlist_LZR,
            final_energies,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax1.set_xlabel("T")
        ax1.set_ylabel("Energy at final time")
        ax1.legend()

        ax2.plot(
            Tlist_LZR2,
            final_energies2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax2.set_xlabel("T")
        ax2.legend()
        fig.suptitle("Final Energy vs T")
    elif graph_type == "max_entang":
        ax1.plot(
            Tlist_linear,
            max_entanglement_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax1.plot(
            Tlist_linear,
            max_entanglement_gs_level,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax1.plot(
            Tlist_LZR,
            max_entanglement_LZR,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax1.set_xlabel("T")
        ax1.set_ylabel("Maximum Entanglement")
        ax1.legend()

        ax2.plot(
            Tlist_linear2,
            max_entanglement_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax2.plot(
            Tlist_linear2,
            max_entanglement_gs_level2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax2.plot(
            Tlist_LZR2,
            max_entanglement_LZR2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax2.set_xlabel("T")
        ax2.legend()
        fig.suptitle("Maximum Entanglement vs T")
    elif graph_type == "max_magic":
        ax1.plot(
            Tlist_linear,
            max_magic_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax1.plot(
            Tlist_linear,
            max_magic_gs_level,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax1.plot(
            Tlist_LZR,
            max_magic_LZR,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax1.set_xlabel("T")
        ax1.set_ylabel("Maximum Magic")
        ax1.legend()

        ax2.plot(
            Tlist_linear2,
            max_magic_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax2.plot(
            Tlist_linear2,
            max_magic_gs_level2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax2.plot(
            Tlist_LZR2,
            max_magic_LZR2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax2.set_xlabel("T")
        ax2.legend()
        fig.suptitle("Maximum Magic vs T")
    elif graph_type == "entang_int":
        ax1.plot(
            Tlist_linear,
            entanglement_integral_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax1.plot(
            Tlist_linear,
            entanglement_integral_gs_level,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax1.plot(
            Tlist_LZR,
            entanglement_integral_LZR,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax1.set_xlabel("T")
        ax1.set_ylabel("Entanglement Integral")
        ax1.legend()

        ax2.plot(
            Tlist_linear2,
            entanglement_integral_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax2.plot(
            Tlist_linear2,
            entanglement_integral_gs_level2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax2.plot(
            Tlist_LZR2,
            entanglement_integral_LZR2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax2.set_xlabel("T")
        ax2.legend()
        fig.suptitle("Entanglement Integral vs T")
    elif graph_type == "magic_int":
        ax1.plot(
            Tlist_linear,
            magic_integral_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax1.plot(
            Tlist_linear,
            magic_integral_gs_level,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax1.plot(
            Tlist_LZR,
            magic_integral_LZR,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax1.set_xlabel("T")
        ax1.set_ylabel("Magic Integral")
        ax1.legend()

        ax2.plot(
            Tlist_linear2,
            magic_integral_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule",
        )
        ax2.plot(
            Tlist_linear2,
            magic_integral_gs_level2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear schedule (ground state level)",
        )
        ax2.plot(
            Tlist_LZR2,
            magic_integral_LZR2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR schedule",
        )
        ax2.set_xlabel("T")
        ax2.legend()
        fig.suptitle("Magic Integral vs T")
    elif graph_type == "probs":
        times = data_LZR[T]["times"]
        for i in range(nlevels):
            ax1.plot(
                times,
                data_LZR[T][f"p{i}"],
                ".-",
                linewidth=1,
                markersize=4.5,
                label=f"p{i}",
            )

        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel("Probabilities")
        ax1.legend()
        ax1.grid()

        times2 = data_LZR2[T2]["times"]
        for i in range(nlevels):
            ax2.plot(
                times2,
                data_LZR2[T2][f"p{i}"],
                ".-",
                linewidth=1,
                markersize=4.5,
                label=f"p{i}",
            )
        ax2.set_xlabel(r"$t$")
        ax2.legend()
        ax2.grid()
        fig.suptitle("Probabilities vs Time")
    elif graph_type == "schedules":
        ax1.plot(
            data_LZR[T]["times"],
            data_LZR[T]["schedule"],
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Schedule",
        )
        ax1.plot(
            data_LZR[T]["times"],
            1 - data_LZR[T]["schedule"],
            ".-",
            linewidth=1,
            markersize=4.5,
            label="1 - Schedule",
        )
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel("Schedule")
        ax1.legend()
        ax1.grid()

        ax2.plot(
            data_LZR2[T2]["times"],
            data_LZR2[T2]["schedule"],
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Schedule",
        )
        ax2.plot(
            data_LZR2[T2]["times"],
            1 - data_LZR2[T2]["schedule"],
            ".-",
            linewidth=1,
            markersize=4.5,
            label="1 - Schedule",
        )
        ax2.set_xlabel(r"$t$")
        ax2.legend()
        ax2.grid()
        fig.suptitle("Schedules vs Time")
    elif graph_type == "spectrum":
        times = data_LZR[T]["times"]
        for i in range(nlevels):
            ax1.plot(
                times,
                data_LZR[T][f"e{i}"],
                ".-",
                linewidth=1,
                markersize=4.5,
                label=f"E{i}",
            )
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel("Energy")
        ax1.legend()
        ax1.grid()

        times2 = data_LZR2[T2]["times"]
        for i in range(nlevels):
            ax2.plot(
                times2,
                data_LZR2[T2][f"e{i}"],
                ".-",
                linewidth=1,
                markersize=4.5,
                label=f"E{i}",
            )
        ax2.set_xlabel(r"$t$")
        ax2.legend()
        ax2.grid()
        fig.suptitle("Energy Spectrum vs Time")
    elif graph_type == "entang_evo":
        times_linear = data_linear[T]["time_sub"]
        entanglement_linear = data_linear[T]["entanglement"]
        entanglement_linear_gs = data_linear[T]["entanglement_gs_level"]
        times_LZR = data_LZR[T]["time_sub"]
        entanglement_LZR = data_LZR[T]["entanglement"]
        ax1.plot(
            times_linear,
            entanglement_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear",
        )
        ax1.plot(
            times_linear,
            entanglement_linear_gs,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear (ground state level)",
        )
        ax1.plot(
            times_LZR, entanglement_LZR, ".-", linewidth=1, markersize=4.5, label="LZR"
        )
        ax1.legend()
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel("Entanglement")
        ax1.grid()

        times_linear2 = data_linear2[T2]["time_sub"]
        entanglement_linear2 = data_linear2[T2]["entanglement"]
        entanglement_linear_gs2 = data_linear2[T2]["entanglement_gs_level"]
        times_LZR2 = data_LZR2[T2]["time_sub"]
        entanglement_LZR2 = data_LZR2[T2]["entanglement"]
        ax2.plot(
            times_linear2,
            entanglement_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear",
        )
        ax2.plot(
            times_linear2,
            entanglement_linear_gs2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear (ground state level)",
        )
        ax2.plot(
            times_LZR2,
            entanglement_LZR2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="LZR",
        )
        ax2.legend()
        ax2.set_xlabel(r"$t$")
        ax2.grid()
        fig.suptitle("Entanglement Evolution vs Time")
    elif graph_type == "magic_evo":
        times_linear = data_linear[T]["time_sub"]
        magic_linear = data_linear[T]["magic"]
        magic_linear_gs = data_linear[T]["magic_gs_level"]
        times_LZR = data_LZR[T]["time_sub"]
        magic_LZR = data_LZR[T]["magic"]
        ax1.plot(
            times_linear,
            magic_linear,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear",
        )
        ax1.plot(
            times_linear,
            magic_linear_gs,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear (ground state level)",
        )
        ax1.plot(times_LZR, magic_LZR, ".-", linewidth=1, markersize=4.5, label="LZR")
        ax1.legend()
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel("Magic")
        ax1.grid()

        times_linear2 = data_linear2[T2]["time_sub"]
        magic_linear2 = data_linear2[T2]["magic"]
        magic_linear_gs2 = data_linear2[T2]["magic_gs_level"]
        times_LZR2 = data_LZR2[T2]["time_sub"]
        magic_LZR2 = data_LZR2[T2]["magic"]
        ax2.plot(
            times_linear2,
            magic_linear2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear",
        )
        ax2.plot(
            times_linear2,
            magic_linear_gs2,
            ".-",
            linewidth=1,
            markersize=4.5,
            label="Linear (ground state level)",
        )
        ax2.plot(times_LZR2, magic_LZR2, ".-", linewidth=1, markersize=4.5, label="LZR")
        ax2.legend()
        ax2.set_xlabel(r"$t$")
        ax2.grid()
        fig.suptitle("Magic Evolution vs Time")

    if graph_type in [
        "min_gap",
        "final_energy",
        "max_entanglement",
        "max_magic",
        "entanglement_integral",
        "magic_integral",
    ]:
        ax1.set_title(f"Frustrated Ring file={tag+tag_T}, N={N}")
        ax2.set_title(f"Frustrated Ring file={tag2+tag_T2}, N={N2}")
    elif graph_type in [
        "probabilities",
        "schedules",
        "energy_spectrum",
        "entanglement_evolution",
        "magic_evolution",
    ]:
        ax1.set_title(f"Frustrated Ring file={tag+tag_T}, N={N}, T={T}")
        ax2.set_title(f"Frustrated Ring file={tag2+tag_T2}, N={N2}, T={T2}")

    plt.tight_layout()
    plt.show()


comparation_plot(graph_type)
if graph_type2 != "":
    comparation_plot(graph_type2)
if graph_type3 != "":
    comparation_plot(graph_type3)
