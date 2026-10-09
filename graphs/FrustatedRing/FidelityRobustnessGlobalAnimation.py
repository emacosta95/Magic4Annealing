import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.fidelity_robustness import AXES


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


tag = "_FF_bounded"  # "_NoGrad", "_step_test", "_bounded", "_no_random" or ""
tag_T = ""  # "_more_T" or ""
N = 7
T_step = 1  # one frame every T_step values of T in the file

# "exact": the evolution re-run for every lambda (saved on a grid of lambda)
# "second_order": 1 - lam^T Re(G) lam, relative to the unperturbed final state
method = "exact"
# what the exact fidelity is measured against (see src/fidelity_robustness.py):
# "ground_space" (population in the ground space of H_P), "ground_symmetric"
# (its Z2 +1 ground state) or "final_state" (the unperturbed final state)
reference = "ground_space"

# plotted range of the global field lambda, V = lambda * sum_i sigma^a_i, for
# each axis a ("exact" only has the values of lambda saved by the build script)
lambda_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)}
n_lambdas = 1001  # points of the second-order parabola

duration = 30  # seconds of video; the frame rate is (number of T) / duration

# mp4 needs ffmpeg: the one on PATH, or the binary shipped by imageio-ffmpeg
if shutil.which("ffmpeg") is None:
    import imageio_ffmpeg

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# exact fidelities and robustness matrices G of the LZR and linear schedules,
# from build/FrustatedRing/run_fidelity_robustness.py + merge_fidelity_robustness.py
data = load_data(
    f"../../generated/FrustatedRing/FidelityRobustnessvsT_N={N}_LZR"
    + tag
    + tag_T
    + ".npz"
)
Tlist = sorted(data.keys())[::T_step]
fps = len(Tlist) / duration

nqubits = N
curves = {"LZR schedule": "", "Linear schedule": "_linear"}  # key suffixes
ylabels = {
    "ground_space": "Ground-space population",
    "ground_symmetric": "Fidelity with the symmetric ground state",
    "final_state": "Fidelity with the unperturbed final state",
}

# the default choice keeps the plain name; the others are told apart
if method == "second_order":
    variant = "_second_order"
elif reference == "ground_space":
    variant = ""
else:
    variant = f"_{reference}"
name_tag = (tag + tag_T).lstrip("_")
filename = (name_tag + "_" if name_tag else "") + f"N={N}{variant}.mp4"

for a, axis in enumerate(AXES):
    # lam = [x_0..x_{N-1}, y_0..y_{N-1}, z_0..z_{N-1}]: lambda on the N
    # entries of this axis, 0 on the other 2N
    direction = np.zeros(3 * nqubits)
    direction[a * nqubits : (a + 1) * nqubits] = 1.0

    lambda_min, lambda_max = lambda_range[axis]
    lambdas_parabola = np.linspace(lambda_min, lambda_max, n_lambdas)

    fig, ax = plt.subplots(figsize=(7, 5.6))
    fig.subplots_adjust(bottom=0.25)

    def animate(i):
        ax.clear()
        T = Tlist[i]
        for k, (name, suffix) in enumerate(curves.items()):
            if method == "exact":
                lambdas = data[T]["lambdas"][a]
                fidelity = data[T][f"fidelity_{reference}{suffix}"][a]
                label = rf"{name}, $F(0)$ = {np.interp(0.0, lambdas, fidelity):.3f}"
            else:
                # each fidelity is relative to the unperturbed final state of
                # its own schedule; second order: 1 - lam^T Re(G) lam
                g_axis = direction @ data[T]["G" + suffix].real @ direction
                lambda_star = 1 / np.sqrt(g_axis) if g_axis > 0 else np.inf
                lambdas = lambdas_parabola
                fidelity = 1 - g_axis * lambdas**2
                label = rf"{name}, $\lambda^*$ = {lambda_star:.2g}"
                # band where the second order is reliable, in the color of
                # its curve
                if np.isfinite(lambda_star):
                    ax.axvspan(
                        max(-lambda_star / 3, lambda_min),
                        min(lambda_star / 3, lambda_max),
                        color=f"C{k}",
                        alpha=0.2,
                    )

            ax.plot(
                lambdas, fidelity, "-", linewidth=1.5, color=f"C{k}", label=label
            )
        # the second-order fidelity is not bounded below: the axis stops at 0
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(lambda_min, lambda_max)
        ax.set_xlabel(r"$\lambda$")
        if method == "exact":
            ax.set_ylabel(ylabels[reference])
        else:
            ax.set_ylabel("Fidelity (second order)")
        ax.set_title(
            rf"$V = \lambda \sum_i \sigma^{axis}_i$,  N={N}, T={T}, LZR{tag + tag_T}"
        )
        ax.legend(
            title=(
                None
                if method == "exact"
                else r"shaded: $|\lambda| \leq \lambda^*/3$"
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
        )
        ax.grid()

    ani = animation.FuncAnimation(fig, animate, frames=len(Tlist))

    path = f"../../images/FrustatedRing/FidelityRobustnessGlobal{axis.upper()}/"
    if not os.path.exists(path):
        os.makedirs(path)
    ani.save(f"{path}{filename}", writer=animation.FFMpegWriter(fps=fps), dpi=150)
    plt.close(fig)
    print(f"Guardado: {path}{filename}")
