import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse.linalg import expm_multiply


def final_energy_annealing(
    schedule, times, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
):
    """
    Runs the annealing process and returns only the final energy.
    schedule: array of the same length as `times`, giving s(t) at each step.
    """
    psi = psi0.copy()

    for i, t in enumerate(times):
        s = schedule[1][i]
        hamiltonian_t = (1 - s) * driver_hamiltonian_s + s * target_hamiltonian_s
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)

    # final energy: only need the Hamiltonian at the last time step
    final_energy = np.real(np.vdot(psi, hamiltonian_t @ psi))
    return final_energy


def max_magic_annealing(
    schedule, times, delta_t, psi0, sre, driver_hamiltonian_s, target_hamiltonian_s
):
    """
    Runs the annealing process and returns only the final energy.
    schedule: array of the same length as `times`, giving s(t) at each step.
    """
    psi = psi0.copy()
    max_magic = sre(psi)
    for i, t in enumerate(times):
        s = schedule[1][i]
        hamiltonian_t = (1 - s) * driver_hamiltonian_s + s * target_hamiltonian_s
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)
        magic = sre(psi)
        max_magic = max(max_magic, magic)

    return max_magic


def max_entanglement_annealing(
    schedule,
    times,
    delta_t,
    psi0,
    sector,
    entanglement_entropy,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Runs the annealing process and returns only the final energy.
    schedule: array of the same length as `times`, giving s(t) at each step.
    """

    psi = psi0.copy()
    psi_full = sector.lift(psi)
    max_entanglement = entanglement_entropy.von_neumann(psi_full)
    for i, t in enumerate(times):
        s = schedule[1][i]
        hamiltonian_t = (1 - s) * driver_hamiltonian_s + s * target_hamiltonian_s
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)
        psi_full = sector.lift(psi)
        entanglement = entanglement_entropy.von_neumann(psi_full)
        max_entanglement = max(max_entanglement, entanglement)

    return max_entanglement


def energy_fn(
    theta,
    build_schedule,
    times,
    delta_t,
    psi0,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> final energy.
    build_schedule: function that, given theta and times, returns the schedule array.
    """
    schedule = build_schedule(theta, times)
    return final_energy_annealing(
        schedule, times, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
    )


def max_magic_fn(
    theta,
    build_schedule,
    times,
    delta_t,
    psi0,
    sre,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> final energy.
    build_schedule: function that, given theta and times, returns the schedule array.
    """
    schedule = build_schedule(theta, times)
    return max_magic_annealing(
        schedule,
        times,
        delta_t,
        psi0,
        sre,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    )


def max_entanglement_fn(
    theta,
    build_schedule,
    times,
    delta_t,
    psi0,
    sector,
    entanglement_entropy,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> final energy.
    build_schedule: function that, given theta and times, returns the schedule array.
    """
    schedule = build_schedule(theta, times)
    return max_entanglement_annealing(
        schedule,
        times,
        delta_t,
        psi0,
        sector,
        entanglement_entropy,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    )


def build_plane_basis(theta1, theta2, theta3):
    """
    Given 3 points in R^n, returns:
    - origin (theta1)
    - orthonormal basis (e1, e2) of the plane passing through the 3 points
    - coordinates (a, b) of theta1, theta2, theta3 in that basis
    """
    v1 = theta2 - theta1
    v2 = theta3 - theta1

    # Gram-Schmidt
    e1 = v1 / np.linalg.norm(v1)
    v2_proj = v2 - np.dot(v2, e1) * e1
    norm_v2_proj = np.linalg.norm(v2_proj)
    if norm_v2_proj < 1e-10:
        raise ValueError("The three points are collinear: they do not span a plane.")
    e2 = v2_proj / norm_v2_proj

    # coordinates of the 3 points in the (a, b) basis
    coords = {
        "theta1": (0.0, 0.0),
        "theta2": (np.dot(v1, e1), np.dot(v1, e2)),
        "theta3": (np.dot(v2, e1), np.dot(v2, e2)),
    }

    return theta1, e1, e2, coords


def energy_landscape(theta1, theta2, theta3, energy_fn, resolution=30, margin=0.3):
    """
    energy_fn: function that takes a theta vector (1D, same size as theta1/2/3)
               and returns a scalar (the final energy).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    """
    origin, e1, e2, coords = build_plane_basis(theta1, theta2, theta3)

    # range of (a, b) to cover: bounding box of the triangle + margin
    as_ = [coords[k][0] for k in coords]
    bs_ = [coords[k][1] for k in coords]
    a_min, a_max = min(as_), max(as_)
    b_min, b_max = min(bs_), max(bs_)

    range_a = a_max - a_min
    range_b = b_max - b_min
    a_min -= margin * range_a
    a_max += margin * range_a
    b_min -= margin * range_b
    b_max += margin * range_b

    a_vals = np.linspace(a_min, a_max, resolution)
    b_vals = np.linspace(b_min, b_max, resolution)
    A, B = np.meshgrid(a_vals, b_vals)

    E = np.zeros_like(A)
    total = resolution * resolution
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            theta = origin + A[i, j] * e1 + B[i, j] * e2
            E[i, j] = energy_fn(theta)
            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{total}")

    return A, B, E, coords


def max_magic_landscape(
    theta1, theta2, theta3, max_magic_fn, resolution=30, margin=0.3
):
    """
    max_magic_fn: function that takes a theta vector (1D, same size as theta1/2/3)
                  and returns a scalar (the maximum magic).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    """
    origin, e1, e2, coords = build_plane_basis(theta1, theta2, theta3)

    # range of (a, b) to cover: bounding box of the triangle + margin
    as_ = [coords[k][0] for k in coords]
    bs_ = [coords[k][1] for k in coords]
    a_min, a_max = min(as_), max(as_)
    b_min, b_max = min(bs_), max(bs_)

    range_a = a_max - a_min
    range_b = b_max - b_min
    a_min -= margin * range_a
    a_max += margin * range_a
    b_min -= margin * range_b
    b_max += margin * range_b

    a_vals = np.linspace(a_min, a_max, resolution)
    b_vals = np.linspace(b_min, b_max, resolution)
    A, B = np.meshgrid(a_vals, b_vals)

    E = np.zeros_like(A)
    total = resolution * resolution
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            theta = origin + A[i, j] * e1 + B[i, j] * e2
            E[i, j] = max_magic_fn(theta)
            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{total}")

    return A, B, E, coords


def max_entanglement_landscape(
    theta1, theta2, theta3, max_entanglement_fn, resolution=30, margin=0.3
):
    """
    max_entanglement_fn: function that takes a theta vector (1D, same size as theta1/2/3)
                        and returns a scalar (the maximum entanglement).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    """
    origin, e1, e2, coords = build_plane_basis(theta1, theta2, theta3)

    # range of (a, b) to cover: bounding box of the triangle + margin
    as_ = [coords[k][0] for k in coords]
    bs_ = [coords[k][1] for k in coords]
    a_min, a_max = min(as_), max(as_)
    b_min, b_max = min(bs_), max(bs_)

    range_a = a_max - a_min
    range_b = b_max - b_min
    a_min -= margin * range_a
    a_max += margin * range_a
    b_min -= margin * range_b
    b_max += margin * range_b

    a_vals = np.linspace(a_min, a_max, resolution)
    b_vals = np.linspace(b_min, b_max, resolution)
    A, B = np.meshgrid(a_vals, b_vals)

    E = np.zeros_like(A)
    total = resolution * resolution
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            theta = origin + A[i, j] * e1 + B[i, j] * e2
            E[i, j] = max_entanglement_fn(theta)
            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{total}")

    return A, B, E, coords


def plot_energy_landscape(
    A, B, E, coords, energies, title="Energy landscape", save_path=None
):
    # rango real de datos
    rango_a = A.max() - A.min()
    rango_b = B.max() - B.min()
    ratio = rango_b / rango_a

    ancho_heatmap = 6  # pulgadas, tú decides el ancho
    alto_heatmap = ancho_heatmap * ratio

    alto_texto = 1.5  # pulgadas reservadas para la caja de texto abajo
    alto_total = alto_heatmap + alto_texto

    fig, ax = plt.subplots(
        figsize=(ancho_heatmap + 1, alto_total)
    )  # +1 por la colorbar

    cont = ax.contourf(A, B, E, levels=100, cmap="terrain")

    # colorbar con tamaño fijo relativo al heatmap, no a toda la figura
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% del ancho del heatmap
    plt.colorbar(cont, cax=cax, label="Final energy")

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
        "theta3": r"$\theta_3$",
    }

    for name, (a, b) in coords.items():
        ax.plot(a, b, "o", color="red", markersize=8)
        ax.annotate(
            nombres_latex[name],
            (a, b),
            textcoords="offset points",
            xytext=(6, 6),
            color="white",
            fontsize=12,
        )

    ax.set_xlabel("a (direction $e_1$)")
    ax.set_ylabel("b (direction $e_2$)")
    ax.set_title(title)
    ax.set_aspect("equal")

    # bottom como fracción del alto total
    fig.subplots_adjust(bottom=alto_texto / alto_total)

    texto_energias = "       ".join(
        [rf"{nombres_latex[name]}: $E$ = {energies[name]:.6f}" for name in coords]
    )
    fig.text(
        0.5,
        0.2,
        texto_energias,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"),
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    return fig


def plot_max_magic_landscape(
    A, B, max_magic, coords, energies, title="Max magic landscape", save_path=None
):
    # rango real de datos
    rango_a = A.max() - A.min()
    rango_b = B.max() - B.min()
    ratio = rango_b / rango_a

    ancho_heatmap = 6  # pulgadas, tú decides el ancho
    alto_heatmap = ancho_heatmap * ratio

    alto_texto = 1.5  # pulgadas reservadas para la caja de texto abajo
    alto_total = alto_heatmap + alto_texto

    fig, ax = plt.subplots(
        figsize=(ancho_heatmap + 1, alto_total)
    )  # +1 por la colorbar

    cont = ax.contourf(A, B, max_magic, levels=100, cmap="terrain")

    # colorbar con tamaño fijo relativo al heatmap, no a toda la figura
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% del ancho del heatmap
    plt.colorbar(cont, cax=cax, label="Maximum magic")

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
        "theta3": r"$\theta_3$",
    }

    for name, (a, b) in coords.items():
        ax.plot(a, b, "o", color="red", markersize=8)
        ax.annotate(
            nombres_latex[name],
            (a, b),
            textcoords="offset points",
            xytext=(6, 6),
            color="white",
            fontsize=12,
        )

    ax.set_xlabel("a (direction $e_1$)")
    ax.set_ylabel("b (direction $e_2$)")
    ax.set_title(title)
    ax.set_aspect("equal")

    # bottom como fracción del alto total
    fig.subplots_adjust(bottom=alto_texto / alto_total)

    texto_energias = "       ".join(
        [rf"{nombres_latex[name]}: $E$ = {energies[name]:.6f}" for name in coords]
    )
    fig.text(
        0.5,
        0.2,
        texto_energias,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"),
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    return fig


def plot_max_entanglement_landscape(
    A,
    B,
    max_entanglement,
    coords,
    energies,
    title="Max entanglement landscape",
    save_path=None,
):
    # rango real de datos
    rango_a = A.max() - A.min()
    rango_b = B.max() - B.min()
    ratio = rango_b / rango_a

    ancho_heatmap = 6  # pulgadas, tú decides el ancho
    alto_heatmap = ancho_heatmap * ratio

    alto_texto = 1.5  # pulgadas reservadas para la caja de texto abajo
    alto_total = alto_heatmap + alto_texto

    fig, ax = plt.subplots(
        figsize=(ancho_heatmap + 1, alto_total)
    )  # +1 por la colorbar

    cont = ax.contourf(A, B, max_entanglement, levels=100, cmap="terrain")

    # colorbar con tamaño fijo relativo al heatmap, no a toda la figura
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% del ancho del heatmap
    plt.colorbar(cont, cax=cax, label="Maximum entanglement")

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
        "theta3": r"$\theta_3$",
    }

    for name, (a, b) in coords.items():
        ax.plot(a, b, "o", color="red", markersize=8)
        ax.annotate(
            nombres_latex[name],
            (a, b),
            textcoords="offset points",
            xytext=(6, 6),
            color="white",
            fontsize=12,
        )

    ax.set_xlabel("a (direction $e_1$)")
    ax.set_ylabel("b (direction $e_2$)")
    ax.set_title(title)
    ax.set_aspect("equal")

    # bottom como fracción del alto total
    fig.subplots_adjust(bottom=alto_texto / alto_total)

    texto_energias = "       ".join(
        [rf"{nombres_latex[name]}: $E$ = {energies[name]:.6f}" for name in coords]
    )
    fig.text(
        0.5,
        0.2,
        texto_energias,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"),
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    return fig
