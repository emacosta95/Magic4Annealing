"""
Landscapes of the LZR schedule of the frustrated ring: observables as a
function of theta on a plane through 3 points (or a line through 2) of
parameter space. Backend of graphs/FrustatedRing/LZR*Landscape.py.

  - RingLandscape: theta -> final energy, max entanglement, max magic,
    robustness (free fermions; spin basis only where needed).
  - Geometry/scan: build_plane_basis, plane_grid, scan_plane (parallel),
    bounds_region (part of the plane inside the optimization bounds).
  - Plots: plot_landscape and the plot_*_landscape(_1d) wrappers.
  - Older spin-basis function API (energy_fn, max_magic_fn, ...): only
    used by InterpolationGRAPEvsSA.py.
"""

import zlib

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.annealing_utils import get_driver_hamiltonian, get_longitudinal_hamiltonian
from src.fidelity_robustness import FidelityRobustness, global_pauli_operators
from src.free_fermions_grape_method import NambuGRAPEModel
from src.free_fermions_utils import NambuIsing1D
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.parallel import make_pool
from src.time_grid import (
    make_time_grids,
    midpoint_evolution,
    nambu_midpoint_evolution,
    subsample_indices,
)
from src.utils import Z2SymmetricSector

# Time discretization: midpoint rule (src/time_grid.py). In every function
# below `schedule_ctrl` is (h_driver, h_target) evaluated on the CONTROL grid
# (cell midpoints, one value per propagation step) and `schedule` the same
# pair on the STATE grid `times` (len(times) = nsteps + 1, times[-1] = tf).
# Only schedule[1] = s(t) is used: H(t) = (1 - s) H_driver + s H_target.


def _states(schedule_ctrl, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s):
    """States psi(times[k]), k = 0..nsteps, of the annealing driven by schedule_ctrl."""
    s_ctrl = np.asarray(schedule_ctrl[1])
    for _, psi in midpoint_evolution(
        psi0, 1 - s_ctrl, s_ctrl, delta_t, driver_hamiltonian_s, target_hamiltonian_s
    ):
        yield psi


def final_energy_annealing(
    schedule_ctrl,
    schedule,
    delta_t,
    psi0,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Runs the annealing process and returns only the final energy.
    schedule_ctrl: s(t) at the cell midpoints, drives the evolution.
    schedule: s(t) on the state grid; only its last value s(tf) is used, to
              build the Hamiltonian the final energy is measured with.
    """
    for psi in _states(
        schedule_ctrl, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
    ):
        pass

    # final energy: only need the Hamiltonian at t = tf
    s = schedule[1][-1]
    hamiltonian_t = (1 - s) * driver_hamiltonian_s + s * target_hamiltonian_s
    final_energy = np.real(np.vdot(psi, hamiltonian_t @ psi))
    return final_energy


def max_magic_annealing(
    schedule_ctrl, delta_t, psi0, sre, driver_hamiltonian_s, target_hamiltonian_s
):
    """
    Runs the annealing process and returns the maximum magic over the states
    psi(times[k]), k = 0..nsteps.
    schedule_ctrl: s(t) at the cell midpoints, drives the evolution.
    """
    max_magic = -np.inf
    for psi in _states(
        schedule_ctrl, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
    ):
        max_magic = max(max_magic, sre(psi))

    return max_magic


def max_entanglement_annealing(
    schedule_ctrl,
    delta_t,
    psi0,
    sector,
    entanglement_entropy,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Runs the annealing process and returns the maximum entanglement over the
    states psi(times[k]), k = 0..nsteps.
    schedule_ctrl: s(t) at the cell midpoints, drives the evolution.
    """
    max_entanglement = -np.inf
    for psi in _states(
        schedule_ctrl, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
    ):
        psi_full = sector.lift(psi)
        entanglement = entanglement_entropy.von_neumann(psi_full)
        max_entanglement = max(max_entanglement, entanglement)

    return max_entanglement


def energy_fn(
    theta,
    build_schedule,
    times,
    times_ctrl,
    delta_t,
    psi0,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> final energy.
    build_schedule: function that, given theta and an array of times, returns
                    the schedule (h_driver, h_target) on those times.
    times, times_ctrl, delta_t: as returned by src.time_grid.make_time_grids.
    """
    schedule_ctrl = build_schedule(theta, times_ctrl)
    schedule = build_schedule(theta, times)
    return final_energy_annealing(
        schedule_ctrl,
        schedule,
        delta_t,
        psi0,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    )


def max_magic_fn(
    theta,
    build_schedule,
    times,
    times_ctrl,
    delta_t,
    psi0,
    sre,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> maximum magic.
    Same arguments as energy_fn (`times` is unused: the magic does not need
    the instantaneous Hamiltonian; kept so the three *_fn share a signature).
    """
    schedule_ctrl = build_schedule(theta, times_ctrl)
    return max_magic_annealing(
        schedule_ctrl,
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
    times_ctrl,
    delta_t,
    psi0,
    sector,
    entanglement_entropy,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> maximum entanglement.
    Same arguments as energy_fn (`times` is unused, see max_magic_fn).
    """
    schedule_ctrl = build_schedule(theta, times_ctrl)
    return max_entanglement_annealing(
        schedule_ctrl,
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


def build_line_basis(theta1, theta2):
    """
    Given 2 points in R^n, returns:
    - origin (theta1)
    - unit vector e1 along the line from theta1 to theta2
    - coordinate (a) of theta1 and theta2 in that basis
    """
    v1 = theta2 - theta1
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 < 1e-10:
        raise ValueError("The two points are identical: they do not span a line.")
    e1 = v1 / norm_v1

    coords = {
        "theta1": 0.0,
        "theta2": norm_v1,
    }

    return theta1, e1, coords


def plane_grid(theta1, theta2, theta3, resolution=30, margin=0.3):
    """
    Grid of the plane through theta1, theta2, theta3: the bounding box of the
    triangle in the (a, b) coordinates of build_plane_basis, widened by
    `margin` (fraction of its size) on every side.

    Returns (origin, e1, e2, coords, A, B), A and B (resolution, resolution)
    from np.meshgrid: point (i, j) is origin + A[i, j] e1 + B[i, j] e2.
    """
    origin, e1, e2, coords = build_plane_basis(theta1, theta2, theta3)

    # range of (a, b) to cover: bounding box of the triangle + margin
    ab = np.array(list(coords.values()))
    lo, hi = ab.min(axis=0), ab.max(axis=0)
    lo, hi = lo - margin * (hi - lo), hi + margin * (hi - lo)

    A, B = np.meshgrid(
        np.linspace(lo[0], hi[0], resolution), np.linspace(lo[1], hi[1], resolution)
    )
    return origin, e1, e2, coords, A, B


def scan_plane(theta1, theta2, theta3, fn, resolution=30, margin=0.3, n_workers=1):
    """
    fn(theta) on every point of plane_grid.

    fn returns a scalar, or a tuple of K scalars (several observables of the
    same evolution): Z is then (resolution, resolution) or
    (resolution, resolution, K).

    n_workers > 1 spreads the points over that many processes
    (src/parallel.py). fn must then be a module-level function of the calling
    script, and the call must be under `if __name__ == "__main__":` (the
    workers import the script again).

    Returns (A, B, Z, coords).
    """
    origin, e1, e2, coords, A, B = plane_grid(
        theta1, theta2, theta3, resolution, margin
    )
    thetas = [origin + a * e1 + b * e2 for a, b in zip(A.ravel(), B.ravel())]
    total = len(thetas)
    every = max(1, total // 100)

    values = []
    if n_workers > 1:
        with make_pool(n_workers, blas_threads=1) as pool:
            # map returns the results in the order of the points
            chunksize = max(1, total // (4 * n_workers))
            for value in pool.map(fn, thetas, chunksize=chunksize):
                values.append(value)
                if len(values) % every == 0:
                    print(f"Progress: {len(values)}/{total}", flush=True)
    else:
        for theta in thetas:
            values.append(fn(theta))
            if len(values) % every == 0:
                print(f"Progress: {len(values)}/{total}", flush=True)

    Z = np.array(values, dtype=float)
    return A, B, Z.reshape(A.shape + Z.shape[1:]), coords


def energy_landscape(
    theta1, theta2, theta3, energy_fn, resolution=30, margin=0.3, n_workers=1
):
    """
    energy_fn: function that takes a theta vector (1D, same size as theta1/2/3)
               and returns a scalar (the final energy).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    n_workers: processes the points are spread over (see scan_plane).
    """
    return scan_plane(
        theta1, theta2, theta3, energy_fn, resolution, margin, n_workers
    )


def energy_landscape_1d(theta1, theta2, energy_fn, resolution=30, margin=0.3):
    """
    energy_fn: function that takes a theta vector (1D, same size as theta1/2)
               and returns a scalar (the final energy).
    resolution: number of points along the line.
    margin: extra fraction of space around the segment formed by the 2 points.
    """
    origin, e1, coords = build_line_basis(theta1, theta2)

    a_min, a_max = coords["theta1"], coords["theta2"]
    if a_min > a_max:
        a_min, a_max = a_max, a_min

    range_a = a_max - a_min
    a_min -= margin * range_a
    a_max += margin * range_a

    a_vals = np.linspace(a_min, a_max, resolution)

    E = np.zeros_like(a_vals)
    total = resolution
    for i in range(resolution):
        theta = origin + a_vals[i] * e1
        E[i] = energy_fn(theta)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{total}")

    return a_vals, E, coords


def max_magic_landscape(
    theta1, theta2, theta3, max_magic_fn, resolution=30, margin=0.3, n_workers=1
):
    """
    max_magic_fn: function that takes a theta vector (1D, same size as theta1/2/3)
                  and returns a scalar (the maximum magic).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    n_workers: processes the points are spread over (see scan_plane).
    """
    return scan_plane(
        theta1, theta2, theta3, max_magic_fn, resolution, margin, n_workers
    )


def max_magic_landscape_1d(theta1, theta2, max_magic_fn, resolution=30, margin=0.3):
    """
    max_magic_fn: function that takes a theta vector (1D, same size as theta1/2)
                  and returns a scalar (the maximum magic).
    resolution: number of points along the line.
    margin: extra fraction of space around the segment formed by the 2 points.
    """
    origin, e1, coords = build_line_basis(theta1, theta2)

    a_min, a_max = coords["theta1"], coords["theta2"]
    if a_min > a_max:
        a_min, a_max = a_max, a_min

    range_a = a_max - a_min
    a_min -= margin * range_a
    a_max += margin * range_a

    a_vals = np.linspace(a_min, a_max, resolution)

    E = np.zeros_like(a_vals)
    total = resolution
    for i in range(resolution):
        theta = origin + a_vals[i] * e1
        E[i] = max_magic_fn(theta)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{total}")

    return a_vals, E, coords


def max_entanglement_landscape(
    theta1,
    theta2,
    theta3,
    max_entanglement_fn,
    resolution=30,
    margin=0.3,
    n_workers=1,
):
    """
    max_entanglement_fn: function that takes a theta vector (1D, same size as theta1/2/3)
                        and returns a scalar (the maximum entanglement).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    n_workers: processes the points are spread over (see scan_plane).
    """
    return scan_plane(
        theta1, theta2, theta3, max_entanglement_fn, resolution, margin, n_workers
    )


def max_entanglement_landscape_1d(
    theta1, theta2, max_entanglement_fn, resolution=30, margin=0.3
):
    """
    max_entanglement_fn: function that takes a theta vector (1D, same size as theta1/2)
                        and returns a scalar (the maximum entanglement).
    resolution: number of points along the line.
    margin: extra fraction of space around the segment formed by the 2 points.
    """
    origin, e1, coords = build_line_basis(theta1, theta2)

    a_min, a_max = coords["theta1"], coords["theta2"]
    if a_min > a_max:
        a_min, a_max = a_max, a_min

    range_a = a_max - a_min
    a_min -= margin * range_a
    a_max += margin * range_a

    a_vals = np.linspace(a_min, a_max, resolution)

    E = np.zeros_like(a_vals)
    total = resolution
    for i in range(resolution):
        theta = origin + a_vals[i] * e1
        E[i] = max_entanglement_fn(theta)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{total}")

    return a_vals, E, coords


def _clip_polygon(polygon, normal, offset):
    """Part of the convex polygon (k, 2) with normal . x <= offset
    (Sutherland-Hodgman, one half-plane). Keeps the orientation."""
    clipped = []
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        fp, fq = normal @ p - offset, normal @ q - offset
        if fp <= 0:
            clipped.append(p)
        if fp * fq < 0:
            clipped.append(p + fp / (fp - fq) * (q - p))
    return np.array(clipped).reshape(-1, 2)


def bounds_region(theta1, theta2, theta3, A, B, bounds, atol=1e-9):
    """
    Part of the plotted plane whose parameters are within `bounds`: one
    [low, high] per parameter (+-inf for no bound), e.g.
    src.sparse_grape_method.lzs_bounds. theta = origin + a e1 + b e2 is
    linear in (a, b), so every bound is a half-plane and the region is a
    convex polygon, computed exactly (it does not depend on the grid).

    Returns (polygon, inside):
      polygon : (k, 2) vertices (a, b) of the region clipped to the plotted
                rectangle, counterclockwise; None if the region is empty
      inside  : bool, shape of A, grid points within the bounds (up to atol:
                the optimized parameters often sit on a bound)
    """
    origin, e1, e2, _ = build_plane_basis(theta1, theta2, theta3)
    low, high = np.array(bounds, dtype=float).T

    # every bound as a half-plane normal . (a, b) <= offset
    halfplanes = []
    for i in range(len(origin)):
        normal = np.array([e1[i], e2[i]])
        if np.isfinite(low[i]):
            halfplanes.append((-normal, origin[i] - low[i]))
        if np.isfinite(high[i]):
            halfplanes.append((normal, high[i] - origin[i]))

    a0, a1, b0, b1 = A.min(), A.max(), B.min(), B.max()
    polygon = np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1]])
    inside = np.ones(A.shape, dtype=bool)
    for normal, offset in halfplanes:
        if len(polygon):
            polygon = _clip_polygon(polygon, normal, offset)
        inside &= normal[0] * A + normal[1] * B <= offset + atol

    return (polygon if len(polygon) >= 3 else None), inside


def _make_contour(ax, A, B, Z, cmap, vmin, vmax, log=False):
    """
    Helper shared by the 2D landscape plots.
    If vmin/vmax are both given, the colormap is fixed to that range
    (levels are built to be evenly spaced within it, and values that fall
    outside are clipped/shown via the colorbar's extend triangles).
    Otherwise the color scale is computed automatically from the data.
    log: logarithmic color scale (levels evenly spaced in log, Z > 0).
    """
    if log:
        positive = Z[Z > 0]
        if vmin is None or vmax is None:
            vmin, vmax = positive.min(), positive.max()
        levels = np.geomspace(vmin, vmax, 101)
        return ax.contourf(
            A,
            B,
            np.clip(Z, positive.min(), None),
            levels=levels,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extend="both",
        )
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 101)
        cont = ax.contourf(
            A, B, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both"
        )
    else:
        cont = ax.contourf(A, B, Z, levels=100, cmap=cmap)
    return cont


def _draw_bounds_region(ax, A, B, polygon):
    """Fades the plane outside the polygon and draws its edge (dashed)."""
    a0, a1, b0, b1 = A.min(), A.max(), B.min(), B.max()
    # rectangle counterclockwise with the polygon as a clockwise hole
    frame = np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1], [a0, b0]])
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = [frame]
    if polygon is not None:
        hole = np.vstack([polygon[::-1], polygon[-1:]])
        vertices.append(hole)
        codes += [Path.MOVETO] + [Path.LINETO] * (len(polygon) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(
        PathPatch(
            Path(np.vstack(vertices), codes),
            facecolor="white",
            alpha=0.55,
            edgecolor="none",
            zorder=2,
        )
    )
    if polygon is not None:
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            "--",
            color="black",
            linewidth=1.2,
            zorder=3,
            label="Within the optimization bounds",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def plot_landscape(
    A,
    B,
    Z,
    coords,
    energies,
    label,
    title,
    save_path=None,
    vmin=None,
    vmax=None,
    region=None,
    log=False,
    cmap="terrain",
):
    """
    2D landscape Z on the plane grid (A, B), with the three parameter
    vectors and their final energies.

    label  : colorbar label
    vmin, vmax : optional, fix the color scale to this range (e.g. to compare
        several plots on the same scale).
    region : optional (polygon, inside) from bounds_region. The polygon is
        drawn (dashed) and the plane outside it faded; unless vmin/vmax are
        given, the color scale is set by the values inside only, so the
        points outside the bounds do not take up the color range.
    log    : logarithmic color scale (Z > 0).
    Without region, vmin, vmax, the scale is computed from all the data.
    """
    if region is not None and (vmin is None or vmax is None):
        polygon, inside = region
        values = Z[inside & np.isfinite(Z)]
        if log:
            values = values[values > 0]
        if values.size:
            vmin = values.min() if vmin is None else vmin
            vmax = values.max() if vmax is None else vmax
            if vmax <= vmin:
                vmax = vmin + 1e-12 * max(1.0, abs(vmin))

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

    cont = _make_contour(ax, A, B, Z, cmap, vmin, vmax, log=log)

    # colorbar con tamaño fijo relativo al heatmap, no a toda la figura
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% del ancho del heatmap
    cbar = plt.colorbar(cont, cax=cax, label=label)
    if log:
        # ticks at 1, 2, 5 x 10^n instead of the (long) values of the levels
        cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
        cbar.ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))

    if region is not None:
        _draw_bounds_region(ax, A, B, region[0])

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
        "theta3": r"$\theta_3$",
    }

    for name, (a, b) in coords.items():
        ax.plot(a, b, "o", color="red", markersize=8, zorder=4)
        ax.annotate(
            nombres_latex[name],
            (a, b),
            textcoords="offset points",
            xytext=(6, 6),
            color="white",
            fontsize=12,
            zorder=4,
            # dark outline: readable on the faded (white) part of the plane too
            path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
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


def plot_energy_landscape(
    A,
    B,
    E,
    coords,
    energies,
    title="Energy landscape",
    save_path=None,
    vmin=None,
    vmax=None,
    region=None,
):
    """plot_landscape of the final energy (see there for vmin, vmax, region)."""
    return plot_landscape(
        A, B, E, coords, energies, "Final energy", title, save_path, vmin, vmax, region
    )


def plot_energy_landscape_1d(
    a_vals, E, coords, energies, title="Energy landscape", save_path=None
):
    ancho = 8  # pulgadas
    alto_plot = 4.5
    alto_texto = 1.5  # pulgadas reservadas para la caja de texto abajo
    alto_total = alto_plot + alto_texto

    fig, ax = plt.subplots(figsize=(ancho, alto_total))

    ax.plot(a_vals, E, "-", color="steelblue", linewidth=2)

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
    }

    for name, a in coords.items():
        idx = np.argmin(np.abs(a_vals - a))
        ax.plot(a, E[idx], "o", color="red", markersize=8)
        ax.annotate(
            nombres_latex[name],
            (a, E[idx]),
            textcoords="offset points",
            xytext=(6, 6),
            color="black",
            fontsize=12,
        )

    ax.set_xlabel("a (direction $e_1$)")
    ax.set_ylabel("Final energy")
    ax.set_title(title)
    bottom_frac = alto_texto / alto_total

    # bottom como fracción del alto total
    fig.subplots_adjust(bottom=alto_texto / alto_total)

    texto_energias = "       ".join(
        [rf"{nombres_latex[name]}: $E$ = {energies[name]:.6f}" for name in coords]
    )
    fig.text(
        0.5,
        0.6 * bottom_frac,
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
    A,
    B,
    max_magic,
    coords,
    energies,
    title="Max magic landscape",
    save_path=None,
    vmin=None,
    vmax=None,
    region=None,
):
    """plot_landscape of the maximum magic (see there for vmin, vmax, region)."""
    return plot_landscape(
        A,
        B,
        max_magic,
        coords,
        energies,
        "Maximum magic",
        title,
        save_path,
        vmin,
        vmax,
        region,
    )


def plot_max_magic_landscape_1d(
    a_vals, max_magic, coords, energies, title="Max magic landscape", save_path=None
):
    ancho = 8
    alto_plot = 4.5
    alto_texto = 1.5
    alto_total = alto_plot + alto_texto

    fig, ax = plt.subplots(figsize=(ancho, alto_total))

    ax.plot(a_vals, max_magic, "-", color="steelblue", linewidth=2)

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
    }

    for name, a in coords.items():
        idx = np.argmin(np.abs(a_vals - a))
        ax.plot(a, max_magic[idx], "o", color="red", markersize=8)
        ax.annotate(
            nombres_latex[name],
            (a, max_magic[idx]),
            textcoords="offset points",
            xytext=(6, 6),
            color="black",
            fontsize=12,
        )

    ax.set_xlabel("a (direction $e_1$)")
    ax.set_ylabel("Maximum magic")
    ax.set_title(title)
    bottom_frac = alto_texto / alto_total

    fig.subplots_adjust(bottom=alto_texto / alto_total)

    texto_energias = "       ".join(
        [rf"{nombres_latex[name]}: $E$ = {energies[name]:.6f}" for name in coords]
    )
    fig.text(
        0.5,
        0.6 * bottom_frac,
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
    vmin=None,
    vmax=None,
    region=None,
):
    """plot_landscape of the maximum entanglement (see there for vmin, vmax, region)."""
    return plot_landscape(
        A,
        B,
        max_entanglement,
        coords,
        energies,
        "Maximum entanglement",
        title,
        save_path,
        vmin,
        vmax,
        region,
    )


def plot_max_entanglement_landscape_1d(
    a_vals,
    max_entanglement,
    coords,
    energies,
    title="Max entanglement landscape",
    save_path=None,
):
    ancho = 8
    alto_plot = 4.5
    alto_texto = 1.5
    alto_total = alto_plot + alto_texto

    fig, ax = plt.subplots(figsize=(ancho, alto_total))

    ax.plot(a_vals, max_entanglement, "-", color="steelblue", linewidth=2)

    nombres_latex = {
        "theta1": r"$\theta_1$",
        "theta2": r"$\theta_2$",
    }

    for name, a in coords.items():
        idx = np.argmin(np.abs(a_vals - a))
        ax.plot(a, max_entanglement[idx], "o", color="red", markersize=8)
        ax.annotate(
            nombres_latex[name],
            (a, max_entanglement[idx]),
            textcoords="offset points",
            xytext=(6, 6),
            color="black",
            fontsize=12,
        )

    ax.set_xlabel("a (direction $e_1$)")
    ax.set_ylabel("Maximum entanglement")
    ax.set_title(title)
    bottom_frac = alto_texto / alto_total

    fig.subplots_adjust(bottom=alto_texto / alto_total)

    texto_energias = "       ".join(
        [rf"{nombres_latex[name]}: $E$ = {energies[name]:.6f}" for name in coords]
    )
    fig.text(
        0.5,
        bottom_frac * 0.6,
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


# ── theta -> observables, for the landscape scripts ──────────────────────────
class RingLandscape:
    """
    theta -> observables of the annealing of the frustrated ring, for the
    landscape scripts.

    The schedule is decoded by the optimization model itself
    (NambuGRAPEModel, type "LZS", bounds_opt=True: the ansatz the parameters
    of ParametersLZR are optimized with), so the landscapes follow any change
    of the ansatz. Backends:

      evolution, final energy, entanglement : free fermions (exact)
      magic      : spin basis (Z2 +1 sector, SREJax, exact) for
                   N <= spin_magic_max_n, free fermions (Majorana sampling,
                   NambuIsing1D.sre) above it. For N = 7 SREJax is ~150 times
                   faster per state than the sampling, and has no noise
      robustness : spin basis, full 2^N space (FidelityRobustness with the 3
                   global fields sum_i sigma^a_i). The sigma^y and sigma^z
                   (spin-basis axes) fields are not Gaussian in the
                   free-fermion picture

    The spin-basis objects are only built when `magic` or `robustness` needs
    them.

    Parameters
    ----------
    N, T              : number of qubits and total annealing time
    steps_per_unit    : propagation steps per unit of time,
                        nsteps = steps_per_unit * T
    number_parameters : M plateaus of the LZS ansatz (3 M + 1 parameters)
    J, JL, JR         : couplings of the ring
    magic_dt          : time between the states the maximum magic is taken
                        over (rounded to a whole number of steps). 0.1 gives
                        the maximum within ~1e-4 nats of dt = 0.01 (N=7,
                        T=100); 2.0 was off by up to ~0.05
    n_samples         : Majorana samples per SRE estimate (free-fermion magic)
    spin_magic_max_n  : largest N whose magic is computed in the spin basis
    magic, robustness : build what max_magic / robustness need
    """

    def __init__(
        self,
        N,
        T,
        steps_per_unit=100,
        number_parameters=2,
        J=1.0,
        JL=0.5,
        JR=0.45,
        magic_dt=0.1,
        n_samples=1000,
        spin_magic_max_n=9,
        magic=False,
        robustness=False,
    ):
        self.N = int(N)
        self.T = T
        self.nsteps = int(steps_per_unit * T)
        self.times, self.times_ctrl, self.dt = make_time_grids(T, self.nsteps)
        self.n_samples = n_samples
        self.spin_magic = self.N <= spin_magic_max_n

        # states the maximum magic is taken over (always t = 0 and t = T)
        stride = max(1, int(round(magic_dt / self.dt)))
        self.magic_indices = set(subsample_indices(self.nsteps, stride).tolist())

        # free fermions: same ring as frustrated_ring_jij_hz, in the spin
        # convention of src.free_fermions_utils (axes rotated with respect to
        # the spin basis); pbc=True fixes the Z2 +1 sector
        self.nambu = NambuIsing1D.frustrated_ring(self.N, J, JL, JR)
        _, self.w_init = self.nambu.diagonalize(1.0, 0.0)
        self.model = NambuGRAPEModel(
            self.nambu,
            tf=T,
            number_of_parameters=number_parameters,
            nsteps=self.nsteps,
            type="LZS",
            bounds_opt=True,
        )

        if (magic and self.spin_magic) or robustness:
            jij, hz = frustrated_ring_jij_hz(self.N, J, JL, JR)
            # full 2^N space
            self.target_hamiltonian = get_longitudinal_hamiltonian(jij, hz)
            self.driver_hamiltonian = get_driver_hamiltonian(nqubits=self.N)
            self.psi_init = np.ones(2**self.N, dtype=complex) / np.sqrt(2**self.N)

        if magic and self.spin_magic:
            from src.jax_utils import SREJax  # JAX only when needed

            # the evolution from the uniform superposition stays in the +1
            # sector of Pi = prod_i X_i (the Hamiltonian commutes with it)
            sector = Z2SymmetricSector(self.N, sign=+1)
            self.target_hamiltonian_s = sector.project(self.target_hamiltonian)
            self.driver_hamiltonian_s = sector.project(self.driver_hamiltonian)
            self.psi_init_s = sector.project(self.psi_init)
            self.sre = SREJax(n_qubits=self.N - 1, batch_size=1000)

        if robustness:
            self.global_operators, self.global_labels = global_pauli_operators(
                self.N
            )

    # ─────────────────────────────────────────────────────────────────────────
    def schedule(self, theta):
        """s(t) on the control grid (drives the evolution) and on the state
        grid (instantaneous Hamiltonian of the observables)."""
        s_ctrl = self.model.get_driving(theta, grid="control")[1]
        s_state = self.model.get_driving(theta, grid="state")[1]
        return s_ctrl, s_state

    def _nambu_states(self, s_ctrl):
        """(k, W1 [2N, N]) for the states at times[k], k = 0..nsteps."""
        for k, w in nambu_midpoint_evolution(
            self.nambu, self.w_init, 1 - s_ctrl, s_ctrl, self.dt
        ):
            yield k, w[:, : self.N]

    def _final_energy(self, w1, s_final):
        hamiltonian = self.nambu.hamiltonian(1 - s_final, s_final)
        return float(np.real(np.trace(w1.conj().T @ hamiltonian @ w1)))

    def _max_magic_spin(self, s_ctrl):
        max_magic = -np.inf
        for k, psi in midpoint_evolution(
            self.psi_init_s,
            1 - s_ctrl,
            s_ctrl,
            self.dt,
            self.driver_hamiltonian_s,
            self.target_hamiltonian_s,
        ):
            if k in self.magic_indices:
                max_magic = max(max_magic, self.sre(psi))
        return float(max_magic)

    def _magic_ff(self, theta, k, w1):
        """Sampled SRE of the state at times[k]. The seed depends on theta
        and k: the noise is independent from point to point (a seed shared
        by all the points turns it into a smooth, misleading deformation of
        the landscape) and reproducible."""
        seed = [zlib.crc32(np.ascontiguousarray(theta, dtype=float).tobytes()), k]
        return self.nambu.sre(w1, alpha=2, n_samples=self.n_samples, seed=seed)[
            "m_alpha"
        ]

    # ─────────────────────────────────────────────────────────────────────────
    def energy(self, theta):
        """Final energy <H(s(T))> (free fermions)."""
        s_ctrl, s_state = self.schedule(theta)
        for _, w1 in self._nambu_states(s_ctrl):
            pass
        return self._final_energy(w1, s_state[-1])

    def max_entanglement(self, theta):
        """Maximum over every state of the half-chain von Neumann entropy
        (nats, free fermions)."""
        s_ctrl, _ = self.schedule(theta)
        return max(
            self.nambu.entanglement_entropy(w1, self.N // 2, base=np.e)
            for _, w1 in self._nambu_states(s_ctrl)
        )

    def max_magic(self, theta):
        """Maximum of the SRE M_2 (nats) over the states every magic_dt."""
        s_ctrl, _ = self.schedule(theta)
        if self.spin_magic:
            return self._max_magic_spin(s_ctrl)
        return max(
            self._magic_ff(theta, k, w1)
            for k, w1 in self._nambu_states(s_ctrl)
            if k in self.magic_indices
        )

    def observables(self, theta):
        """(final energy, maximum magic, maximum entanglement) with one
        free-fermion evolution (plus a spin-basis one for the magic if
        N <= spin_magic_max_n)."""
        s_ctrl, s_state = self.schedule(theta)
        max_entanglement, max_magic = -np.inf, -np.inf
        for k, w1 in self._nambu_states(s_ctrl):
            max_entanglement = max(
                max_entanglement,
                self.nambu.entanglement_entropy(w1, self.N // 2, base=np.e),
            )
            if not self.spin_magic and k in self.magic_indices:
                max_magic = max(max_magic, self._magic_ff(theta, k, w1))
        if self.spin_magic:
            max_magic = self._max_magic_spin(s_ctrl)
        return self._final_energy(w1, s_state[-1]), max_magic, max_entanglement

    def robustness(self, theta):
        """(G_xx, G_yy, G_zz): diagonal of the robustness matrix G of
        src/fidelity_robustness.py for the global fields lambda sum_i
        sigma^a_i (spin-basis axes, full space): 1 - F ~ lambda^2 G_aa, F the
        fidelity with the unperturbed final state."""
        s_ctrl, _ = self.schedule(theta)
        robustness = FidelityRobustness(
            self.N,
            self.driver_hamiltonian,
            self.target_hamiltonian,
            self.psi_init,
            s_ctrl,
            self.T,
            operators=self.global_operators,
            labels=self.global_labels,
        )
        return tuple(np.real(np.diag(robustness.compute())))
