# ── scan_2d + plot_scan (only needed once — skip if already defined) ──
from tqdm import trange

from src.jax_utils import SREJax
from src.utils import EntanglementEntropy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.sparse.linalg import expm_multiply


def scan_2d(
    model,
    center,
    v1,
    v2,
    sector,
    n_qubits,
    mu_range,
    nu_range,
    driver_hamiltonian_s,
    target_hamiltonian_s,
    psi_init_s=None,
    time_subsample=10,
    verbose=True,
):
    sre = SREJax(n_qubits)
    ent = EntanglementEntropy(n_qubits)
    nM, nN = len(mu_range), len(nu_range)

    E_grid = np.zeros((nM, nN))

    M2_mean_grid = np.zeros((nM, nN))
    M2_max_grid = np.zeros((nM, nN))
    M2_integral_grid = np.zeros((nM, nN))
    M2_final_grid = np.zeros((nM, nN))

    S_mean_grid = np.zeros((nM, nN))
    S_max_grid = np.zeros((nM, nN))
    S_integral_grid = np.zeros((nM, nN))
    S_final_grid = np.zeros((nM, nN))

    for i in trange(nM, desc="trajectory scan", disable=not verbose):
        for j in range(nN):
            theta = center + mu_range[i] * v1 + nu_range[j] * v2

            # energy: single cheap forward pass, no need for the full trajectory
            E, _ = model.forward_and_gradient(theta)
            E_grid[i, j] = E

            # full time-resolved propagation for magic + entanglement
            psi_hist, t_sub = propagate_trajectory(
                model,
                theta,
                sector,
                driver_hamiltonian_s,
                target_hamiltonian_s,
                time_subsample=time_subsample,
                psi_init_s=psi_init_s,
            )
            m2_t = np.array([sre(psi) for psi in psi_hist])
            s_t = np.array([ent.von_neumann(psi) for psi in psi_hist])

            M2_mean_grid[i, j] = m2_t.mean()
            M2_max_grid[i, j] = m2_t.max()
            M2_integral_grid[i, j] = np.trapz(m2_t, t_sub) / (t_sub[-1] - t_sub[0])
            M2_final_grid[i, j] = m2_t[-1]

            S_mean_grid[i, j] = s_t.mean()
            S_max_grid[i, j] = s_t.max()
            S_integral_grid[i, j] = np.trapz(s_t, t_sub) / (t_sub[-1] - t_sub[0])
            S_final_grid[i, j] = s_t[-1]

    model.forward_and_gradient(center)  # reset model state

    return {
        "E": E_grid,
        "M2_mean": M2_mean_grid,
        "M2_max": M2_max_grid,
        "M2_integral": M2_integral_grid,
        "M2_final": M2_final_grid,
        "S_mean": S_mean_grid,
        "S_max": S_max_grid,
        "S_integral": S_integral_grid,
        "S_final": S_final_grid,
    }


def plot_scan(mu_range, nu_range, E_grid, M2_grid, S_grid, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, grid, label in zip(
        axes, [E_grid, M2_grid, S_grid], ["E", "M2 (SRE)", "S_vN"]
    ):
        im = ax.pcolormesh(nu_range, mu_range, grid, shading="auto", cmap="viridis")
        ax.plot(0, 0, marker="*", color="red", markersize=14, markeredgecolor="k")
        ax.set_xlabel(r"$\nu$")
        ax.set_ylabel(r"$\mu$")
        ax.set_title(f"{title_prefix}{label}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    return fig


def propagate_trajectory(
    model,
    theta,
    sector,
    driver_hamiltonian_s,
    target_hamiltonian_s,
    time_subsample=10,
    psi_init_s=None,
):
    """
    Full time-resolved propagation for a given theta. Returns psi_history_full
    (n_sub_steps, 2^n) — the state lifted to the full space at each *sampled*
    time step, plus the corresponding subsampled time indices.

    time_subsample: keep every k-th step. SRE is not free (4^n Paulis per
    call), so sampling every ~10-20 steps out of nsteps=500 is usually enough
    to resolve the magic profile without recomputing all 500.
    """
    model.forward_and_gradient(theta)  # syncs h_driver/h_target for this theta
    h_driver, h_target = model.get_driving()
    dt = model.dt

    idx = np.arange(0, model.nsteps, time_subsample)
    psi = psi_init_s.copy()
    psi_history_full = np.zeros((len(idx), sector.dim), dtype=complex)

    k = 0
    for i in range(model.nsteps):
        H_t = h_driver[i] * driver_hamiltonian_s + h_target[i] * target_hamiltonian_s
        psi = expm_multiply(-1j * dt * H_t, psi)
        if i in idx:
            psi_full = sector.lift(psi)
            psi_history_full[k] = psi_full / np.linalg.norm(psi_full)
            k += 1

    return psi_history_full, model.time[idx]
