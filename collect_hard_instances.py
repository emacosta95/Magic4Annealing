"""
collect_hard_instances.py
─────────────────────────
Samples random 3-regular graphs with ±1 couplings, runs a linear quantum
annealing protocol for each instance, ranks by spectral gap (smallest gap
= hardest), and for each instance computes:

  - M2 (Stabilizer Rényi Entropy) of the adiabatic GS and first excited state
  - Bipartite entanglement entropy (von Neumann) of GS and first excited state

Results are saved sorted by gap magnitude (hardest first) in a pickle file.

Usage
-----
    python collect_hard_instances.py \
        --nqubits 8 --n_instances 200 --n_keep 20 \
        --tau 10 --time_steps 100 --output hard_instances.pkl
"""

import argparse
import pickle
import time as _time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply
from tqdm import trange

# ── project imports ───────────────────────────────────────────────────────────
from src.annealing_utils import get_driver_hamiltonian, get_longitudinal_hamiltonian
from src.utils import Sector
from src.jax_utils import SREJax


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian builders
# ─────────────────────────────────────────────────────────────────────────────
def sample_instance(nqubits: int, seed: int) -> np.ndarray:
    """Sample a random 3-regular graph with ±1 couplings."""
    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(d=3, n=nqubits, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.uniform(0.0, 1)
    return nx.to_numpy_array(G, weight="weight")


# ─────────────────────────────────────────────────────────────────────────────
# Gap computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_minimum_gap(
    driver_hamiltonian_s,
    target_hamiltonian_s,
    tau: float,
    time_steps: int,
    nlevels: int = 6,
) -> tuple:
    """
    Run the linear annealing schedule and return the minimum spectral gap
    (between the degenerate GS pair and the first excited subspace)
    and the time at which it occurs.

    Returns (min_gap, t_min_gap, spectrum)
    """
    time = np.linspace(0, tau, time_steps)
    delta_t = time[1] - time[0]
    spectrum = np.zeros((time_steps, nlevels))

    for i, t in enumerate(time):
        H_t = (1 - t / tau) * driver_hamiltonian_s + (t / tau) * target_hamiltonian_s
        evals, _ = eigsh(H_t.astype(complex), which="SA", k=nlevels)
        spectrum[i] = np.sort(evals)

    # gap between degenerate GS pair (indices 0,1) and first excited subspace
    # (index 2) — consistent with the doubly-degenerate Z2-symmetric Ising GS
    gaps = spectrum[:, 1] - spectrum[:, 0]
    i_min = int(np.argmin(gaps))
    min_gap = float(gaps[i_min])
    t_min = float(time[i_min])
    return min_gap, t_min, spectrum


# ─────────────────────────────────────────────────────────────────────────────
# Entanglement entropy
# ─────────────────────────────────────────────────────────────────────────────
def von_neumann_entropy(psi: np.ndarray, n_A: int) -> float:
    """
    Bipartite von Neumann entropy S_A for a pure state psi.
    Bipartition: first n_A qubits vs rest.
    """
    dim_A = 2**n_A
    dim_B = psi.shape[0] // dim_A
    M = psi.reshape(dim_A, dim_B)
    sv = np.linalg.svd(M, compute_uv=False)
    lam2 = sv**2
    lam2 = lam2[lam2 > 1e-15]
    return float(-np.sum(lam2 * np.log(lam2)))


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────
def collect_instances(
    nqubits: int,
    n_instances: int,
    n_keep: int,
    tau: float,
    time_steps: int,
    nlevels: int = 6,
    verbose: bool = True,
) -> list:
    """
    Sample n_instances random graphs, compute their minimum gaps,
    keep the n_keep hardest, then compute magic and entanglement for each.

    Returns a list of dicts sorted by gap (hardest first).
    """
    # ── build shared objects ─────────────────────────────────────────────────
    PS = Sector(nqubits=nqubits)
    sre = SREJax(n_qubits=nqubits, batch_size=4096)
    n_A = nqubits // 2
    dim = 2**nqubits
    dim_s = dim // 2
    time = np.linspace(0, tau, time_steps)
    delta_t = time[1] - time[0]

    psi_init_full = np.ones(dim, dtype=complex) / np.sqrt(dim)
    psi_init_s = PS.project(psi_init_full)

    driver_hamiltonian = get_driver_hamiltonian(nqubits=nqubits)
    driver_hamiltonian_s = PS.project(driver_hamiltonian)

    # ── phase 1: rank all instances by gap ───────────────────────────────────
    print(f"\n── Phase 1: sampling {n_instances} instances ──")
    gap_records = []

    for seed in trange(n_instances, desc="Sampling gaps"):
        jij = sample_instance(nqubits, seed)
        target_hamiltonian = get_longitudinal_hamiltonian(jij)
        target_hamiltonian_s = PS.project(target_hamiltonian)

        min_gap, t_min, spectrum = compute_minimum_gap(
            driver_hamiltonian_s,
            target_hamiltonian_s,
            tau,
            time_steps,
            nlevels,
        )

        gap_records.append(
            {
                "seed": seed,
                "jij": jij,
                "min_gap": min_gap,
                "t_min_gap": t_min,
                "spectrum": spectrum,
            }
        )

    # sort by gap ascending (hardest = smallest gap first)
    gap_records.sort(key=lambda x: x["min_gap"])
    hardest = gap_records[:n_keep]

    print(
        f"\nGap range of kept instances: "
        f'{hardest[0]["min_gap"]:.4f} … {hardest[-1]["min_gap"]:.4f}'
    )

    # ── phase 2: full magic + entanglement for hardest instances ──────────────
    print(f"\n── Phase 2: computing magic & entanglement for top {n_keep} ──")
    results = []

    for rank, rec in enumerate(hardest):
        t0 = _time.time()
        print(f'\n[{rank+1}/{n_keep}] seed={rec["seed"]}  gap={rec["min_gap"]:.6f}')

        jij = rec["jij"]
        target_hamiltonian = get_longitudinal_hamiltonian(jij)
        target_hamiltonian_s = PS.project(target_hamiltonian)

        # ── linear time evolution ─────────────────────────────────────────────
        psi_history = np.zeros((time_steps, dim_s), dtype=complex)
        psi = psi_init_s.copy()

        # also store instantaneous GS and 1st excited state along the path
        gs_history = np.zeros((time_steps, dim_s), dtype=complex)
        ex_history = np.zeros((time_steps, dim_s), dtype=complex)

        for i, t in enumerate(time):
            H_t = (1 - t / tau) * driver_hamiltonian_s + (
                t / tau
            ) * target_hamiltonian_s
            psi = expm_multiply(-1j * delta_t * H_t, psi)
            psi_history[i] = psi

            # instantaneous eigenstates
            evals, evecs = eigsh(H_t.astype(complex), which="SA", k=4)
            order = np.argsort(evals)
            evecs = evecs[:, order]
            gs_history[i] = evecs[:, 0].astype(complex)
            ex_history[i] = evecs[:, 2].astype(complex)  # skip degenerate partner

        # ── magic along path ──────────────────────────────────────────────────
        print("  Computing M2 along path...")
        magic_evolution = np.zeros(time_steps)
        magic_gs = np.zeros(time_steps)
        magic_ex = np.zeros(time_steps)

        for i in trange(time_steps, desc="  M2", leave=False):
            magic_evolution[i] = sre(PS.lift(psi_history[i]))
            magic_gs[i] = sre(PS.lift(gs_history[i]))
            magic_ex[i] = sre(PS.lift(ex_history[i]))

        # ── entanglement along path ───────────────────────────────────────────
        print("  Computing entanglement along path...")
        ee_evolution = np.zeros(time_steps)
        ee_gs = np.zeros(time_steps)
        ee_ex = np.zeros(time_steps)

        for i in range(time_steps):
            ee_evolution[i] = von_neumann_entropy(PS.lift(psi_history[i]), n_A)
            ee_gs[i] = von_neumann_entropy(PS.lift(gs_history[i]), n_A)
            ee_ex[i] = von_neumann_entropy(PS.lift(ex_history[i]), n_A)

        # ── final state magic & entanglement ──────────────────────────────────
        # GS and first excited state of target Hamiltonian
        evals_t, evecs_t = eigsh(target_hamiltonian_s.astype(complex), which="SA", k=4)
        order_t = np.argsort(evals_t)
        evecs_t = evecs_t[:, order_t]

        psi_gs_target = evecs_t[:, 0].astype(complex)
        psi_ex_target = evecs_t[:, 2].astype(complex)

        magic_gs_target = sre(PS.lift(psi_gs_target))
        magic_ex_target = sre(PS.lift(psi_ex_target))
        ee_gs_target = von_neumann_entropy(PS.lift(psi_gs_target), n_A)
        ee_ex_target = von_neumann_entropy(PS.lift(psi_ex_target), n_A)

        # ── fidelity ──────────────────────────────────────────────────────────
        # sum over degenerate GS pair (indices 0 and 1)
        gs_subspace = evecs_t[:, :2]
        fidelity = float(np.sum(np.abs(gs_subspace.conj().T @ psi_history[-1]) ** 2))

        elapsed = _time.time() - t0
        print(
            f"  Done in {elapsed:.1f}s  fidelity={fidelity:.4f}  "
            f"M2_gs_target={magic_gs_target:.4f}  "
            f"EE_gs_target={ee_gs_target:.4f}"
        )

        results.append(
            {
                # ── instance identity ────────────────────────────────────────────
                "rank": rank,
                "seed": rec["seed"],
                "jij": jij,
                # ── gap info ─────────────────────────────────────────────────────
                "min_gap": rec["min_gap"],
                "t_min_gap": rec["t_min_gap"],
                "spectrum": rec["spectrum"],
                # ── time axis ────────────────────────────────────────────────────
                "time": time,
                "tau": tau,
                # ── magic along path ─────────────────────────────────────────────
                "magic_evolution": magic_evolution,  # M2 of annealing state
                "magic_gs": magic_gs,  # M2 of instantaneous GS
                "magic_ex": magic_ex,  # M2 of instantaneous 1st excited
                # ── entanglement along path ───────────────────────────────────────
                "ee_evolution": ee_evolution,  # S_A of annealing state
                "ee_gs": ee_gs,  # S_A of instantaneous GS
                "ee_ex": ee_ex,  # S_A of instantaneous 1st excited
                # ── target Hamiltonian final states ───────────────────────────────
                "magic_gs_target": magic_gs_target,  # M2 of target GS
                "magic_ex_target": magic_ex_target,  # M2 of target 1st excited
                "ee_gs_target": ee_gs_target,  # S_A of target GS
                "ee_ex_target": ee_ex_target,  # S_A of target 1st excited
                # ── final fidelity ────────────────────────────────────────────────
                "fidelity": fidelity,
                # ── integrated quantities ─────────────────────────────────────────
                "magic_production": float(np.trapz(magic_evolution, time)),
                "ee_production": float(np.trapz(ee_evolution, time)),
                "magic_gs_production": float(np.trapz(magic_gs, time)),
                "ee_gs_production": float(np.trapz(ee_gs, time)),
            }
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Collect hardest QA instances")
    parser.add_argument("--nqubits", type=int, default=8)
    parser.add_argument(
        "--n_instances", type=int, default=200, help="Total instances to sample"
    )
    parser.add_argument(
        "--n_keep", type=int, default=20, help="Number of hardest instances to analyse"
    )
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--time_steps", type=int, default=100)
    parser.add_argument("--nlevels", type=int, default=6)
    parser.add_argument("--output", type=str, default="hard_instances.pkl")
    args = parser.parse_args()

    print(
        f"nqubits={args.nqubits}  n_instances={args.n_instances}  "
        f"n_keep={args.n_keep}  tau={args.tau}  time_steps={args.time_steps}"
    )

    results = collect_instances(
        nqubits=args.nqubits,
        n_instances=args.n_instances,
        n_keep=args.n_keep,
        tau=args.tau,
        time_steps=args.time_steps,
        nlevels=args.nlevels,
    )

    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {len(results)} instances to {output_path}")
    print("\nSummary (sorted by gap, hardest first):")
    print(
        f'{"rank":>4}  {"seed":>6}  {"gap":>10}  {"fidelity":>10}  '
        f'{"M2_gs":>10}  {"EE_gs":>10}'
    )
    print("-" * 60)
    for r in results:
        print(
            f'{r["rank"]:>4}  {r["seed"]:>6}  {r["min_gap"]:>10.6f}  '
            f'{r["fidelity"]:>10.4f}  '
            f'{r["magic_gs_target"]:>10.4f}  '
            f'{r["ee_gs_target"]:>10.4f}'
        )


if __name__ == "__main__":
    main()
