"""
lzs_tau_scan.py
────────────────
Systematic study of LZS-schedule optimal control for the frustrated Ising
ring (Cote et al. / Werner et al.), matching the model + hyperparameters of
FrustratedRing.ipynb ("Optimal Control — LZS scheduler" section):

    N=7, (J, JL, JR) = (1.0, 0.5, 0.45), Z2 +1-sector projection
    LZS schedule, M=2 plateaus (number_of_parameters=2 -> 7 raw parameters)
    JaxSchedulerModel + JaxTrainer (exact gradients, L-BFGS-B),
        maxiter=500, tol=1e-6, ftol=1e-5, gtol=1e-4
    time_steps = int(30 * tau)

For each tau in a sweep, 10 independent optimizations are run (seeds 1..10,
random LZS parameter initialization), in parallel across (tau, seed) pairs.
For every (tau, seed) run we record:

    - energy(t)                  (full time resolution, cheap to compute)
    - magic M2(t)                (subsampled, stride=10 -- O(4^N) per call)
    - entanglement S_vN(t)       (subsampled, stride=10)
    - final energy, energy error vs the true target ground energy
    - max magic, max entanglement over the trajectory

All 10 seeds for a given tau are saved together in one file:
    data/lzs_scan/lzs_scan_tau{tau:07.2f}.npz

Usage
-----
    python lzs_tau_scan.py
    python lzs_tau_scan.py --taus 5 10 20 30 50 70 100 150 200 250 300 \
                           --seeds 1 2 3 4 5 6 7 8 9 10 --workers 6
"""

import argparse
import os
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── model + hyperparameters (mirrors FrustratedRing.ipynb) ────────────────────
N_QUBITS = 7
J, JL, JR = 1.0, 0.5, 0.45
NUMBER_PARAMETERS = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7
SCHEDULE_TYPE = "LZS"
STEPS_PER_TAU = 30  # time_steps = int(STEPS_PER_TAU * tau), matches the notebook

MAXITER = 500
OPT_TOL = 1e-6
OPT_FTOL = 1e-5
OPT_GTOL = 1e-4

STRIDE = 10  # magic/entanglement subsampling, matches the notebook

DEFAULT_TAUS = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300]
DEFAULT_SEEDS = list(range(1, 11))
OUTPUT_DIR = Path("data/lzs_scan")


# ─────────────────────────────────────────────────────────────────────────────
def frustrated_ring_jij_hz(N, J=1.0, JL=0.5, JR=0.45):
    """
    Build (jij, hz) for get_longitudinal_hamiltonian(jij, hz) -- exact copy
    of the helper defined in FrustratedRing.ipynb.
    """
    assert N % 2 == 1, "N must be odd"
    jij = np.zeros((N, N))
    mid1 = (N - 1) // 2
    mid2 = (N + 1) // 2
    for j in range(1, N + 1):
        if j == N:
            Jj = -JR
        elif j == mid1 or j == mid2:
            Jj = JL
        else:
            Jj = J
        i0 = j - 1
        i1 = j % N
        jij[i0, i1] += -Jj
        jij[i1, i0] += -Jj
    hz = np.zeros(N)
    return jij, hz


def _limit_threads(n_threads: int = 1):
    """Must run BEFORE numpy/jax are imported in this process to take effect."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n_threads)
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={n_threads}"
    )


def _build_system():
    """Sector-projected Hamiltonians + initial state -- shared across all runs."""
    from src.annealing_utils import get_driver_hamiltonian, get_longitudinal_hamiltonian
    from src.utils import Z2SymmetricSector

    jij, hz = frustrated_ring_jij_hz(N_QUBITS, J, JL, JR)
    target_hamiltonian = get_longitudinal_hamiltonian(jij, hz)
    driver_hamiltonian = get_driver_hamiltonian(nqubits=N_QUBITS)

    sector = Z2SymmetricSector(N_QUBITS, sign=+1)
    dim = 2**N_QUBITS
    psi_init_full = np.ones(dim, dtype=complex) / np.sqrt(dim)
    assert sector.check_confined(
        psi_init_full
    ), "initial state is not confined to the +1 sector!"

    target_hamiltonian_s = sector.project(target_hamiltonian)
    driver_hamiltonian_s = sector.project(driver_hamiltonian)
    psi_init_s = sector.project(psi_init_full)

    return sector, target_hamiltonian_s, driver_hamiltonian_s, psi_init_s


# ── per-worker-process state, built once via the pool initializer ─────────────
_STATE = {}


def _worker_init(threads_per_worker: int):
    _limit_threads(threads_per_worker)

    from scipy.sparse.linalg import eigsh

    from src.jax_utils import SREJax
    from src.utils import EntanglementEntropy

    sector, target_hamiltonian_s, driver_hamiltonian_s, psi_init_s = _build_system()

    _STATE["sector"] = sector
    _STATE["target_hamiltonian_s"] = target_hamiltonian_s
    _STATE["driver_hamiltonian_s"] = driver_hamiltonian_s
    _STATE["psi_init_s"] = psi_init_s
    _STATE["sre"] = SREJax(n_qubits=N_QUBITS, batch_size=1000)
    _STATE["entanglement_entropy"] = EntanglementEntropy(
        nqubits=N_QUBITS, n_A=N_QUBITS // 2
    )
    _STATE["e0_target"] = float(
        eigsh(
            target_hamiltonian_s.astype(complex),
            which="SA",
            k=1,
            return_eigenvectors=False,
        )[0]
    )


# ─────────────────────────────────────────────────────────────────────────────
def run_single_seed(tau: float, seed: int) -> dict:
    """Optimize the LZS schedule for one (tau, seed) pair and propagate it."""
    from scipy.sparse.linalg import expm_multiply

    from src.jax_utils import JaxSchedulerModel, JaxTrainer

    sector = _STATE["sector"]
    target_hamiltonian_s = _STATE["target_hamiltonian_s"]
    driver_hamiltonian_s = _STATE["driver_hamiltonian_s"]
    psi_init_s = _STATE["psi_init_s"]
    sre = _STATE["sre"]
    entanglement_entropy = _STATE["entanglement_entropy"]
    e0_target = _STATE["e0_target"]

    t0 = _time.time()

    time_steps = int(STEPS_PER_TAU * tau)
    time = np.linspace(0, tau, time_steps)
    delta_t = time[1] - time[0]

    model = JaxSchedulerModel(
        initial_state=psi_init_s,
        target_hamiltonian=target_hamiltonian_s,
        initial_hamiltonian=driver_hamiltonian_s,
        reference_hamiltonian=target_hamiltonian_s,
        tf=tau,
        number_of_parameters=NUMBER_PARAMETERS,
        nsteps=time_steps,
        type=SCHEDULE_TYPE,
        seed=seed,
        mode="annealing ansatz",
        random=True,
    )
    trainer = JaxTrainer(
        model, maxiter=MAXITER, tol=OPT_TOL, ftol=OPT_FTOL, gtol=OPT_GTOL, verbose=False
    )
    opt_results = trainer.run()
    h_driver, h_target = opt_results["h_driver"], opt_results["h_target"]

    # ── propagate the optimized schedule (sparse expm_multiply) ───────────────
    dim_s = psi_init_s.shape[0]
    psi = psi_init_s.copy()
    energy_per_time = np.zeros(time_steps)
    psi_history_s = np.zeros((time_steps, dim_s), dtype=complex)
    for i in range(time_steps):
        hamiltonian_t = (
            h_driver[i] * driver_hamiltonian_s + h_target[i] * target_hamiltonian_s
        )
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)
        psi_history_s[i] = psi
        energy_per_time[i] = (psi.conj() @ (hamiltonian_t @ psi)).real

    # ── magic / entanglement, subsampled (O(4^N) per call) ────────────────────
    idx_sub = np.arange(0, time_steps, STRIDE)
    magic_per_time = np.zeros(idx_sub.shape[0])
    entanglement_per_time = np.zeros(idx_sub.shape[0])
    for k, i in enumerate(idx_sub):
        state_full = sector.lift(psi_history_s[i])
        magic_per_time[k] = sre(state_full)
        entanglement_per_time[k] = entanglement_entropy.von_neumann(state_full)

    final_energy = float(energy_per_time[-1])
    elapsed = _time.time() - t0

    return {
        "tau": tau,
        "seed": seed,
        "time": time,
        "time_sub": time[idx_sub],
        "energy_per_time": energy_per_time,
        "magic_per_time": magic_per_time,
        "entanglement_per_time": entanglement_per_time,
        "final_energy": final_energy,
        "energy_error": final_energy - e0_target,
        "max_magic": float(np.max(magic_per_time)),
        "max_entanglement": float(np.max(entanglement_per_time)),
        "e0_target": e0_target,
        "success": bool(opt_results["success"]),
        "n_iterations": int(opt_results["n_iterations"]),
        "parameters": opt_results["parameters"],
        "h_driver": h_driver,
        "h_target": h_target,
        "elapsed_seconds": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _save_tau_results(tau: float, seed_results: dict, output_dir: Path):
    """seed_results: {seed: result_dict}, all seeds sharing the same tau."""
    seeds = sorted(seed_results.keys())
    stacked = {
        "tau": tau,
        "seeds": np.array(seeds),
        "time": seed_results[seeds[0]]["time"],
        "time_sub": seed_results[seeds[0]]["time_sub"],
        "energy_per_time": np.stack([seed_results[s]["energy_per_time"] for s in seeds]),
        "magic_per_time": np.stack([seed_results[s]["magic_per_time"] for s in seeds]),
        "entanglement_per_time": np.stack(
            [seed_results[s]["entanglement_per_time"] for s in seeds]
        ),
        "final_energy": np.array([seed_results[s]["final_energy"] for s in seeds]),
        "energy_error": np.array([seed_results[s]["energy_error"] for s in seeds]),
        "max_magic": np.array([seed_results[s]["max_magic"] for s in seeds]),
        "max_entanglement": np.array(
            [seed_results[s]["max_entanglement"] for s in seeds]
        ),
        "success": np.array([seed_results[s]["success"] for s in seeds]),
        "n_iterations": np.array([seed_results[s]["n_iterations"] for s in seeds]),
        "parameters": np.stack([seed_results[s]["parameters"] for s in seeds]),
        "h_driver": np.stack([seed_results[s]["h_driver"] for s in seeds]),
        "h_target": np.stack([seed_results[s]["h_target"] for s in seeds]),
        "e0_target": seed_results[seeds[0]]["e0_target"],
        # ── metadata ────────────────────────────────────────────────────────
        "n_qubits": N_QUBITS,
        "J": J,
        "JL": JL,
        "JR": JR,
        "number_of_parameters": NUMBER_PARAMETERS,
        "schedule_type": SCHEDULE_TYPE,
        "steps_per_tau": STEPS_PER_TAU,
        "maxiter": MAXITER,
        "stride": STRIDE,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"lzs_scan_tau{tau:07.2f}.npz"
    np.savez(path, **stacked)
    return path


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taus", type=float, nargs="+", default=DEFAULT_TAUS, help="tau values to scan"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="seeds per tau"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel worker processes",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="BLAS/XLA intra-op threads per worker process",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    tasks = [(tau, seed) for tau in args.taus for seed in args.seeds]
    print(
        f"Scanning {len(args.taus)} tau values x {len(args.seeds)} seeds "
        f"= {len(tasks)} runs, on {args.workers} workers "
        f"({args.threads_per_worker} thread(s) each)."
    )

    results_by_tau = {tau: {} for tau in args.taus}
    t_start = _time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(args.threads_per_worker,),
    ) as pool:
        futures = {
            pool.submit(run_single_seed, tau, seed): (tau, seed) for tau, seed in tasks
        }
        n_done = 0
        for future in as_completed(futures):
            tau, seed = futures[future]
            result = future.result()
            results_by_tau[tau][seed] = result
            n_done += 1
            print(
                f"[{n_done}/{len(tasks)}] tau={tau:<7g} seed={seed:<3d} "
                f"final_energy={result['final_energy']:.6f} "
                f"energy_error={result['energy_error']:.3e} "
                f"max_magic={result['max_magic']:.4f} "
                f"max_ent={result['max_entanglement']:.4f} "
                f"({result['elapsed_seconds']:.1f}s)"
            )

            # save as soon as every seed for this tau has finished
            if len(results_by_tau[tau]) == len(args.seeds):
                path = _save_tau_results(tau, results_by_tau[tau], args.output_dir)
                print(f"  -> saved {path}")

    print(f"\nDone in {_time.time() - t_start:.1f}s total.")


if __name__ == "__main__":
    main()
