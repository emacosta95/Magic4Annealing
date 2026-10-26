import sys
import time
from functools import partial

import numpy as np
from tqdm import tqdm

from src.free_fermions_grape_method import NambuGRAPEModel
from src.free_fermions_utils import NambuIsing1D
from src.parallel import available_cpus, best_of_seeds, make_pool
from src.sparse_grape_method import SparseGRAPETrainer
from src.time_grid import (
    make_time_grids,
    nambu_midpoint_evolution,
    subsample_indices,
)

start = time.perf_counter()

# usage: python run_LZR_FF_bounded_ws.py T0 N T_MIN T_MAX STEP
# T0 is the only T optimized from random initial points; every other T of
# range(T_MIN, T_MAX + 1, STEP) starts from the optimum of its neighbour.
# The random initial points of T0 are spread over all the cores of the job;
# then the two branches (T < T0 and T > T0) run in two processes at the same
# time.
T0 = int(sys.argv[1])
N = int(sys.argv[2])  # odd; cost is O(N^3) per time step, no 2^N limit
T_MIN = int(sys.argv[3])
T_MAX = int(sys.argv[4])
STEP = int(sys.argv[5])

# for the filename: bounded version + warm start (ws) along the list of T,
# and the T0 of the chain, so that chains seeded at different T0 do
# not overwrite each other
tag = f"_FF_bounded_ws_T0={T0}"

T_list = list(range(T_MIN, T_MAX + 1, STEP))
if T0 not in T_list:
    raise ValueError(
        f"T0={T0} is not in range(T_MIN={T_MIN}, T_MAX={T_MAX} + 1, "
        f"STEP={STEP})"
    )

J, JL, JR = 1.0, 0.5, 0.45

nqubits = N
# Nambu/BdG representation of the same ring as frustrated_ring_jij_hz, in the
# spin convention of src.free_fermions_utils (driver = -sum_i sz_i,
# target = -sum_i J_i sx_i sx_{i+1}). pbc=True fixes the even-parity sector,
# i.e. the Z2 +1 sector the spin-basis scripts project onto.
nambu = NambuIsing1D.frustrated_ring(N, J, JL, JR)

# driver ground state: same initial state NambuGRAPEModel uses internally
_, w_init = nambu.diagonalize(1.0, 0.0)


# ── time evolution parameters ─────────────────────────────────────────────────
nlevels = 2

number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"

# subsample if time_steps is large — Majorana sampling is O(N^4) per sample
stride = max(1, 10)
n_samples = 1000  # Majorana samples per SRE estimate (statistical error ~ 1/sqrt)


def build_model(T, seed=0, random=False):
    tau = T  # try a range of tau; the ring is expected to need LARGE tau
    # for a linear ramp to reach the ground state (exponential
    # slowdown at the AC) -- this is exactly the motivation for
    # optimal control / LZS below.
    time_steps = int(100 * tau)  # number of propagation steps
    return NambuGRAPEModel(
        nambu,
        tf=tau,
        number_of_parameters=number_parameters,
        nsteps=time_steps,
        type=type,
        seed=seed,
        random=random,
        bounds_opt=True,
    )


def optimize_seed(T, seed):
    """One optimization from the random initial point of `seed`, run in a
    worker process (src/parallel.py). Returns only (energy, theta), all that
    is needed to keep the best one. verbose=False: the per-iteration output
    of many simultaneous optimizations would be unreadable (and the histories
    it stores are not used)."""
    model = build_model(T, seed=seed, random=True)

    trainer = SparseGRAPETrainer(model, verbose=False)
    result = trainer.run()
    return result["energy"], result["parameters"]


def optimize_cold(T, n_workers):
    """Usual optimization: best of 200 random initial points, one process per
    initial point, n_workers at a time."""
    best_seed, _, theta = best_of_seeds(
        partial(optimize_seed, T), range(200), n_workers
    )

    model = build_model(T)
    model.load(theta)
    return model, theta, best_seed


def optimize_warm(T, theta_init):
    """Single optimization starting from the optimum of the neighbouring T
    (theta_init). What is passed is the fraction of the total time of each
    segment, not the raw durations: the ansatz only sees D / sum(D), so the
    initial schedule is the neighbour's one rescaled to the new T either way,
    but the raw durations of every T of the chain then start with sum = 1."""
    n_seg = 2 * number_parameters + 1  # raw durations: theta[:n_seg]
    theta_init = theta_init.copy()
    theta_init[:n_seg] /= theta_init[:n_seg].sum()

    model = build_model(T)
    model.load(theta_init)

    trainer = SparseGRAPETrainer(model, verbose=True)
    result = trainer.run()
    return model, result["parameters"]


def evaluate_and_save(T, model, theta, seed):
    """Time-resolved observables of the optimized schedule of `model` and
    write of the data file of this T. Everything allocated here (w1_history
    is the big one) is released on return."""
    tau = T
    time_steps = int(100 * tau)
    # midpoint rule (src/time_grid.py): the states live on `times`
    # (time_steps + 1 points, 0..tau); the schedule that drives each step is
    # evaluated at the cell midpoints `times_ctrl`
    times, times_ctrl, delta_t = make_time_grids(tau, time_steps)

    # s(t) at the cell midpoints (what the propagator uses) and on the state
    # grid (instantaneous Hamiltonian of the observables, plots)
    schedule_ctrl = model.get_driving(grid="control")[1]
    schedule = model.get_driving(grid="state")[1]

    n_times = len(times)  # time_steps + 1 states: t = 0, ..., tau
    idx_sub = subsample_indices(time_steps, stride)  # includes t = 0 and t = tau
    sub_position = {int(i): k for k, i in enumerate(idx_sub)}

    spectrum = np.zeros((n_times, nlevels))
    energy = np.zeros(n_times)
    probabilities = np.zeros((n_times, nlevels))
    # only the Bogoliubov vacuum W1 [2N, N] at the subsampled steps is kept
    w1_history = np.zeros((len(idx_sub), 2 * nqubits, nqubits), dtype=complex)

    # w is the Bogoliubov matrix at times[i], i = 0..time_steps (i = 0: w_init,
    # passed explicitly: the first control value is s(dt/2), not s = 0)
    for i, w in nambu_midpoint_evolution(
        nambu, w_init, 1 - schedule_ctrl, schedule_ctrl, delta_t
    ):
        # instantaneous Hamiltonian at the time of the state, H(s(times[i]))
        h_driver_t, h_target_t = 1 - schedule[i], schedule[i]
        w1 = w[:, :nqubits]

        # lowest physical levels (even sector) and their populations
        spectrum[i], probabilities[i], _ = nambu.level_probabilities(
            w1, h_driver_t, h_target_t, n_levels=nlevels
        )
        hamiltonian_t = nambu.hamiltonian(h_driver_t, h_target_t)
        energy[i] = np.real(np.trace(w1.conj().T @ hamiltonian_t @ w1))
        if i in sub_position:
            w1_history[sub_position[i]] = w1

    e0 = spectrum[:, 0]
    e1 = spectrum[:, 1]
    gap = spectrum[:, 1] - spectrum[:, 0]
    p0 = probabilities[:, 0]
    p1 = probabilities[:, 1]

    magic = []
    magic_filtered = []
    magic_err = []
    entanglement = []

    # base=np.e -> nats, the unit of the spin-basis scripts
    for k in tqdm(range(len(w1_history))):
        sre_k = nambu.sre(
            w1_history[k], alpha=2, n_samples=n_samples, seed=k, base=np.e
        )
        magic.append(sre_k["m_alpha"])
        magic_filtered.append(sre_k["m_alpha_filtered"])
        magic_err.append(sre_k["err"])
        entanglement.append(
            nambu.entanglement_entropy(w1_history[k], nqubits // 2, base=np.e)
        )

    time_sub = times[idx_sub]

    # formateo consistente de T para evitar problemas de precisión en el nombre
    T_str = str(T)

    nombre_archivo = f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_T={T_str}_LZR{tag}.npz"

    np.savez(
        nombre_archivo,
        T=np.array([T]),  # guardamos T explícitamente también, por seguridad
        T0=np.array([T0]),  # T where the warm-start chain was seeded
        seed=np.array([seed]),  # best random seed of the T0 optimization
        theta=np.array([theta]),
        dt=np.array([delta_t]),
        nsteps=np.array([time_steps]),
        times=times,  # state grid: every time-resolved observable below
        times_ctrl=times_ctrl,  # cell midpoints: where schedule_ctrl is sampled
        evo_energy=energy,
        e0=e0,
        e1=e1,
        gap=gap,
        schedule=schedule,  # s on `times` (plots, spectrum)
        schedule_ctrl=schedule_ctrl,  # s on `times_ctrl` (what was propagated)
        p0=p0,
        p1=p1,
        time_sub=time_sub,
        magic=magic,
        magic_filtered=magic_filtered,
        magic_err=magic_err,
        entanglement=entanglement,
    )
    print(f"Guardado: {nombre_archivo}")


def run_warm(T, theta_init, seed):
    """Warm-started T: optimize, write its file and return only theta, the
    one thing the next T of the branch needs."""
    model, theta = optimize_warm(T, theta_init)
    evaluate_and_save(T, model, theta, seed)
    return theta


def run_branch(T_branch, theta, seed):
    """One branch of the warm start: the T of T_branch in the given order,
    each one starting from the optimum of the previous one (theta: optimum
    of T0). All that is kept between two T is that theta."""
    for T in T_branch:
        theta = run_warm(T, theta, seed)


# the guard is needed by the worker processes: they import this file again
# (everything above) and must not repeat what follows
if __name__ == "__main__":
    n_cpus = available_cpus()  # cores given to the job

    # ── T0: usual optimization ────────────────────────────────────────────────
    model, theta_0, best_seed = optimize_cold(T0, n_cpus)
    evaluate_and_save(T0, model, theta_0, best_seed)
    del model

    # ── warm start: T0 - STEP, T0 - 2*STEP, ... and T0 + STEP, ... ────────────
    # the two branches only share theta_0, so each one runs in its own process
    i_0 = T_list.index(T0)
    branches = [b for b in (T_list[:i_0][::-1], T_list[i_0 + 1 :]) if b]

    with make_pool(2, blas_threads=max(1, n_cpus // 2)) as pool:
        futures = [pool.submit(run_branch, b, theta_0, best_seed) for b in branches]
        for future in futures:
            future.result()  # re-raises here whatever failed in the branch

    end = time.perf_counter()

    elapsed = end - start
    print("Completed!! ")
    print(f"Elapsed time: {elapsed:.2f} seconds")
