"""Optima of 10 seeds (free fermions, bounded) for the landscape scripts. See ../README.md."""

import os
import sys
import time

import numpy as np

from src.free_fermions_grape_method import NambuGRAPEModel
from src.free_fermions_utils import NambuIsing1D
from src.parallel import available_cpus, make_pool
from src.sparse_grape_method import SparseGRAPETrainer
from src.time_grid import make_time_grids

start = time.perf_counter()

T = int(sys.argv[1])  # total annealing time, in units of 1/J

N = int(sys.argv[2])  # odd; cost is O(N^3) per time step, no 2^N limit
J, JL, JR = 1.0, 0.5, 0.45

# Nambu/BdG representation of the same ring as frustrated_ring_jij_hz, in the
# spin convention of src.free_fermions_utils (driver = -sum_i sz_i,
# target = -sum_i J_i sx_i sx_{i+1}). pbc=True fixes the even-parity sector,
# i.e. the Z2 +1 sector the spin-basis scripts project onto.
nambu = NambuIsing1D.frustrated_ring(N, J, JL, JR)


# ── time evolution parameters ─────────────────────────────────────────────────
tau = T  # try a range of tau; the ring is expected to need LARGE tau
# for a linear ramp to reach the ground state (exponential
# slowdown at the AC) -- this is exactly the motivation for
# optimal control / LZS below.
time_steps = int(100 * tau)  # number of propagation steps
# midpoint rule (src/time_grid.py): the states live on `times`
# (time_steps + 1 points, 0..tau); the schedule that drives each step is
# evaluated at the cell midpoints `times_ctrl`
times, times_ctrl, delta_t = make_time_grids(tau, time_steps)

number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"
seed_list = list(range(10))


def optimize_seed(seed):
    """One optimization from the random initial point of `seed`, run in a
    worker process (src/parallel.py). Returns (energy, theta). verbose=False:
    the per-iteration output of several simultaneous optimizations would be
    unreadable."""
    model = NambuGRAPEModel(
        nambu,
        tf=tau,
        number_of_parameters=number_parameters,
        nsteps=time_steps,
        type=type,
        seed=seed,
        random=True,
        bounds_opt=True,
    )

    trainer = SparseGRAPETrainer(model, verbose=False)
    result = trainer.run()
    return result["energy"], result["parameters"]


# the guard is needed by the worker processes: they import this file again
# (everything above) and must not repeat what follows
if __name__ == "__main__":
    # all the optima are kept (not only the best one): one process per seed,
    # spread over the cores of the job
    n_workers = min(available_cpus(), len(seed_list))
    with make_pool(n_workers, blas_threads=1) as pool:
        results = list(pool.map(optimize_seed, seed_list))

    theta_list = []
    for seed, (energy, theta) in zip(seed_list, results):
        print(f"seed {seed}: energy = {energy:.6f}", flush=True)
        theta_list.append(theta)

    # formateo consistente de T para evitar problemas de precisión en el nombre
    T_str = str(T)

    nombre_archivo = f"../../../generated/FrustatedRing/ParametersLZR/ParametersLZR_T={T_str}_N={N}.npz"

    os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
    np.savez(
        nombre_archivo,
        N=np.array([N]),
        T=np.array([T]),  # guardamos T explícitamente también, por seguridad
        seeds=seed_list,
        theta_list=np.array(theta_list),
    )

    end = time.perf_counter()

    elapsed = end - start
    print("Completed!! ")
    print(f"Guardado: {nombre_archivo}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
