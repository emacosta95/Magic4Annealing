"""
Process-level parallelism for the build scripts.

The optimizations from different random initial points are independent, so
they are spread over the cores of the job: one process per initial point.

The workers are started with "spawn" (the only start method on Windows, and
the one that lets each worker read its own BLAS thread count), i.e. they are
new interpreters that import the calling script again. So, in the script:

  - the function handed to the pool must be defined at module level;
  - everything that must run only once (launching the pool, the evaluation
    of the best schedule, the write of the data file) must be under
    `if __name__ == "__main__":`.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def available_cpus():
    """Cores given to this process (the ones of the job under SLURM), not
    the ones of the whole node."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


@contextmanager
def make_pool(n_workers, blas_threads):
    """Pool of n_workers processes with blas_threads BLAS threads each, so
    that they do not open one thread per core each. The workers are new
    interpreters ("spawn"), so they read these variables when they import
    numpy; the environment of the calling process is restored on exit."""
    saved = {var: os.environ.get(var) for var in _THREAD_VARS}
    for var in _THREAD_VARS:
        os.environ[var] = str(blas_threads)
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            yield pool
    finally:
        for var, value in saved.items():
            if value is None:
                del os.environ[var]
            else:
                os.environ[var] = value


def best_of_seeds(optimize_seed, seeds, n_workers=None):
    """Runs optimize_seed(seed) -> (energy, theta) for every seed, one process
    per seed, n_workers at a time (default: all the cores of the job), and
    returns (seed, energy, theta) of the lowest energy.

    Every worker holds its own model, so the memory of the job is n_workers
    times the one of a single optimization: pass n_workers to cap it."""
    seeds = list(seeds)
    if n_workers is None:
        n_workers = available_cpus()

    best = None
    with make_pool(min(n_workers, len(seeds)), blas_threads=1) as pool:
        # map returns the results in the order of the seeds, so the best one
        # (lowest seed in case of a tie) is the one a serial loop would keep
        for seed, (energy, theta) in zip(seeds, pool.map(optimize_seed, seeds)):
            print(f"seed {seed}: energy = {energy:.6f}", flush=True)
            if best is None or energy < best[1]:
                best = (seed, energy, theta)
    return best
