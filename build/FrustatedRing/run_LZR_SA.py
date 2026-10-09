import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.parallel import best_of_seeds
from src.sparse_grape_method import SimulatedAnnealingTrainer, SparseGRAPEModel
from src.time_grid import make_time_grids, midpoint_evolution, subsample_indices
from src.utils import EntanglementEntropy, Z2SymmetricSector

start = time.perf_counter()
tag = "_SA"
T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; N=9,11,13 feasible for full 2^N exact diagonalization
J, JL, JR = 1.0, 0.5, 0.45

jij, hz = frustrated_ring_jij_hz(N, J, JL, JR)

nqubits = N
target_hamiltonian = get_longitudinal_hamiltonian(
    jij, hz
)  # sparse scipy matrix, full 2^N space
driver_hamiltonian = get_driver_hamiltonian(
    nqubits=nqubits
)  # sparse scipy matrix, full 2^N space


# The uniform superposition (driver ground state) is manifestly +1 under the
# global flip Pi = prod_i X_i, so annealing dynamics from this initial state
# stays confined to the +1 sector for all s in [0,1] (H(s) commutes with Pi
# throughout, since target has only ZZ terms and driver only X terms).
sector = Z2SymmetricSector(nqubits, sign=+1)

dim = 2**nqubits
psi_init_full = np.ones(dim, dtype=complex) / np.sqrt(dim)
assert sector.check_confined(
    psi_init_full
), "initial state is not confined to the +1 sector!"

target_hamiltonian_s = sector.project(
    target_hamiltonian
)  # sparse, dim_sector x dim_sector
driver_hamiltonian_s = sector.project(driver_hamiltonian)
psi_init_s = sector.project(psi_init_full)


# ── time evolution parameters ─────────────────────────────────────────────────
nlevels = 2
tau = T  # try a range of tau; the ring is expected to need LARGE tau
# for a linear ramp to reach the ground state (exponential
# slowdown at the AC) -- this is exactly the motivation for
# optimal control / LZS below.
time_steps = int(50 * tau)  # number of propagation steps
# midpoint rule (src/time_grid.py): the states live on `times`
# (time_steps + 1 points, 0..tau); the schedule that drives each step is
# evaluated at the cell midpoints `times_ctrl`
times, times_ctrl, delta_t = make_time_grids(tau, time_steps)

number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"


def build_model(seed=0, random=False):
    return SparseGRAPEModel(
        initial_state=psi_init_s,
        target_hamiltonian=target_hamiltonian_s,
        initial_hamiltonian=driver_hamiltonian_s,
        reference_hamiltonian=target_hamiltonian_s,
        tf=tau,
        number_of_parameters=number_parameters,
        nsteps=time_steps,
        type=type,
        seed=seed,
        random=random,
    )


def optimize_seed(seed):
    """One optimization from the random initial point of `seed`, run in a
    worker process (src/parallel.py). Returns only (energy, theta), all that
    is needed to keep the best one. verbose=False: the per-iteration output
    of many simultaneous optimizations would be unreadable (and the histories
    it stores are not used)."""
    model = build_model(seed=seed, random=True)

    trainer = SimulatedAnnealingTrainer(model, seed=seed, verbose=False)
    result = trainer.run()
    return result["energy"], result["parameters"]


# the guard is needed by the worker processes: they import this file again
# (everything above) and must not repeat what follows
if __name__ == "__main__":
    # imported here: the workers only optimize and do not need JAX
    from src.jax_utils import SREJax

    # best of 5 random initial points, one process per initial point,
    # spread over all the cores of the job
    best_seed, _, theta = best_of_seeds(optimize_seed, range(5))
    model = build_model()
    model.load(theta)

    # s(t) at the cell midpoints (what the propagator uses) and on the state
    # grid (instantaneous Hamiltonian of the observables, plots)
    schedule_ctrl = model.get_driving(grid="control")[1]
    schedule = model.get_driving(grid="state")[1]

    dim_s = driver_hamiltonian_s.shape[0]
    n_times = len(times)  # time_steps + 1 states: t = 0, ..., tau
    spectrum = np.zeros((n_times, nlevels))
    energy = np.zeros(n_times)
    probabilities = np.zeros((n_times, nlevels))
    psi_history_s = np.zeros((n_times, dim_s), dtype=complex)
    eigenstates_history_s = np.zeros((n_times, dim_s, nlevels), dtype=complex)

    sre = SREJax(n_qubits=nqubits - 1, batch_size=1000)
    entanglement_entropy = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)


    # psi is the state at times[i], i = 0..time_steps (i = 0: initial state)
    for i, psi in midpoint_evolution(
        psi_init_s,
        1 - schedule_ctrl,
        schedule_ctrl,
        delta_t,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    ):
        # instantaneous Hamiltonian at the time of the state, H(s(times[i]))
        hamiltonian_t = (1 - schedule[i]) * driver_hamiltonian_s + (
            schedule[i]
        ) * target_hamiltonian_s

        spectrum_t, eigenstates_t = eigsh(
            hamiltonian_t.astype(complex), which="SA", k=nlevels
        )
        order = np.argsort(spectrum_t)
        spectrum[i] = spectrum_t[order]
        eigenstates_raw = eigenstates_t[:, order].astype(complex)
        eigenstates_history_s[i] = eigenstates_raw

        probabilities[i] = (
            np.einsum("i,ia->a", psi.conj(), eigenstates_raw)
            * np.einsum("i,ia->a", psi.conj(), eigenstates_raw).conj()
        ).real
        energy[i] = np.real(np.vdot(psi, hamiltonian_t @ psi))
        psi_history_s[i] = psi

    e0 = spectrum[:, 0]
    e1 = spectrum[:, 1]
    gap = spectrum[:, 1] - spectrum[:, 0]
    p0 = probabilities[:, 0]
    p1 = probabilities[:, 1]

    magic = []
    entanglement = []

    # subsample if time_steps is large — SRE is O(4^N) per call
    stride = max(1, 10)

    idx_sub = subsample_indices(time_steps, stride)  # includes t = 0 and t = tau
    for i in tqdm(idx_sub):
        state_full = sector.lift(psi_history_s[i])
        magic.append(sre(psi_history_s[i]))
        entanglement.append(entanglement_entropy.von_neumann(state_full))


    time_sub = times[idx_sub]


    # formateo consistente de T para evitar problemas de precisión en el nombre
    T_str = str(T)

    nombre_archivo = f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_T={T_str}_LZR{tag}.npz"

    np.savez(
        nombre_archivo,
        T=np.array([T]),  # guardamos T explícitamente también, por seguridad
        seed=np.array([best_seed]),
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
        entanglement=entanglement,
    )

    end = time.perf_counter()

    elapsed = end - start
    print("Completed!! ")
    print(f"Guardado: {nombre_archivo}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
