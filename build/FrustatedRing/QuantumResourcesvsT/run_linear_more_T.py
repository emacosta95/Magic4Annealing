"""Linear schedule s = t/T, free fermions (same as run_linear.py, output _linear_more_T). See ../README.md."""

import os
import sys
import time

import numpy as np
from tqdm import tqdm

from src.free_fermions_utils import NambuIsing1D
from src.time_grid import (
    make_time_grids,
    nambu_midpoint_evolution,
    subsample_indices,
)

start = time.perf_counter()

T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; cost is O(N^3) per time step, no 2^N limit
J, JL, JR = 1.0, 0.5, 0.45

nqubits = N
# Nambu/BdG representation of the same ring as frustrated_ring_jij_hz, in the
# spin convention of src.free_fermions_utils (driver = -sum_i sz_i,
# target = -sum_i J_i sx_i sx_{i+1}). pbc=True fixes the even-parity sector,
# i.e. the Z2 +1 sector the spin-basis scripts project onto.
nambu = NambuIsing1D.frustrated_ring(N, J, JL, JR)

# driver ground state: initial state of the anneal
_, w_init = nambu.diagonalize(1.0, 0.0)


# ── time evolution parameters ─────────────────────────────────────────────────
nlevels = 2
tau = T  # try a range of tau; the ring is expected to need LARGE tau
# for a linear ramp to reach the ground state (exponential
# slowdown at the AC) -- this is exactly the motivation for
# optimal control / LZS below.
time_steps = int(100 * tau)  # number of propagation steps
# midpoint rule (src/time_grid.py): the states live on `times`
# (time_steps + 1 points, 0..tau); the schedule that drives each step is
# evaluated at the cell midpoints `times_ctrl`
times, times_ctrl, delta_t = make_time_grids(tau, time_steps)


# linear schedule s = t / tau, evaluated analytically on both grids
schedule = times / tau  # state grid (observables)
schedule_ctrl = times_ctrl / tau  # cell midpoints (propagation)

# subsample if time_steps is large — Majorana sampling is O(N^4) per sample
stride = max(1, 10)
n_samples = 1000  # Majorana samples per SRE estimate (statistical error ~ 1/sqrt)

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

# instantaneous physical ground state at the subsampled steps, as a
# Bogoliubov vacuum W1 [2N, N] like the evolved state
gs_history, _, _ = nambu.instantaneous_states(
    1 - schedule[idx_sub], schedule[idx_sub], level=0
)

magic = []
magic_filtered = []
magic_err = []
magic_gs_level = []
magic_gs_level_filtered = []
magic_gs_level_err = []
entanglement = []
entanglement_gs_level = []

# base=np.e -> nats, the unit of the spin-basis scripts
n_sub = len(idx_sub)
for k in tqdm(range(n_sub)):
    sre_k = nambu.sre(w1_history[k], alpha=2, n_samples=n_samples, seed=k, base=np.e)
    magic.append(sre_k["m_alpha"])
    magic_filtered.append(sre_k["m_alpha_filtered"])
    magic_err.append(sre_k["err"])
    # independent samples from those of the evolved state
    sre_gs_k = nambu.sre(
        gs_history[k], alpha=2, n_samples=n_samples, seed=n_sub + k, base=np.e
    )
    magic_gs_level.append(sre_gs_k["m_alpha"])
    magic_gs_level_filtered.append(sre_gs_k["m_alpha_filtered"])
    magic_gs_level_err.append(sre_gs_k["err"])
    entanglement.append(
        nambu.entanglement_entropy(w1_history[k], nqubits // 2, base=np.e)
    )
    entanglement_gs_level.append(
        nambu.entanglement_entropy(gs_history[k], nqubits // 2, base=np.e)
    )

time_sub = times[idx_sub]


# formateo consistente de T para evitar problemas de precisión en el nombre
T_str = str(T)

nombre_archivo = f"../../../generated/FrustatedRing/QuantumResourcesvsT/QuantumResourcesvsT_N={N}_T={T_str}_linear_more_T.npz"

os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
np.savez(
    nombre_archivo,
    T=np.array([T]),  # guardamos T explícitamente también, por seguridad
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
    magic_gs_level=magic_gs_level,
    magic_gs_level_filtered=magic_gs_level_filtered,
    magic_gs_level_err=magic_gs_level_err,
    entanglement_gs_level=entanglement_gs_level,
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
