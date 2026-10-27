import sys
import time

import numpy as np
from tqdm import tqdm

from src.free_fermions_grape_method import NambuGRAPEModel
from src.free_fermions_utils import NambuIsing1D
from src.sparse_grape_method import SparseGRAPETrainer
from src.time_grid import (
    make_time_grids,
    nambu_midpoint_evolution,
    subsample_indices,
)

start = time.perf_counter()

# usage: python run_LZR_seed_scan_FF_bounded.py T N SEED
# Same as run_LZR_FF_bounded.py, but instead of the best of 200 random initial
# points, a single optimization from the random initial point of SEED. One job
# per seed (e.g. a SLURM array over SEED) scans the optimizer's minima at a
# fixed T; seeds 0..199 are the same initial points best_of_seeds compares in
# run_LZR_FF_bounded.py.
T = int(sys.argv[1])
N = int(sys.argv[2])  # odd; cost is O(N^3) per time step, no 2^N limit
SEED = int(sys.argv[3])  # seed of the random initial point

tag = "_FF_bounded"  # for the filename: the optimization is the bounded FF one

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


# ── optimization from the initial point of SEED ──────────────────────────────
model = NambuGRAPEModel(
    nambu,
    tf=tau,
    number_of_parameters=number_parameters,
    nsteps=time_steps,
    type=type,
    seed=SEED,
    random=True,
    bounds_opt=True,
)
trainer = SparseGRAPETrainer(model, verbose=True)
result = trainer.run()
theta = result["parameters"]
model.load(theta)

# s(t) at the cell midpoints (what the propagator uses) and on the state
# grid (instantaneous Hamiltonian of the observables, plots)
schedule_ctrl = model.get_driving(grid="control")[1]
schedule = model.get_driving(grid="state")[1]

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

magic = []
magic_filtered = []
magic_err = []
entanglement = []

# base=np.e -> nats, the unit of the spin-basis scripts
for k in tqdm(range(len(w1_history))):
    sre_k = nambu.sre(w1_history[k], alpha=2, n_samples=n_samples, seed=k, base=np.e)
    magic.append(sre_k["m_alpha"])
    magic_filtered.append(sre_k["m_alpha_filtered"])
    magic_err.append(sre_k["err"])
    entanglement.append(
        nambu.entanglement_entropy(w1_history[k], nqubits // 2, base=np.e)
    )

time_sub = times[idx_sub]

# position of the avoided crossing s* (minimum of the gap of H(s), a property
# of the Hamiltonian only: the same for every seed and T), computed only in
# the job of seed 0. Coarse grid over [0, 1], then a fine one around its
# minimum, since the gap minimum is very narrow for large N.
extra = {}
if SEED == 0:
    s_grid = np.linspace(0.0, 1.0, 2001)
    for _ in range(2):
        levels = nambu.spectrum_along_schedule(1 - s_grid, s_grid, n_levels=2)
        static_gap = levels[:, 1] - levels[:, 0]
        i_min = np.argmin(static_gap)
        s_star, gap_min = s_grid[i_min], static_gap[i_min]
        ds = s_grid[1] - s_grid[0]
        s_grid = np.linspace(max(s_star - ds, 0.0), min(s_star + ds, 1.0), 2001)
    extra["s_star"] = np.array([s_star])
    extra["gap_min"] = np.array([gap_min])
    print(f"s* = {extra['s_star'][0]:.6f} (gap min = {extra['gap_min'][0]:.4g})")

# formateo consistente de T para evitar problemas de precisión en el nombre
T_str = str(T)

# prefix QuantumResourcesSeedScan (not QuantumResourcesvsT): the glob of
# merge_LZR.py must not pick these files up
nombre_archivo = (
    f"../../generated/FrustatedRing/QuantumResourcesSeedScan_N={N}_T={T_str}"
    f"_seed={SEED}_LZR{tag}.npz"
)

np.savez(
    nombre_archivo,
    T=np.array([T]),  # guardamos T explícitamente también, por seguridad
    seed=np.array([SEED]),
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
    **extra,  # s_star, gap_min: only in the file of seed 0
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
