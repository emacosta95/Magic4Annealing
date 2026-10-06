import sys
import time

import numpy as np
from tqdm import trange

from src.free_fermions_grape_method import NambuGRAPEModel
from src.free_fermions_utils import NambuIsing1D
from src.sparse_grape_method import SparseGRAPETrainer

start = time.perf_counter()
tag = "_FF"
T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; cost is O(N^3) per time step, no 2^N limit
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
time_steps = int(10 * tau)
times = np.linspace(0, tau, time_steps)
delta_t = times[1] - times[0]

number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"

best_result = None
for i in range(50):
    model_i = NambuGRAPEModel(
        nambu,
        tf=tau,
        number_of_parameters=number_parameters,
        nsteps=time_steps,
        type=type,
        seed=i,
        random=True,
    )

    trainer = SparseGRAPETrainer(model_i, verbose=True)
    result = trainer.run()
    if best_result is None or result["energy"] < best_result["energy"]:
        best_result = result
        model = model_i
        best_seed = i

h_driver, h_target = model.get_driving()
schedule = h_target

theta = best_result["parameters"]

# subsample if time_steps is large — Majorana sampling is O(N^4) per sample
stride = max(1, 10)
n_samples = 1000  # Majorana samples per SRE estimate (statistical error ~ 1/sqrt)

spectrum = np.zeros((time_steps, nlevels))
energy = np.zeros(time_steps)
probabilities = np.zeros((time_steps, nlevels))
# only the Bogoliubov vacuum W1 [2N, N] at the subsampled steps is kept
w1_history = np.zeros(
    (len(range(0, time_steps, stride)), 2 * nqubits, nqubits), dtype=complex
)

w = w_init
for i, t in enumerate(times):
    h_driver_t, h_target_t = 1 - schedule[i], schedule[i]
    w, _ = nambu.evolve([h_driver_t], [h_target_t], delta_t, w0=w)
    w1 = w[:, :nqubits]

    # lowest physical levels (even sector) and their populations
    spectrum[i], probabilities[i], _ = nambu.level_probabilities(
        w1, h_driver_t, h_target_t, n_levels=nlevels
    )
    hamiltonian_t = nambu.hamiltonian(h_driver_t, h_target_t)
    energy[i] = np.real(np.trace(w1.conj().T @ hamiltonian_t @ w1))
    if i % stride == 0:
        w1_history[i // stride] = w1

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
for k in trange(len(w1_history)):
    sre_k = nambu.sre(
        w1_history[k], alpha=2, n_samples=n_samples, seed=k, base=np.e
    )
    magic.append(sre_k["m_alpha"])
    magic_filtered.append(sre_k["m_alpha_filtered"])
    magic_err.append(sre_k["err"])
    entanglement.append(
        nambu.entanglement_entropy(w1_history[k], nqubits // 2, base=np.e)
    )


time_sub = times[::stride]


# formateo consistente de T para evitar problemas de precisión en el nombre
T_str = str(T)

nombre_archivo = (
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_T={T_str}_LZR{tag}.npz"
)

np.savez(
    nombre_archivo,
    T=np.array([T]),  # guardamos T explícitamente también, por seguridad
    seed=np.array([best_seed]),
    theta=np.array([theta]),
    times=times,
    evo_energy=energy,
    e0=e0,
    e1=e1,
    gap=gap,
    schedule=schedule,
    p0=p0,
    p1=p1,
    time_sub=time_sub,
    magic=magic,
    magic_filtered=magic_filtered,
    magic_err=magic_err,
    entanglement=entanglement,
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
