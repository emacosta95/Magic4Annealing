import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply
from tqdm import trange

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.free_fermions_utils import NambuIsing1D
from src.free_fermions_grape_method import NambuGRAPEModel
from src.sparse_grape_method import SparseGRAPETrainer


def report(label, w, h_drv, h_tgt):
    """Final-time diagnostics on W1 = w[:, :l]."""
    w1 = w[:, :N]
    e_res = nambu.residual_energy(w1, h_drv[-1], h_tgt[-1])
    _, p, _ = nambu.level_probabilities(w1, h_drv[-1], h_tgt[-1], n_levels=4)
    s_half = nambu.entanglement_entropy(w1, N // 2)
    m2 = nambu.sre(w1, alpha=2, n_samples=1000)
    print(
        f"{label:>10} | E_res={e_res:.3e}  P_gs={p[0]:.4f}  P_1..3={np.round(p[1:],4)}"
        f"  S_L/2={s_half:.3f}  M2={m2['m_alpha']:.3f}±{m2['err']:.3f}"
    )


start = time.perf_counter()
tag = "_bounded_FF"

T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; N=9,11,13 feasible for full 2^N exact diagonalization
J, JL, JR = 1.0, 0.5, 0.45

nambu = NambuIsing1D.frustrated_ring(N, J=J, JL=JL, JR=JR)

nqubits = N

# ── time evolution parameters ─────────────────────────────────────────────────
nlevels = 5
tau = T  # try a range of tau; the ring is expected to need LARGE tau
# for a linear ramp to reach the ground state (exponential
# slowdown at the AC) -- this is exactly the motivation for
# optimal control / LZS below.
time_steps = int(10 * tau)
times = np.linspace(0, tau, time_steps)
delta_t = times[1] - times[0]
maxit = 10**3

_, w0 = nambu.diagonalize(1.0, 0.0)  # driver ground state (all up)


number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"

best_result = None
for i in range(2):
    model_i = NambuGRAPEModel(
        nambu,
        tf=tau,
        number_of_parameters=number_parameters,
        nsteps=time_steps,
        type=type,
        seed=i,
        random=True,
        bounds_opt=True,
    )

    trainer = SparseGRAPETrainer(model_i, maxiter=maxit, verbose=True)
    result = trainer.run()
    if best_result is None or result["energy"] < best_result["energy"]:
        best_result = result
        model = model_i
        best_seed = i

print(
    f"GRAPE: {best_result['n_iterations']} it, {time.time()-start:.1f}s, E={best_result['energy']:.6f}, seed={best_seed}"
)
report(
    "GRAPE-LZS", best_result["psi"], best_result["h_driver"], best_result["h_target"]
)
h_driver = best_result["h_driver"]
h_target = best_result["h_target"]
schedule = h_target

wlist = nambu.evolve(h_driver, h_target, dt=delta_t, w0=w0, store_every=1)

spectrum = np.zeros((time_steps, nlevels))
probabilities = np.zeros((time_steps, nlevels))
magic = []
entanglement = []
for i in range(len(h_driver)):
    spectrum_t, probs_t, _ = nambu.level_probabilities(
        wlist[i], h_driver[i], h_target[i], n_levels=nlevels
    )
    spectrum[i] = spectrum_t
    probabilities[i] = probs_t
    entanglement.append(nambu.entanglement_entropy(wlist[i], N // 2))
    magic.append(nambu.sre(wlist[i], alpha=2))


e0 = spectrum[:, 0]
e1 = spectrum[:, 1]
e2 = spectrum[:, 2]
e3 = spectrum[:, 3]
e4 = spectrum[:, 4]
p0 = probabilities[:, 0]
p1 = probabilities[:, 1]
p2 = probabilities[:, 2]
p3 = probabilities[:, 3]
p4 = probabilities[:, 4]
gap = e1 - e0


dim_s = driver_hamiltonian_s.shape[0]
psi = psi_init_s.copy()
theta = best_result["parameters"]
spectrum = np.zeros((time_steps, nlevels))
energy = np.zeros(time_steps)
probabilities = np.zeros((time_steps, nlevels))
psi_history_s = np.zeros((time_steps, dim_s), dtype=complex)
eigenstates_history_s = np.zeros((time_steps, dim_s, nlevels), dtype=complex)


for i, t in enumerate(times):
    hamiltonian_t = (1 - schedule[i]) * driver_hamiltonian_s + (
        schedule[i]
    ) * target_hamiltonian_s
    psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)

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
    e2=e2,
    e3=e3,
    e4=e4,
    gap=gap,
    schedule=schedule,
    p0=p0,
    p1=p1,
    p2=p2,
    p3=p3,
    p4=p4,
    time_sub=time_sub,
    magic=magic,
    entanglement=entanglement,
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
