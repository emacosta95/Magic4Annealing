import sys
import time

import numpy as np

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.fidelity_robustness import FidelityRobustness
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.time_grid import schedules_from_saved

start = time.perf_counter()
T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; the augmented system has (3N + 1) 2^N components
# tag of the optimized schedules to read: "_bounded", "_bounded_SA", "_FF", ...
tag = sys.argv[3] if len(sys.argv) > 3 else ""
J, JL, JR = 1.0, 0.5, 0.45

jij, hz = frustrated_ring_jij_hz(N, J, JL, JR)

nqubits = N
# full 2^N space, nothing projected: sigma^y and sigma^z fields leave the
# Z2 +1 sector the other build scripts work in
target_hamiltonian = get_longitudinal_hamiltonian(jij, hz)
driver_hamiltonian = get_driver_hamiltonian(nqubits=nqubits)

dim = 2**nqubits
psi_init_full = np.ones(dim, dtype=complex) / np.sqrt(dim)

# ── schedules ─────────────────────────────────────────────────────────────────
data = np.load(f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR{tag}.npz")
# s(t) at the cell midpoints of the optimized schedule (old and new files)
saved = schedules_from_saved(data, prefix=f"T={T}_")
times_ctrl = saved["times_ctrl"]
schedule_ctrl = saved["schedule_ctrl"]
# linear schedule s = t / T on the same control grid
schedule_ctrl_linear = times_ctrl / T

# ── robustness matrices ───────────────────────────────────────────────────────
# G (3N x 3N) with respect to the local fields lam_k sigma^a_i, ordered by
# axis; each one is relative to the unperturbed final state of its own schedule
robustness = FidelityRobustness(
    nqubits, driver_hamiltonian, target_hamiltonian, psi_init_full, schedule_ctrl, T
)
robustness.compute()

robustness_linear = FidelityRobustness(
    nqubits,
    driver_hamiltonian,
    target_hamiltonian,
    psi_init_full,
    schedule_ctrl_linear,
    T,
)
robustness_linear.compute()

print(f"lambda* LZR{tag} = {robustness.lambda_star:.4g}")
print(f"lambda* linear = {robustness_linear.lambda_star:.4g}")


# formateo consistente de T para evitar problemas de precisión en el nombre
T_str = str(T)

nombre_archivo = (
    f"../../generated/FrustatedRing/FidelityRobustnessvsT_N={N}_T={T_str}_LZR{tag}.npz"
)

np.savez(
    nombre_archivo,
    T=np.array([T]),  # guardamos T explícitamente también, por seguridad
    dt=np.array([saved["dt"]]),
    nsteps=np.array([saved["nsteps"]]),
    labels=np.array(robustness.labels),  # order of the rows/columns of G
    space=np.array(["full"]),
    times_ctrl=times_ctrl,  # cell midpoints: where the schedules are sampled
    schedule_ctrl=schedule_ctrl,  # s on `times_ctrl` (what was propagated)
    G=robustness.G,  # complex; the fidelity uses Re G
    cov=robustness.cov,  # G / T^2
    lambda_star=np.array([robustness.lambda_star]),
    schedule_ctrl_linear=schedule_ctrl_linear,
    G_linear=robustness_linear.G,
    cov_linear=robustness_linear.cov,
    lambda_star_linear=np.array([robustness_linear.lambda_star]),
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
