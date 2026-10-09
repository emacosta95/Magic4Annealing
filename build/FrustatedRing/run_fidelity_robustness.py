import sys
import time

import numpy as np

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.fidelity_robustness import AXES, REFERENCES, FidelityRobustness
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.time_grid import schedules_from_saved

start = time.perf_counter()
T = int(sys.argv[1])

N = int(sys.argv[2])  # odd; the augmented system has (3N + 1) 2^N components
# tag of the optimized schedules to read: "_bounded", "_bounded_SA", "_FF", ...
tag = sys.argv[3] if len(sys.argv) > 3 else ""
J, JL, JR = 1.0, 0.5, 0.45

# exact fidelity: global field lambda * sum_i sigma^a_i on this grid of lambda
# for each axis a (one evolution per value: the cost is linear in n_lambdas)
lambda_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)}
n_lambdas = 101

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

# ── exact fidelities ──────────────────────────────────────────────────────────
lambdas = np.array([np.linspace(*lambda_range[axis], n_lambdas) for axis in AXES])


def exact_fidelities(robustness):
    """{reference: (3, n_lambdas)}: population of the exact perturbed final
    state in each reference subspace, for the global field along each axis."""
    fidelities = {reference: np.zeros(lambdas.shape) for reference in REFERENCES}
    for a in range(len(AXES)):
        # lam = [x_0..x_{N-1}, y_0..y_{N-1}, z_0..z_{N-1}]: lambda on the N
        # entries of this axis, 0 on the other 2N
        lam = np.zeros((n_lambdas, 3, nqubits))
        lam[:, a, :] = lambdas[a][:, None]
        states = robustness.final_states(lam)
        for reference in REFERENCES:
            fidelities[reference][a] = robustness.population(states, reference)
    return fidelities


fidelities = exact_fidelities(robustness)
fidelities_linear = exact_fidelities(robustness_linear)

print(
    f"ground-space population at lambda = 0: LZR{tag} "
    f"{robustness.population(robustness.psi_final):.6f}, linear "
    f"{robustness_linear.population(robustness_linear.psi_final):.6f}"
)


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
    # exact fidelity under lambda * sum_i sigma^a_i: one row per axis
    axes=np.array(AXES),
    lambdas=lambdas,  # (3, n_lambdas)
    **{f"fidelity_{ref}": fidelities[ref] for ref in REFERENCES},
    **{f"fidelity_{ref}_linear": fidelities_linear[ref] for ref in REFERENCES},
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
