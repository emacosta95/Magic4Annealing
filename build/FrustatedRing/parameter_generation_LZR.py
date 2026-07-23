import numpy as np
from src.annealing_utils import (
    get_longitudinal_hamiltonian,
    get_driver_hamiltonian,
)
from src.sparse_grape_method import SparseGRAPEModel, SparseGRAPETrainer

from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.utils import Z2SymmetricSector

import time

start = time.perf_counter()

T = 120

N = 7  # odd; N=9,11,13 feasible for full 2^N exact diagonalization
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
time_steps = int(10 * tau)
times = np.linspace(0, tau, time_steps)
delta_t = times[1] - times[0]

number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"
seed_list = list(range(20))
theta_list = []
for i in seed_list:
    model_i = SparseGRAPEModel(
        initial_state=psi_init_s,
        target_hamiltonian=target_hamiltonian_s,
        initial_hamiltonian=driver_hamiltonian_s,
        reference_hamiltonian=target_hamiltonian_s,
        tf=tau,
        number_of_parameters=number_parameters,
        nsteps=time_steps,
        type=type,
        seed=i,
        random=True,
    )

    trainer = SparseGRAPETrainer(model_i, verbose=True)
    result = trainer.run()
    theta_list.append(result.parameters)


# formateo consistente de T para evitar problemas de precisión en el nombre
T_str = str(T)

nombre_archivo = f"../../generated/FrustatedRing/ParametersLZR_T={T_str}_N={N}.npz"

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
