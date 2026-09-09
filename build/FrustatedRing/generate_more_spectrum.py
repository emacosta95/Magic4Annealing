import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply
from src.annealing_utils import (
    get_longitudinal_hamiltonian,
    get_driver_hamiltonian,
)
from src.sparse_grape_method import SparseGRAPEModel, SimulatedAnnealingTrainer

from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.utils import Z2SymmetricSector
from src.jax_utils import SREJax
from src.utils import EntanglementEntropy
from tqdm import trange
import sys
import time
import re

start = time.perf_counter()
N = int(sys.argv[1])  # odd; N=9,11,13 feasible for full 2^N exact diagonalization
tag = sys.argv[2]  # tag for the output file
nlevels = int(sys.argv[3])  # number of levels to compute in the spectrum

filename = f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR" + tag + ".npz"
data = dict(np.load(filename, allow_pickle=True))

Tlist = sorted(set(int(re.match(r"T=(\d+)_", k).group(1)) for k in data.keys()))

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

for Ti in Tlist:
    times = data[f"T={Ti}_times"]
    delta_t = times[1] - times[0]
    time_steps = len(times)
    probabilities = np.zeros((time_steps, nlevels))
    spectrum = np.zeros((time_steps, nlevels))
    psi = psi_init_s.copy()
    for i, t in enumerate(times):
        hamiltonian_t = (1 - data[f"T={Ti}_schedule"][i]) * driver_hamiltonian_s + (
            data[f"T={Ti}_schedule"][i]
        ) * target_hamiltonian_s
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)

        spectrum_t, eigenstates_t = eigsh(
            hamiltonian_t.astype(complex), which="SA", k=nlevels
        )

        order = np.argsort(spectrum_t)
        spectrum[i] = spectrum_t[order]
        eigenstates_raw = eigenstates_t[:, order].astype(complex)

        probabilities[i] = (
            np.einsum("i,ia->a", psi.conj(), eigenstates_raw)
            * np.einsum("i,ia->a", psi.conj(), eigenstates_raw).conj()
        ).real

    for i in range(nlevels):
        data[f"T={Ti}_p{i}"] = probabilities[:, i]
        data[f"T={Ti}_e{i}"] = spectrum[:, i]

filename_tmp = (
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR" + tag + "_tmp.npz"
)
np.savez(filename_tmp, **data)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {filename}")
print(f"Elapsed time: {elapsed:.2f} seconds")
