import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply
from src.annealing_utils import (
    get_longitudinal_hamiltonian,
    get_driver_hamiltonian,
)

from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.utils import Z2SymmetricSector
from src.jax_utils import SREJax
from src.utils import EntanglementEntropy
from tqdm import trange
import sys
import time

start = time.perf_counter()

T = int(sys.argv[1])

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
time_steps = int(100 * tau)
times = np.linspace(0, tau, time_steps)
delta_t = times[1] - times[0]


dim_s = driver_hamiltonian_s.shape[0]
psi = psi_init_s.copy()

spectrum = np.zeros((time_steps, nlevels))
energy = np.zeros(time_steps)
schedule = np.zeros(time_steps)
probabilities = np.zeros((time_steps, nlevels))
psi_history_s = np.zeros((time_steps, dim_s), dtype=complex)
eigenstates_history_s = np.zeros((time_steps, dim_s, nlevels), dtype=complex)

sre = SREJax(n_qubits=nqubits - 1, batch_size=1000)
entanglement_entropy = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)


for i, t in enumerate(times):
    schedule[i] = t / tau
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

e0 = spectrum[:, 0]
e1 = spectrum[:, 1]
gap = spectrum[:, 1] - spectrum[:, 0]
p0 = probabilities[:, 0]
p1 = probabilities[:, 1]

magic = []
magic_gs_level = []
entanglement = []
entanglement_gs_level = []


# subsample if time_steps is large — SRE is O(4^N) per call
stride = max(1, 10)

for i in trange(0, time_steps, stride):
    state_full = sector.lift(psi_history_s[i])
    gs_full = sector.lift(eigenstates_history_s[i, :, 0])
    magic.append(sre(psi_history_s[i]))
    magic_gs_level.append(sre(eigenstates_history_s[i, :, 0]))
    entanglement.append(entanglement_entropy.von_neumann(state_full))
    entanglement_gs_level.append(entanglement_entropy.von_neumann(gs_full))

time_sub = times[::stride]


# formateo consistente de T para evitar problemas de precisión en el nombre
T_str = str(T)

nombre_archivo = (
    f"../../generated/FrustatedRing/QuantumResourcesvsT_T={T_str}_linear.npz"
)

np.savez(
    nombre_archivo,
    T=np.array([T]),  # guardamos T explícitamente también, por seguridad
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
    entanglement=entanglement,
    magic_gs_level=magic_gs_level,
    entanglement_gs_level=entanglement_gs_level,
)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {nombre_archivo}")
print(f"Elapsed time: {elapsed:.2f} seconds")
