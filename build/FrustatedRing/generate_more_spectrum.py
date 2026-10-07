import re
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.time_grid import midpoint_evolution, schedules_from_saved
from src.utils import Z2SymmetricSector

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
    # Schedule on both grids of the midpoint rule (src/time_grid.py). The
    # propagation grid is always rebuilt from T; for files written with the old
    # discretization s is re-evaluated from theta at the cell midpoints (the
    # saved `schedule` is never used as if it were the control schedule).
    grids = schedules_from_saved(data, prefix=f"T={Ti}_")
    times, delta_t = grids["times"], grids["dt"]
    schedule, schedule_ctrl = grids["schedule"], grids["schedule_ctrl"]
    print(f"T={Ti}: schedule from {grids['source']}, nsteps={grids['nsteps']}")

    n_times = len(times)
    probabilities = np.zeros((n_times, nlevels))
    spectrum = np.zeros((n_times, nlevels))
    energy = np.zeros(n_times)
    # psi is the state at times[i], i = 0..nsteps (i = 0: initial state)
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

        probabilities[i] = (
            np.einsum("i,ia->a", psi.conj(), eigenstates_raw)
            * np.einsum("i,ia->a", psi.conj(), eigenstates_raw).conj()
        ).real
        energy[i] = np.real(np.vdot(psi, hamiltonian_t @ psi))

    for i in range(nlevels):
        data[f"T={Ti}_p{i}"] = probabilities[:, i]
        data[f"T={Ti}_e{i}"] = spectrum[:, i]

    # Keep the file self-consistent: every full-resolution key of this T lives
    # on the state grid just used. For a file written with the old
    # discretization this REPLACES times/schedule/evo_energy/gap (old grid, one
    # point less) by the new-convention ones and drops levels beyond `nlevels`
    # that would be left on the old grid; the subsampled keys (magic,
    # entanglement, ...) are untouched and stay paired with their own time_sub.
    n_old = len(data[f"T={Ti}_times"])
    if n_old != n_times:
        stale = [
            k
            for k in data
            if re.fullmatch(rf"T={Ti}_[pe]\d+", k) and len(data[k]) == n_old
        ]
        for key in stale:
            del data[key]
        print(f"  old-grid file upgraded to the midpoint grid; dropped {stale}")
    data[f"T={Ti}_dt"] = np.array([delta_t])
    data[f"T={Ti}_nsteps"] = np.array([grids["nsteps"]])
    data[f"T={Ti}_times"] = times
    data[f"T={Ti}_times_ctrl"] = grids["times_ctrl"]
    data[f"T={Ti}_schedule"] = schedule
    data[f"T={Ti}_schedule_ctrl"] = schedule_ctrl
    data[f"T={Ti}_evo_energy"] = energy
    if nlevels >= 2:
        data[f"T={Ti}_gap"] = spectrum[:, 1] - spectrum[:, 0]
    elif n_old != n_times:
        data.pop(f"T={Ti}_gap", None)

filename_tmp = (
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR" + tag + ".npz"
)
np.savez(filename_tmp, **data)

end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Guardado: {filename}")
print(f"Elapsed time: {elapsed:.2f} seconds")
