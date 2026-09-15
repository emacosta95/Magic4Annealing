import os
import re

import numpy as np

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.jax_utils import SREJax
from src.landscape_utils import (
    energy_fn,
    energy_landscape_1d,
    max_entanglement_fn,
    max_entanglement_landscape_1d,
    max_magic_fn,
    max_magic_landscape_1d,
    plot_energy_landscape_1d,
    plot_max_entanglement_landscape_1d,
    plot_max_magic_landscape_1d,
)
from src.utils import EntanglementEntropy, Z2SymmetricSector


def load_data(archivo_salida):
    """
    Lee el .npz combinado y devuelve un diccionario:
    { T (int): {nombre_variable: array, ...}, ... }
    """
    data = np.load(archivo_salida)

    patron = re.compile(r"^T=(\d+)_(.+)$")
    resultado = {}

    for clave in data.files:
        match = patron.match(clave)
        if not match:
            print(
                f"Aviso: clave '{clave}' no coincide con el patrón esperado, se omite."
            )
            continue

        Ti = int(match.group(1))
        nombre_variable = match.group(2)

        if Ti not in resultado:
            resultado[Ti] = {}

        resultado[Ti][nombre_variable] = data[clave]

    return resultado


def build_schedule(theta, t):
    # Direct s(t) parametrization: h_driver=1-s, h_target=s, so BOTH
    # depend on the FULL parameter vector — unlike the branches
    # above, where driver/target params are disjoint. Durations are
    # jointly softplus-normalized to sum to tf, so a change in any
    # single raw_duration_m shifts EVERY segment boundary, not just
    # its own segment — this couples all n_seg duration params
    # together in the Jacobian (see dTb below).
    tf = times[-1]
    parameters = theta
    M = 2  # number of plateaus/arms
    n_seg = 5
    raw_durations = parameters[:n_seg]
    raw_splateaus = parameters[n_seg : n_seg + M]

    D = raw_durations
    Ssum = D.sum()
    scaled_durations = D / Ssum * tf
    t_bounds = np.concatenate(([0.0], np.cumsum(scaled_durations)))
    t_bounds[-1] = tf  # guard against fp drift

    sig_S = raw_splateaus
    s_way = np.concatenate(([0.0], sig_S, [1.0]))  # (M+2,)

    s = np.zeros_like(t)

    for seg in range(n_seg):
        t0, t1 = t_bounds[seg], t_bounds[seg + 1]
        mask = (t >= t0) & (t <= t1)  # heaviside condition
        tm = t[mask]
        denom = (t1 - t0) if t1 > t0 else 1.0

        if seg % 2 == 0:
            k = seg // 2
            s0, s1_ = s_way[k], s_way[k + 1]
            frac = (tm - t0) / denom
            s[mask] = s0 + (s1_ - s0) * frac
        else:
            k = (seg + 1) // 2
            s[mask] = s_way[k]

    h_driver = 1.0 - s
    h_target = s

    return h_driver, h_target


# theta1, theta2 = your two parameter vectors (1D arrays of the same size),
# assumed already defined elsewhere.
# build_schedule = your function mapping theta -> schedule
# times, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s = your simulation setup


def energy_fn_wrapper(theta):
    return energy_fn(
        theta,
        build_schedule,
        times,
        delta_t,
        psi_init_s,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    )


def max_magic_fn_wrapper(theta):
    return max_magic_fn(
        theta,
        build_schedule,
        times,
        delta_t,
        psi_init_s,
        sre,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    )


def max_entanglement_fn_wrapper(theta):
    return max_entanglement_fn(
        theta,
        build_schedule,
        times,
        delta_t,
        psi_init_s,
        sector,
        entanglement_entropy,
        driver_hamiltonian_s,
        target_hamiltonian_s,
    )


T = 120
T2 = T

tag = "_bounded_SA"  # "_NoGrad", "_step_test", "_bounded", "_no_random" or ""
tag_T = ""  # "_more_T" or ""
N = 7

tag2 = "_bounded"  # "_NoGrad", "_step_test", "_bounded", "_no_random" or ""
tag_T2 = ""
N2 = N

data_LZR = load_data(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N}_LZR"
    + tag
    + tag_T
    + ".npz"
)

data_LZR2 = load_data(
    f"../../generated/FrustatedRing/QuantumResourcesvsT_N={N2}_LZR"
    + tag2
    + tag_T2
    + ".npz"
)


theta1 = data_LZR[T]["theta"]
theta2 = data_LZR2[T2]["theta"]

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

# ── optimization parameters ───────────────────────────────────────────────────
number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"
resolution = 100

sre = SREJax(n_qubits=nqubits - 1, batch_size=1000)
entanglement_entropy = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)

a_vals, E, coords = energy_landscape_1d(
    theta1, theta2, energy_fn_wrapper, resolution=resolution
)

_, max_magic, coords = max_magic_landscape_1d(
    theta1, theta2, max_magic_fn_wrapper, resolution=resolution
)

_, max_entanglement, coords = max_entanglement_landscape_1d(
    theta1, theta2, max_entanglement_fn_wrapper, resolution=resolution
)

path = f"../../images/FrustatedRing/InterpolationGRAPEvsSA_T={T}_N={N}/"
if not os.path.exists(path):
    os.makedirs(path)

filename_img_energy = f"{path}FinalEnergy.png"
filename_img_max_entanglement = f"{path}MaxEntanglement.png"
filename_img_max_magic = f"{path}MaxMagic.png"

energies = {
    "theta1": energy_fn_wrapper(theta1),
    "theta2": energy_fn_wrapper(theta2),
}

plot_energy_landscape_1d(
    a_vals,
    E,
    coords,
    energies,
    title=f"Energy landscape T={T} N={N}",
    save_path=filename_img_energy,
)

plot_max_magic_landscape_1d(
    a_vals,
    max_magic,
    coords,
    energies,
    title=f"Max magic landscape T={T} N={N}",
    save_path=filename_img_max_magic,
)

plot_max_entanglement_landscape_1d(
    a_vals,
    max_entanglement,
    coords,
    energies,
    title=f"Max entanglement landscape T={T} N={N}",
    save_path=filename_img_max_entanglement,
)
