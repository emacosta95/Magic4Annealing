import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse.linalg import expm_multiply

from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
)
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.jax_utils import SREJax
from src.utils import EntanglementEntropy, Z2SymmetricSector
from src.landscape_utils import (
    energy_fn,
    max_magic_fn,
    max_entanglement_fn,
    energy_landscape,
    max_magic_landscape,
    max_entanglement_landscape,
    plot_energy_landscape,
    plot_max_magic_landscape,
    plot_max_entanglement_landscape,
)


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

    # # Step 1 — decode segment durations.
    # # softplus(raw_durations) > 0 guarantees positive durations;
    # # dividing by their sum and multiplying by tf renormalizes them
    # # to add up to exactly the total annealing time.
    # # ── softplus and its derivative ───────────────────────────────────────────────
    # def _softplus(x: np.ndarray) -> np.ndarray:
    #     """
    #     log(1 + exp(x)), numerically stable.

    #     Used to map an unconstrained real parameter onto a strictly-positive
    #     number (e.g. a segment duration, which must be > 0). Computed as
    #     log1p(exp(-|x|)) + max(x, 0) instead of the naive log(1+exp(x)) to
    #     avoid overflow for large x.
    #     """
    #     return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    # def _sigmoid(x: np.ndarray) -> np.ndarray:
    #     """
    #     Derivative of softplus = sigmoid(x) = 1 / (1 + exp(-x)).

    #     Two independent uses in this file:
    #     1. As d(softplus)/dx, needed by the chain rule wherever a
    #         softplus-mapped parameter (e.g. a raw duration) is differentiated.
    #     2. As a standalone squashing function 0->1, used to map the raw
    #         LZS plateau-height parameters into the physical range s in [0, 1].
    #     """
    #     return 1.0 / (1.0 + np.exp(-x))

    D = raw_durations
    Ssum = D.sum()
    scaled_durations = D / Ssum * tf
    t_bounds = np.concatenate(([0.0], np.cumsum(scaled_durations)))
    t_bounds[-1] = tf  # guard against fp drift

    # Step 3 — decode plateau heights via sigmoid into (0,1), and
    # assemble the full waypoint list s_way = [0, plateau_1, ...,
    # plateau_M, 1] (M+2 entries: the boundary values 0 and 1 are
    # NOT free parameters).
    sig_S = raw_splateaus
    s_way = np.concatenate(([0.0], sig_S, [1.0]))  # (M+2,)

    # Step 4 — walk through the 2M+1 alternating ramp/plateau
    # segments, filling in s(t) and its Jacobian ds_dtheta segment
    # by segment. ds_dtheta packs BOTH parameter blocks into one
    # (n_params, nsteps) array: rows [0:n_seg] are duration
    # sensitivities, rows [n_seg:n_seg+M] are plateau-height
    # sensitivities.
    s = np.zeros_like(t)

    for seg in range(n_seg):
        t0, t1 = t_bounds[seg], t_bounds[seg + 1]
        mask = (t >= t0) & (t <= t1)  # heaviside condition
        tm = t[mask]
        denom = (t1 - t0) if t1 > t0 else 1.0

        if seg % 2 == 0:
            # Ramp segment (even index): linear interpolation
            # between waypoint k and k+1, k = seg // 2.
            k = seg // 2
            s0, s1_ = s_way[k], s_way[k + 1]
            frac = (tm - t0) / denom
            s[mask] = s0 + (s1_ - s0) * frac

        else:
            # Plateau segment (odd index): s is held constant at
            # s_way[k], k = (seg+1)//2, for the whole segment — so
            # there is no time-dependence and hence NO duration
            # sensitivity (a plateau's height doesn't change if you
            # stretch or shrink how long it lasts).
            k = (seg + 1) // 2
            s[mask] = s_way[k]

    # h_driver = 1 - s, h_target = s (no ramp envelope for LZS), so
    # their theta-Jacobians are just -ds_dtheta and +ds_dtheta.
    h_driver = 1.0 - s
    h_target = s

    return h_driver, h_target


# theta1, theta2, theta3 = your three parameter vectors (1D arrays of the same size)
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

# ── optimization parameters ───────────────────────────────────────────────────
number_parameters = 2  # M=2 plateaus/arms -> n_params = 3*M+1 = 7, matching
# Werner et al.'s reduction from Cote et al.'s ~100-parameter
# variational schedule down to 7 parameters
type = "LZS"
resolution = 25


filename = f"../../generated/FrustatedRing/ParametersLZR_T={T}_N={N}.npz"
data = np.load(filename)
chosen_seeds = [7, 8, 9]

theta1 = data["theta_list"][chosen_seeds[0]]
theta2 = data["theta_list"][chosen_seeds[1]]
theta3 = data["theta_list"][chosen_seeds[2]]

sre = SREJax(n_qubits=nqubits - 1, batch_size=1000)
entanglement_entropy = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)

A, B, E, coords = energy_landscape(
    theta1, theta2, theta3, energy_fn_wrapper, resolution=resolution
)

A, B, max_magic, coords = max_magic_landscape(
    theta1, theta2, theta3, max_magic_fn_wrapper, resolution=resolution
)

A, B, max_entanglement, coords = max_entanglement_landscape(
    theta1, theta2, theta3, max_entanglement_fn_wrapper, resolution=resolution
)

filename_img_energy = f"../../images/FrustatedRing/FinalEnergyLandscapeLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"
filename_img_max_entanglement = f"../../images/FrustatedRing/MaxEntanglementLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"
filename_img_max_magic = f"../../images/FrustatedRing/MaxMagicLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"

energies = {
    "theta1": energy_fn_wrapper(theta1),
    "theta2": energy_fn_wrapper(theta2),
    "theta3": energy_fn_wrapper(theta3),
}

plot_energy_landscape(
    A,
    B,
    E,
    coords,
    energies,
    title=f"Energy landscape T={T} N={N}",
    save_path=filename_img_energy,
)

plot_max_magic_landscape(
    A,
    B,
    max_magic,
    coords,
    energies,
    title=f"Max magic landscape T={T} N={N}",
    save_path=filename_img_max_magic,
)

plot_max_entanglement_landscape(
    A,
    B,
    max_entanglement,
    coords,
    energies,
    title=f"Max entanglement landscape T={T} N={N}",
    save_path=filename_img_max_entanglement,
)
