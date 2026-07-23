import numpy as np
from scipy.sparse.linalg import expm_multiply
from src.annealing_utils import (
    get_longitudinal_hamiltonian,
    get_driver_hamiltonian,
)
import matplotlib.pyplot as plt
from src.hamiltonian_utils import frustrated_ring_jij_hz
from src.utils import Z2SymmetricSector


def final_energy_annealing(
    schedule, times, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
):
    """
    Runs the annealing process and returns only the final energy.
    schedule: array of the same length as `times`, giving s(t) at each step.
    """
    psi = psi0.copy()

    for i, t in enumerate(times):
        s = schedule[1][i]
        hamiltonian_t = (1 - s) * driver_hamiltonian_s + s * target_hamiltonian_s
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)

    # final energy: only need the Hamiltonian at the last time step
    final_energy = np.real(np.vdot(psi, hamiltonian_t @ psi))
    return final_energy


def energy_fn(
    theta,
    build_schedule,
    times,
    delta_t,
    psi0,
    driver_hamiltonian_s,
    target_hamiltonian_s,
):
    """
    Maps a parameter vector theta -> schedule -> final energy.
    build_schedule: function that, given theta and times, returns the schedule array.
    """
    schedule = build_schedule(theta, times)
    return final_energy_annealing(
        schedule, times, delta_t, psi0, driver_hamiltonian_s, target_hamiltonian_s
    )


def build_plane_basis(theta1, theta2, theta3):
    """
    Given 3 points in R^n, returns:
    - origin (theta1)
    - orthonormal basis (e1, e2) of the plane passing through the 3 points
    - coordinates (a, b) of theta1, theta2, theta3 in that basis
    """
    v1 = theta2 - theta1
    v2 = theta3 - theta1

    # Gram-Schmidt
    e1 = v1 / np.linalg.norm(v1)
    v2_proj = v2 - np.dot(v2, e1) * e1
    norm_v2_proj = np.linalg.norm(v2_proj)
    if norm_v2_proj < 1e-10:
        raise ValueError("The three points are collinear: they do not span a plane.")
    e2 = v2_proj / norm_v2_proj

    # coordinates of the 3 points in the (a, b) basis
    coords = {
        "theta1": (0.0, 0.0),
        "theta2": (np.dot(v1, e1), np.dot(v1, e2)),
        "theta3": (np.dot(v2, e1), np.dot(v2, e2)),
    }

    return theta1, e1, e2, coords


def energy_landscape(theta1, theta2, theta3, energy_fn, resolution=30, margin=0.3):
    """
    energy_fn: function that takes a theta vector (1D, same size as theta1/2/3)
               and returns a scalar (the final energy).
    resolution: number of points per axis in the grid.
    margin: extra fraction of space around the triangle formed by the 3 points.
    """
    origin, e1, e2, coords = build_plane_basis(theta1, theta2, theta3)

    # range of (a, b) to cover: bounding box of the triangle + margin
    as_ = [coords[k][0] for k in coords]
    bs_ = [coords[k][1] for k in coords]
    a_min, a_max = min(as_), max(as_)
    b_min, b_max = min(bs_), max(bs_)

    range_a = a_max - a_min
    range_b = b_max - b_min
    a_min -= margin * range_a
    a_max += margin * range_a
    b_min -= margin * range_b
    b_max += margin * range_b

    a_vals = np.linspace(a_min, a_max, resolution)
    b_vals = np.linspace(b_min, b_max, resolution)
    A, B = np.meshgrid(a_vals, b_vals)

    E = np.zeros_like(A)
    total = resolution * resolution
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            theta = origin + A[i, j] * e1 + B[i, j] * e2
            E[i, j] = energy_fn(theta)
            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{total}")

    return A, B, E, coords


def plot_landscape(A, B, E, coords, title="Energy landscape", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    cont = ax.contourf(A, B, E, levels=50, cmap="viridis")
    plt.colorbar(cont, ax=ax, label="Final energy")

    # mark the 3 original points
    for name, (a, b) in coords.items():
        ax.plot(a, b, "o", color="red", markersize=8)
        ax.annotate(
            name, (a, b), textcoords="offset points", xytext=(6, 6), color="white"
        )

    ax.set_xlabel("a (direction e1)")
    ax.set_ylabel("b (direction e2)")
    ax.set_title(title)
    ax.set_aspect(
        "equal"
    )  # important: since e1, e2 are orthonormal, this doesn't distort the plane
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


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

    # Step 1 — decode segment durations.
    # softplus(raw_durations) > 0 guarantees positive durations;
    # dividing by their sum and multiplying by tf renormalizes them
    # to add up to exactly the total annealing time.
    # ── softplus and its derivative ───────────────────────────────────────────────
    def _softplus(x: np.ndarray) -> np.ndarray:
        """
        log(1 + exp(x)), numerically stable.

        Used to map an unconstrained real parameter onto a strictly-positive
        number (e.g. a segment duration, which must be > 0). Computed as
        log1p(exp(-|x|)) + max(x, 0) instead of the naive log(1+exp(x)) to
        avoid overflow for large x.
        """
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Derivative of softplus = sigmoid(x) = 1 / (1 + exp(-x)).

        Two independent uses in this file:
        1. As d(softplus)/dx, needed by the chain rule wherever a
            softplus-mapped parameter (e.g. a raw duration) is differentiated.
        2. As a standalone squashing function 0->1, used to map the raw
            LZS plateau-height parameters into the physical range s in [0, 1].
        """
        return 1.0 / (1.0 + np.exp(-x))

    D = _softplus(raw_durations)
    Ssum = D.sum()
    scaled_durations = D / Ssum * tf
    t_bounds = np.concatenate(([0.0], np.cumsum(scaled_durations)))
    t_bounds[-1] = tf  # guard against fp drift

    # Step 3 — decode plateau heights via sigmoid into (0,1), and
    # assemble the full waypoint list s_way = [0, plateau_1, ...,
    # plateau_M, 1] (M+2 entries: the boundary values 0 and 1 are
    # NOT free parameters).
    sig_S = _sigmoid(raw_splateaus)
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


filename = f"../../generated/FrustatedRing/ParametersLZR_T={T}_N={N}.npz"
data = np.load(filename)
chosen_seeds = [2, 4, 6]

theta1 = data["theta_list"][chosen_seeds[0]]
theta2 = data["theta_list"][chosen_seeds[1]]
theta3 = data["theta_list"][chosen_seeds[2]]

A, B, E, coords = energy_landscape(
    theta1, theta2, theta3, energy_fn_wrapper, resolution=30
)

filename_img = f"../../images/FrustatedRing/LossLandscapeLZR_T={T}_N={N}_{chosen_seeds[0]}_{chosen_seeds[1]}_{chosen_seeds[2]}.png"

plot_landscape(
    A, B, E, coords, title=f"Energy landscape T={T} N={N}", save_path=filename_img
)
