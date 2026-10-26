# src/time_grid.py
"""
Single source of truth for the time discretization of every annealing
evolution in the project (midpoint rule, second order in dt).

Convention
----------
    nsteps      number of propagation steps
    dt          = tf / nsteps
    times       STATE grid,    t_k    = k dt,           k = 0..nsteps
                (nsteps + 1 points, times[0] = 0, times[-1] = tf)
    times_ctrl  CONTROL grid,  tbar_k = (k + 1/2) dt,   k = 0..nsteps-1
                (nsteps points, the midpoint of every cell)

    |psi_{k+1}> = exp(-i dt H(tbar_k)) |psi_k>,         k = 0..nsteps-1

so |psi_k> is the state at times[k]: |psi_0> is the initial state and
|psi_nsteps> = |psi(tf)>, after a total simulated time nsteps * dt = tf.

Rules that follow from it (see the callers for examples):
  - the schedule that DRIVES the evolution is the ansatz evaluated
    analytically on `times_ctrl` (never an average/interpolation of the
    state-grid values: the LZS ansatz has corners);
  - observables at times[k] use |psi_k> together with H(s(times[k])), i.e.
    the ansatz evaluated on `times`;
  - the initial state is always passed in explicitly (it must not be derived
    from the first control value, which is s(dt/2) != 0).
"""

import warnings

import numpy as np
from scipy.sparse.linalg import expm_multiply


def make_time_grids(tf: float, nsteps: int):
    """
    Returns (times, times_ctrl, dt) for `nsteps` propagation steps over [0, tf].

    times      : (nsteps + 1,) state grid, 0..tf inclusive
    times_ctrl : (nsteps,)     control grid, cell midpoints
    dt         : tf / nsteps
    """
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError(f"nsteps must be >= 1, got {nsteps}")
    dt = tf / nsteps
    times = np.linspace(0.0, tf, nsteps + 1)
    times_ctrl = (np.arange(nsteps) + 0.5) * dt
    return times, times_ctrl, dt


def subsample_indices(nsteps: int, stride: int) -> np.ndarray:
    """
    Indices into the state grid for observables that are too expensive to
    evaluate at every step: every `stride`-th point, always including the
    last one (t = tf).
    """
    idx = np.arange(0, nsteps + 1, max(int(stride), 1))
    if idx[-1] != nsteps:
        idx = np.append(idx, nsteps)
    return idx


def midpoint_evolution(
    psi0,
    h_driver_ctrl,
    h_target_ctrl,
    dt,
    driver_hamiltonian,
    target_hamiltonian,
):
    """
    Generator over the states of a midpoint-rule evolution in the spin basis.

    Yields (k, psi_k) for k = 0..nsteps, where psi_k is the state at
    times[k] (k = 0 is a copy of psi0, k = nsteps is psi(tf)). Step k applies
    exp(-i dt (h_driver_ctrl[k] H_driver + h_target_ctrl[k] H_target)) through
    scipy's sparse expm_multiply.

    h_driver_ctrl, h_target_ctrl : (nsteps,) schedules on the CONTROL grid.
    """
    nsteps = len(h_driver_ctrl)
    if len(h_target_ctrl) != nsteps:
        raise ValueError("h_driver_ctrl and h_target_ctrl must have the same length")
    psi = np.array(psi0, dtype=complex)
    yield 0, psi
    for k in range(nsteps):
        hamiltonian_k = (
            h_driver_ctrl[k] * driver_hamiltonian
            + h_target_ctrl[k] * target_hamiltonian
        )
        psi = expm_multiply(-1j * dt * hamiltonian_k, psi)
        yield k + 1, psi


def nambu_midpoint_evolution(nambu, w0, h_driver_ctrl, h_target_ctrl, dt):
    """
    Free-fermion counterpart of midpoint_evolution.

    Yields (k, w_k) for k = 0..nsteps, w_k [2l, 2l] being the Bogoliubov
    matrix at times[k]. `w0` is mandatory: NambuIsing1D.evolve would otherwise
    default to the ground state of H at the first control point, s(dt/2) != 0.
    """
    nsteps = len(h_driver_ctrl)
    if len(h_target_ctrl) != nsteps:
        raise ValueError("h_driver_ctrl and h_target_ctrl must have the same length")
    w = np.asarray(w0, dtype=np.complex128)
    yield 0, w
    for k in range(nsteps):
        w, _ = nambu.evolve([h_driver_ctrl[k]], [h_target_ctrl[k]], dt, w0=w)
        yield k + 1, w


def schedules_from_saved(data, prefix: str = "", verbose: bool = True) -> dict:
    """
    Schedule of a saved run on BOTH grids of the current convention, for code
    that re-evolves it (works with files written before and after the
    midpoint-rule change).

    data   : mapping with the keys of one run (an open .npz or a dict)
    prefix : key prefix inside a merged file, e.g. "T=10_"

    The propagation grid is always rebuilt from T with make_time_grids; `dt`
    is never taken from the saved `times`. Order of preference for s(t) at
    the cell midpoints:
      1. `schedule_ctrl` stored in the file (new files);
      2. old file with `theta`: the LZS ansatz re-evaluated analytically on
         the new grids. The parametrization (bounds_opt or softplus/sigmoid)
         is not stored, so it is identified as the one that reproduces the
         saved `schedule` on the saved `times`;
      3. old file with the linear schedule: s = t / T;
      4. last resort: np.interp of the saved schedule (small error only in
         the cells that contain a corner of the schedule).
    For old files nsteps = len(saved times), which keeps the steps per unit
    time of the script that generated them.

    Returns dict(times, times_ctrl, dt, nsteps, schedule, schedule_ctrl, source).
    """

    def has(key):
        return (prefix + key) in data

    def get(key):
        return np.asarray(data[prefix + key])

    old_times = get("times")
    tf = float(get("T").ravel()[0]) if has("T") else float(old_times[-1])

    # 1. new file: both grids were stored
    if has("schedule_ctrl"):
        schedule_ctrl = get("schedule_ctrl")
        nsteps = len(schedule_ctrl)
        times, times_ctrl, dt = make_time_grids(tf, nsteps)
        return dict(
            times=times,
            times_ctrl=times_ctrl,
            dt=dt,
            nsteps=nsteps,
            schedule=get("schedule"),
            schedule_ctrl=schedule_ctrl,
            source="schedule_ctrl",
        )

    # old file: `times` is linspace(0, T, nsteps) and `schedule` is s on it
    old_schedule = get("schedule")
    nsteps = len(old_times)
    times, times_ctrl, dt = make_time_grids(tf, nsteps)
    out = dict(times=times, times_ctrl=times_ctrl, dt=dt, nsteps=nsteps)

    # 2. re-evaluate the LZS ansatz from theta
    if has("theta"):
        import scipy.sparse as sp

        from src.sparse_grape_method import SparseGRAPEModel

        theta = get("theta").ravel()
        if (len(theta) - 1) % 3 == 0:
            one = sp.identity(1, format="csr")
            for bounds_opt in (True, False):
                # schedule machinery only: the Hamiltonians are placeholders
                ansatz = SparseGRAPEModel(
                    initial_state=np.ones(1),
                    target_hamiltonian=one,
                    initial_hamiltonian=one,
                    reference_hamiltonian=one,
                    tf=tf,
                    number_of_parameters=(len(theta) - 1) // 3,
                    nsteps=nsteps,
                    type="LZS",
                    bounds_opt=bounds_opt,
                )
                with np.errstate(all="ignore"):
                    s_old = ansatz.get_driving(theta, grid=old_times)[1]
                if s_old.shape == old_schedule.shape and np.allclose(
                    s_old, old_schedule, rtol=0.0, atol=1e-9
                ):
                    out.update(
                        schedule=ansatz.get_driving(theta, grid="state")[1],
                        schedule_ctrl=ansatz.get_driving(theta, grid="control")[1],
                        source=f"theta (LZS, bounds_opt={bounds_opt})",
                    )
                    return out

    # 3. linear schedule
    if np.allclose(old_schedule, old_times / tf, rtol=0.0, atol=1e-12):
        out.update(schedule=times / tf, schedule_ctrl=times_ctrl / tf, source="linear")
        return out

    # 4. interpolation of the saved samples
    if verbose:
        warnings.warn(
            f"{prefix}: schedule could not be rebuilt analytically; "
            "interpolating the saved samples onto the midpoint grid."
        )
    out.update(
        schedule=np.interp(times, old_times, old_schedule),
        schedule_ctrl=np.interp(times_ctrl, old_times, old_schedule),
        source="interp",
    )
    return out
