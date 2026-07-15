# src/sparse_grape_method.py
"""
Sparse GRAPE implementation — drop-in replacement for JaxSchedulerModel/JaxTrainer.

Forward pass  : scipy sparse expm_multiply  — O(nsteps × dim × nnz)
Gradient      : GRAPE backpropagation        — exact, same cost as 2 forward passes
                No autodiff needed — gradient is derived analytically.

Supports all schedule types from JaxSchedule:
    'fourier'           : sin-only Fourier correction          — 2*dim params
    'F-CRAB'            : same with randomised frequencies     — 2*dim params
    'positive fourier'  : softplus-wrapped sin+cos             — 4*dim params
    'squared fourier'   : squared sin+cos correction           — 4*dim params
    'power law'         : polynomial correction                — 2*dim params
    'LZS'               : M-plateau interferometer ansatz      — 3*dim+1 params

Schedule structure (same as JaxSchedule):
    h_driver(t_i) = ramp_drv(t_i) * (1 + correction_drv(t_i))
    h_target(t_i) = ramp_tgt(t_i) * (1 + correction_tgt(t_i))

    where ramp_drv = (1 - t/tf),  ramp_tgt = t/tf

    Exception: 'LZS' parameterizes s(t) directly (piecewise linear ramps +
    plateaus), with h_driver = 1 - s(t), h_target = s(t) — no ramp envelope,
    mirrors schedule_utils.Schedule and JaxSchedule exactly.

GRAPE formula:
    dE/dh_drv_i = -2 dt Im[ <χ_i | H_driver | ψ_i> ]
    dE/dh_tgt_i = -2 dt Im[ <χ_i | H_target | ψ_i> ]

    grad[k] = Σ_i dE/dh_drv_i * dh_drv_i/dθ_k
            + Σ_i dE/dh_tgt_i * dh_tgt_i/dθ_k

    where |χ_i> is the co-state propagated backward from |χ_T> = H_ref |ψ_T>.

Usage in study_1d_ising.py:
    from src.sparse_grape_method import SparseGRAPEModel, SparseGRAPETrainer

    model = SparseGRAPEModel(
        initial_state=psi_init_s,
        target_hamiltonian=target_hamiltonian_s,
        initial_hamiltonian=driver_hamiltonian_s,
        reference_hamiltonian=target_hamiltonian_s,
        tf=tau,
        number_of_parameters=n_params,
        nsteps=time_steps,
        type=schedule_type,
        seed=42,
    )
    trainer = SparseGRAPETrainer(model, maxiter=maxiter, verbose=verbose)
    opt_results = trainer.run()

    # opt_results keys are identical to JaxTrainer output:
    # success, message, n_iterations, n_evals, energy,
    # parameters, psi, h_driver, h_target, time,
    # history_energy, history_parameters, history_drivings, history_psi

-------------------------------------------------------------------------------
COLLABORATOR QUICK-START — reading order for this file
-------------------------------------------------------------------------------
1. The physics setup: at every timestep t_i we build a time-dependent
   Hamiltonian H(t_i) = h_driver(t_i) * H_driver + h_target(t_i) * H_target
   and propagate the wavefunction psi one small step forward with a matrix
   exponential. h_driver/h_target are two curves ("schedules") between 0
   and 1 that control the annealing protocol; everything in this file is
   about (a) how those curves are generated from a small set of free
   parameters theta ("schedule ansatz"), and (b) how to differentiate the
   final energy w.r.t. theta so an optimizer (L-BFGS-B) can improve it.
2. SparseGRAPEModel.__init__            — allocates parameters per ansatz type.
3. SparseGRAPEModel._compute_driving_and_jacobian
                                         — theta -> (h_driver, h_target) and
                                           their exact derivatives w.r.t. theta.
                                           This is the only place that needs
                                           touching if you add a new ansatz.
4. SparseGRAPEModel._forward_and_grad   — propagates psi forward, computes
                                           energy, propagates a "co-state"
                                           backward, and combines everything
                                           into the GRAPE gradient (the
                                           analytic quantum-control analogue
                                           of backprop).
5. SparseGRAPETrainer.run               — wraps the above in scipy's
                                           L-BFGS-B, with a finite-difference
                                           sanity check on the gradient at
                                           the starting point.

If you only need to *use* this module (not modify the ansatz), you can
mostly ignore _compute_driving_and_jacobian's internals and just treat
SparseGRAPEModel/SparseGRAPETrainer like a black box with the same
interface as JaxSchedulerModel/JaxTrainer (see "Usage" above).
-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
from typing import Optional


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


_SOFTPLUS_1 = _softplus(np.ones(1))[0]  # softplus(1) ≈ 1.3133


# ─────────────────────────────────────────────────────────────────────────────
class SparseGRAPEModel:
    """
    Sparse GRAPE optimal control model.
    Drop-in replacement for JaxSchedulerModel — identical public interface.

    Holds the (sparse) Hamiltonians, the current parameter vector theta,
    and the optimization-history bookkeeping. Does not itself run an
    optimizer — see SparseGRAPETrainer for that.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        target_hamiltonian: sp.spmatrix,
        initial_hamiltonian: sp.spmatrix,
        reference_hamiltonian: sp.spmatrix,
        tf: float,
        number_of_parameters: int,
        nsteps: int,
        type: str = "fourier",
        seed: int = 42,
        mode: Optional[str] = "annealing ansatz",
        random: bool = False,
    ):
        """
        Parameters
        ----------
        initial_state : (dim,) complex ndarray
            psi(t=0), typically the driver Hamiltonian's ground state.
        target_hamiltonian, initial_hamiltonian : sparse (dim, dim)
            H_target (problem Hamiltonian, coefficient h_target(t)) and
            H_driver (mixer/driver Hamiltonian, coefficient h_driver(t)).
            H(t) = h_driver(t) * initial_hamiltonian + h_target(t) * target_hamiltonian.
        reference_hamiltonian : sparse (dim, dim)
            Hamiltonian used to evaluate the cost/energy at t=tf, i.e.
            E = <psi(tf)| reference_hamiltonian |psi(tf)>. Usually equal to
            target_hamiltonian (minimize the problem energy), but kept
            separate in case a different observable is optimized against.
        tf : float
            Total annealing time.
        number_of_parameters : int
            The "dim" knob for the chosen ansatz — its meaning depends on
            `type` (see the per-type sizing block below); for 'LZS' this is
            M, the number of plateaus, NOT the total parameter count.
        nsteps : int
            Number of points in the time discretization (uniform grid).
        type : str
            Which schedule ansatz to use — see module docstring for the
            full list and each one's parameter count.
        seed : int
            RNG seed, used for F-CRAB's randomised frequencies and/or
            random parameter initialization.
        mode : str, optional
            Present for interface parity with JaxSchedulerModel; has no
            effect for 'LZS' (boundary conditions s(0)=0, s(tf)=1 are
            hard-coded there regardless).
        random : bool
            If True, initialize parameters ~ Uniform(-0.5, 0.5) instead of
            all-zeros.
        """
        self.tf = tf
        self.nsteps = nsteps
        self.type = type
        self.number_parameters = number_of_parameters
        self.time = np.linspace(0, tf, nsteps)
        self.dt = self.time[1] - self.time[0]

        dim = number_of_parameters
        t = self.time

        # ── parameter sizing ──────────────────────────────────────────────────
        # Each ansatz type packs its free parameters into a single flat
        # vector `self.parameters`. The slicing convention used here (which
        # chunk of the vector belongs to which physical quantity) is repeated
        # in _compute_driving_and_jacobian and must stay in sync with it.
        if type in ("fourier", "F-CRAB"):
            # sin only: [a_drv(dim), a_tgt(dim)]
            n_params = 2 * dim
        elif type in ("positive fourier", "squared fourier"):
            # sin + cos: [a_drv(dim), b_drv(dim), a_tgt(dim), b_tgt(dim)]
            n_params = 4 * dim
        elif type == "power law":
            # [c_drv(dim), c_tgt(dim)]
            n_params = 2 * dim
        elif type == "LZS":
            # dim = M plateaus/interferometer arms.
            # (2M+1) segment durations + M plateau heights — mirrors
            # schedule_utils.Schedule / JaxSchedule exactly. Boundary
            # conditions s(0)=0, s(tf)=1 are built in, so 'mode' has no
            # effect for this type.
            n_params = 3 * dim + 1
        else:
            raise ValueError(
                f"Unknown schedule type '{type}'. "
                f"Choose from: fourier, F-CRAB, positive fourier, "
                f"squared fourier, power law, LZS."
            )

        self.parameters = np.zeros(n_params)
        if random:
            rng = np.random.default_rng(seed)
            self.parameters = rng.uniform(-0.5, 0.5, size=n_params)

        # ── basis functions ───────────────────────────────────────────────────
        # Precompute the (fixed, parameter-independent) time-series basis
        # functions once here so _compute_driving_and_jacobian only needs to
        # do a cheap matrix-vector product per optimizer call.
        if type in ("fourier", "F-CRAB", "positive fourier", "squared fourier"):
            if type == "F-CRAB":
                # F-CRAB ("Free" CRAB): frequencies are randomised around
                # the harmonic grid pi*k/tf to break the periodicity that
                # would otherwise let the optimizer get trapped by the
                # basis's built-in symmetries.
                rng = np.random.default_rng(seed)
                self.omegas = (
                    np.pi
                    * np.arange(1, dim + 1)
                    * (1 + rng.uniform(-0.5, 0.5, dim))
                    / tf
                )
            else:
                self.omegas = np.pi * np.arange(1, dim + 1) / tf

            # sin_basis[k, i] = sin(ω_k * t_i)   shape: (dim, nsteps)
            self._sin_basis = np.sin(np.outer(self.omegas, t))

            if type in ("positive fourier", "squared fourier"):
                # cos only needed for these two types
                self._cos_basis = np.cos(np.outer(self.omegas, t))

        elif type == "power law":
            exponents = np.arange(1, dim + 1)[:, None]
            # pw_basis[k, i] = (t_i / tf)^(k+1)   shape: (dim, nsteps)
            self._pw_basis = (t[None, :] / tf) ** exponents

        elif type == "LZS":
            # Direct s(t) parametrization — nothing to precompute besides
            # the segment count (kept for clarity/symmetry with JaxSchedule).
            # M plateaus split the schedule into 2M+1 alternating segments:
            # ramp, plateau, ramp, plateau, ..., ramp (M+1 ramps, M plateaus).
            self._lzs_M = dim
            self._lzs_n_seg = 2 * dim + 1

        # ── sparse Hamiltonians (kept sparse throughout) ───────────────────────
        # Cast to complex up front so every downstream matrix-vector product
        # (expm_multiply, inner products) is complex without repeated casts.
        self._H_driver = initial_hamiltonian.astype(complex)
        self._H_target = target_hamiltonian.astype(complex)
        self._H_ref = reference_hamiltonian.astype(complex)
        self._psi_init = initial_state.astype(complex)

        # ── state tracking (mirrors JaxSchedulerModel) ────────────────────────
        # `energy`/`psi` reflect the most recent forward() or
        # forward_and_gradient() call; the history_* lists are appended to
        # only inside callback() (i.e. once per accepted optimizer iteration,
        # not per internal line-search evaluation).
        self.energy = 1000.0
        self.psi = None
        self.history = []
        self.history_parameters = []
        self.history_drivings = []
        self.history_psi = []
        self.history_run = []
        self.run_number = 0

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_driving_and_jacobian(self, parameters: np.ndarray):
        """
        Compute schedules and their Jacobians w.r.t. parameters.

        This is the core "ansatz" function: it maps the flat parameter
        vector theta onto the two schedule curves h_driver(t), h_target(t)
        actually fed into the Hamiltonian, AND the analytic derivative of
        every timestep's h value w.r.t. every parameter. That Jacobian is
        what lets _forward_and_grad turn a per-timestep gradient (from
        GRAPE) into a per-parameter gradient via the chain rule, without
        any numerical differentiation or autodiff.

        Parameters
        ----------
        parameters : (n_params,) ndarray
            theta, the current parameter vector (one branch is taken below
            depending on self.type; each branch knows its own slicing of
            this vector — see the sizing comment in __init__).

        Returns
        -------
        h_driver      : (nsteps,)        — driver schedule
        h_target      : (nsteps,)        — target schedule
        dh_drv_dtheta : (n_params, nsteps) — ∂h_driver_i / ∂θ_k
        dh_tgt_dtheta : (n_params, nsteps) — ∂h_target_i / ∂θ_k
        """
        dim = self.number_parameters
        t = self.time
        tf = self.tf
        n_params = len(parameters)

        ramp_drv = 1.0 - t / tf  # (nsteps,)
        ramp_tgt = t / tf  # (nsteps,)

        dh_drv = np.zeros((n_params, self.nsteps))
        dh_tgt = np.zeros((n_params, self.nsteps))

        # ── fourier / F-CRAB: sin only ────────────────────────────────────────
        # h(t) = ramp(t) * (1 + sum_k a_k sin(ω_k t)) — a linear ramp from the
        # pure-driver to the pure-target Hamiltonian, modulated by a sine
        # series correction. Because the correction enters linearly, the
        # Jacobian is just the (ramp-weighted) basis functions themselves.
        if self.type in ("fourier", "F-CRAB"):
            a_drv = parameters[:dim]
            a_tgt = parameters[dim : 2 * dim]

            corr_drv = self._sin_basis.T @ a_drv  # (nsteps,)
            corr_tgt = self._sin_basis.T @ a_tgt

            h_driver = ramp_drv * (1.0 + corr_drv)
            h_target = ramp_tgt * (1.0 + corr_tgt)

            # ∂h_driver_i / ∂a_drv_k = ramp_drv_i * sin(ω_k * t_i)
            dh_drv[:dim, :] = self._sin_basis * ramp_drv[None, :]
            # ∂h_target_i / ∂a_tgt_k = ramp_tgt_i * sin(ω_k * t_i)
            dh_tgt[dim : 2 * dim, :] = self._sin_basis * ramp_tgt[None, :]

        # ── positive fourier: softplus(1 + sin+cos correction) ───────────────
        # Same idea as above but the correction is passed through softplus
        # (divided by softplus(1) so the ramp envelope is preserved at
        # theta=0) to GUARANTEE h_driver/h_target stay >= 0 — useful when the
        # optimizer must not be allowed to flip the sign of a coupling.
        elif self.type == "positive fourier":
            a_drv = parameters[:dim]
            b_drv = parameters[dim : 2 * dim]
            a_tgt = parameters[2 * dim : 3 * dim]
            b_tgt = parameters[3 * dim : 4 * dim]

            raw_drv = self._sin_basis.T @ a_drv + self._cos_basis.T @ b_drv  # (nsteps,)
            raw_tgt = self._sin_basis.T @ a_tgt + self._cos_basis.T @ b_tgt

            h_driver = ramp_drv * _softplus(1.0 + raw_drv) / _SOFTPLUS_1
            h_target = ramp_tgt * _softplus(1.0 + raw_tgt) / _SOFTPLUS_1

            # chain rule: d/da_k softplus(1 + raw) = sigmoid(1 + raw) * sin_k
            sig_drv = _sigmoid(1.0 + raw_drv) / _SOFTPLUS_1  # (nsteps,)
            sig_tgt = _sigmoid(1.0 + raw_tgt) / _SOFTPLUS_1

            dh_drv[:dim, :] = (ramp_drv * sig_drv)[None, :] * self._sin_basis
            dh_drv[dim : 2 * dim, :] = (ramp_drv * sig_drv)[None, :] * self._cos_basis
            dh_tgt[2 * dim : 3 * dim, :] = (ramp_tgt * sig_tgt)[
                None, :
            ] * self._sin_basis
            dh_tgt[3 * dim : 4 * dim, :] = (ramp_tgt * sig_tgt)[
                None, :
            ] * self._cos_basis

        # ── squared fourier: (1 + sin+cos correction)^2 ──────────────────────
        # Alternative positivity-enforcing nonlinearity: squaring instead of
        # softplus. Cheaper to differentiate (polynomial chain rule) but,
        # unlike softplus, allows h to touch exactly 0 (at raw = -1).
        elif self.type == "squared fourier":
            a_drv = parameters[:dim]
            b_drv = parameters[dim : 2 * dim]
            a_tgt = parameters[2 * dim : 3 * dim]
            b_tgt = parameters[3 * dim : 4 * dim]

            raw_drv = self._sin_basis.T @ a_drv + self._cos_basis.T @ b_drv  # (nsteps,)
            raw_tgt = self._sin_basis.T @ a_tgt + self._cos_basis.T @ b_tgt

            h_driver = ramp_drv * (1.0 + raw_drv) ** 2
            h_target = ramp_tgt * (1.0 + raw_tgt) ** 2

            # d/da_k (1+raw)^2 = 2*(1+raw) * sin_k
            factor_drv = 2.0 * ramp_drv * (1.0 + raw_drv)  # (nsteps,)
            factor_tgt = 2.0 * ramp_tgt * (1.0 + raw_tgt)

            dh_drv[:dim, :] = factor_drv[None, :] * self._sin_basis
            dh_drv[dim : 2 * dim, :] = factor_drv[None, :] * self._cos_basis
            dh_tgt[2 * dim : 3 * dim, :] = factor_tgt[None, :] * self._sin_basis
            dh_tgt[3 * dim : 4 * dim, :] = factor_tgt[None, :] * self._cos_basis

        # ── power law ─────────────────────────────────────────────────────────
        # Correction built from a polynomial basis (t/tf)^(k+1), k=1..dim,
        # rather than a Fourier series — a smoother, low-frequency-biased
        # alternative ansatz.
        elif self.type == "power law":
            c_drv = parameters[:dim]
            c_tgt = parameters[dim : 2 * dim]

            corr_drv = self._pw_basis.T @ c_drv  # (nsteps,)
            corr_tgt = self._pw_basis.T @ c_tgt

            h_driver = ramp_drv * (1.0 + corr_drv)
            h_target = ramp_tgt * (1.0 + corr_tgt)

            # ∂h_driver_i / ∂c_drv_k = ramp_drv_i * (t_i/tf)^(k+1)
            dh_drv[:dim, :] = self._pw_basis * ramp_drv[None, :]
            dh_tgt[dim : 2 * dim, :] = self._pw_basis * ramp_tgt[None, :]

        # ── LZS: M-plateau Landau-Zener-Stückelberg interference ansatz ──────
        # Unlike every branch above, this one does NOT use a ramp envelope
        # times a correction — it parameterizes the annealing fraction s(t)
        # directly (h_driver = 1-s, h_target = s), as a piecewise-linear
        # curve: ramp up to a plateau, hold, ramp again, hold, ..., ramp to
        # s=1. The idea is to let the optimizer choose to linger (plateau)
        # near a hard avoided crossing so that Landau-Zener-Stückelberg
        # interference between the two crossing diabatic states can build up
        # constructively, instead of sweeping through at constant speed.
        #
        # Free parameters (all unconstrained reals, squashed below):
        #   raw_durations  (n_seg = 2M+1 of them) -> segment durations > 0,
        #                  via softplus, then renormalized to sum to tf.
        #   raw_splateaus  (M of them)            -> plateau heights in
        #                  (0,1), via sigmoid. These are s_way[1..M]; the
        #                  two endpoints s_way[0]=0 and s_way[M+1]=1 are
        #                  fixed boundary conditions, not free parameters.
        elif self.type == "LZS":
            # Direct s(t) parametrization: h_driver=1-s, h_target=s, so BOTH
            # depend on the FULL parameter vector — unlike the branches
            # above, where driver/target params are disjoint. Durations are
            # jointly softplus-normalized to sum to tf, so a change in any
            # single raw_duration_m shifts EVERY segment boundary, not just
            # its own segment — this couples all n_seg duration params
            # together in the Jacobian (see dTb below).
            M = dim
            n_seg = self._lzs_n_seg
            raw_durations = parameters[:n_seg]
            raw_splateaus = parameters[n_seg : n_seg + M]

            # Step 1 — decode segment durations.
            # softplus(raw_durations) > 0 guarantees positive durations;
            # dividing by their sum and multiplying by tf renormalizes them
            # to add up to exactly the total annealing time.
            D = _softplus(raw_durations)
            sig_D = _sigmoid(raw_durations)  # dD/d(raw_durations)
            Ssum = D.sum()
            scaled_durations = D / Ssum * tf
            t_bounds = np.concatenate(([0.0], np.cumsum(scaled_durations)))
            t_bounds[-1] = tf  # guard against fp drift

            # Step 2 — Jacobian of the segment boundary TIMES w.r.t. every
            # duration parameter. Because of the "divide by the sum" step,
            # perturbing ANY raw_durations[m] moves every later boundary, not
            # just segment m's own edges — this is the "normalization
            # coupling" flagged in the comment above.
            # dTb[k, m] = d(t_bounds[k]) / d(raw_durations[m])
            c = sig_D / Ssum
            ind = (np.arange(n_seg)[None, :] < np.arange(n_seg + 1)[:, None]).astype(
                np.float64
            )
            dTb = c[None, :] * (tf * ind - t_bounds[:, None])  # (n_seg+1, n_seg)

            # Step 3 — decode plateau heights via sigmoid into (0,1), and
            # assemble the full waypoint list s_way = [0, plateau_1, ...,
            # plateau_M, 1] (M+2 entries: the boundary values 0 and 1 are
            # NOT free parameters).
            sig_S = _sigmoid(raw_splateaus)
            dsig_S = sig_S * (1.0 - sig_S)  # d(sigmoid)/d(raw_splateaus)
            s_way = np.concatenate(([0.0], sig_S, [1.0]))  # (M+2,)

            # Step 4 — walk through the 2M+1 alternating ramp/plateau
            # segments, filling in s(t) and its Jacobian ds_dtheta segment
            # by segment. ds_dtheta packs BOTH parameter blocks into one
            # (n_params, nsteps) array: rows [0:n_seg] are duration
            # sensitivities, rows [n_seg:n_seg+M] are plateau-height
            # sensitivities.
            s = np.zeros_like(t)
            ds_dtheta = np.zeros((n_params, self.nsteps))

            for seg in range(n_seg):
                t0, t1 = t_bounds[seg], t_bounds[seg + 1]
                dt0, dt1 = dTb[seg, :], dTb[seg + 1, :]  # (n_seg,) each
                mask = (t >= t0) & (t <= t1)
                tm = t[mask]
                denom = (t1 - t0) if t1 > t0 else 1.0

                if seg % 2 == 0:
                    # Ramp segment (even index): linear interpolation
                    # between waypoint k and k+1, k = seg // 2.
                    k = seg // 2
                    s0, s1_ = s_way[k], s_way[k + 1]
                    frac = (tm - t0) / denom
                    s[mask] = s0 + (s1_ - s0) * frac

                    # -- duration sensitivity --
                    # s = s0 + (s1-s0) * (t - t0)/(t1 - t0); differentiating
                    # frac = (t-t0)/(t1-t0) w.r.t. any duration parameter via
                    # the quotient rule (using dt0 = d t0/dtheta,
                    # dt1 = d t1/dtheta from Step 2) gives dfrac below.
                    # Because dt0/dt1 have contributions from every duration
                    # parameter (Step 2), this couples the current segment's
                    # shape to ALL n_seg durations, not just its own.
                    dfrac = (
                        dt0[:, None] * (tm[None, :] - t1)
                        - dt1[:, None] * (tm[None, :] - t0)
                    ) / denom**2  # (n_seg, n_masked)
                    ds_dtheta[:n_seg, mask] += (s1_ - s0) * dfrac

                    # -- plateau-height sensitivity --
                    # s = (1-frac)*s0 + frac*s1, so ds/ds0 = (1-frac) and
                    # ds/ds1 = frac; chain through d(s_way)/d(raw_splateaus)
                    # = dsig_S. Only the two flanking waypoints of THIS ramp
                    # contribute — guarded so the fixed boundaries (index 0
                    # and index M+1 in s_way, which are not free parameters)
                    # never receive a gradient contribution.
                    if k - 1 >= 0:
                        ds_dtheta[n_seg + (k - 1), mask] += (1 - frac) * dsig_S[k - 1]
                    if k <= M - 1:
                        ds_dtheta[n_seg + k, mask] += frac * dsig_S[k]
                else:
                    # Plateau segment (odd index): s is held constant at
                    # s_way[k], k = (seg+1)//2, for the whole segment — so
                    # there is no time-dependence and hence NO duration
                    # sensitivity (a plateau's height doesn't change if you
                    # stretch or shrink how long it lasts).
                    k = (seg + 1) // 2
                    s[mask] = s_way[k]
                    if k - 1 >= 0:
                        ds_dtheta[n_seg + (k - 1), mask] += dsig_S[k - 1]

            # h_driver = 1 - s, h_target = s (no ramp envelope for LZS), so
            # their theta-Jacobians are just -ds_dtheta and +ds_dtheta.
            h_driver = 1.0 - s
            h_target = s
            dh_drv = -ds_dtheta
            dh_tgt = ds_dtheta

        return h_driver, h_target, dh_drv, dh_tgt

    # ─────────────────────────────────────────────────────────────────────────
    def get_driving(self, parameters=None) -> tuple:
        """
        Returns (h_driver, h_target) as numpy arrays, evaluated at
        `parameters` (or at self.parameters if not given). Convenience
        wrapper around _compute_driving_and_jacobian that discards the
        Jacobian — use this when you just want to plot/inspect the schedule.
        """
        if parameters is None:
            parameters = self.parameters
        h_drv, h_tgt, _, _ = self._compute_driving_and_jacobian(parameters)
        return h_drv, h_tgt

    # ─────────────────────────────────────────────────────────────────────────
    def get_lzs_waypoints(self):
        """
        Diagnostic helper (LZS type only): returns (t_bounds, s_way) — the
        segment boundary times and s-values actually used by get_driving(),
        decoded from the current self.parameters. Mirrors
        schedule_utils.Schedule.get_lzs_waypoints() / JaxSchedule's version.

        Useful for: plotting the piecewise-linear schedule directly from
        its waypoints, or comparing two optimized parameter sets to check
        whether they represent genuinely different schedules (as opposed to
        a relabelling/gauge-duplicate of the same waypoints — see the
        "gauge-duplicate dedup check" item in the project's open tasks,
        which is built on top of this method).

        Returns
        -------
        t_bounds : (2M+2,) ndarray — segment boundary times, 0..tf
        s_way    : (M+2,) ndarray  — waypoint s-values, s_way[0]=0, s_way[-1]=1
        """
        if self.type != "LZS":
            raise ValueError("get_lzs_waypoints() only valid for type='LZS'")
        dim = self.number_parameters
        M = dim
        n_seg = self._lzs_n_seg
        raw_durations = self.parameters[:n_seg]
        raw_splateaus = self.parameters[n_seg : n_seg + M]

        durations = np.log1p(np.exp(raw_durations))
        durations = durations / durations.sum() * self.tf
        t_bounds = np.concatenate(([0.0], np.cumsum(durations)))
        t_bounds[-1] = self.tf

        s_plateaus = 1.0 / (1.0 + np.exp(-raw_splateaus))
        s_way = np.concatenate(([0.0], s_plateaus, [1.0]))
        return t_bounds, s_way

    # ─────────────────────────────────────────────────────────────────────────
    def _forward_and_grad(self, parameters: np.ndarray, compute_grad: bool = True):
        """
        Core GRAPE computation.

        Forward pass  : |ψ_0⟩ → |ψ_1⟩ → ... → |ψ_T⟩  via sparse expm_multiply
        Energy        : E = ⟨ψ_T| H_ref |ψ_T⟩
        Backward pass : |χ_T⟩ = H_ref|ψ_T⟩ propagated backward
        GRAPE         : dE/dh_x_i = -2 dt Im[⟨χ_i| H_x |ψ_i⟩]
        Chain rule    : grad[k] = Σ_i dE/dh_drv_i * ∂h_drv_i/∂θ_k
                                + Σ_i dE/dh_tgt_i * ∂h_tgt_i/∂θ_k

        This is the quantum-control analogue of backprop: the forward pass
        is the "network", the co-state |χ_i> plays the role of the
        upstream gradient signal being propagated backward through each
        timestep, and dE/dh_x_i is the local gradient at that "layer"
        (timestep) — exact, no finite differences, same asymptotic cost as
        one extra forward pass.

        Parameters
        ----------
        parameters : (n_params,) ndarray
            theta to evaluate at (does NOT mutate self.parameters — callers
            like forward_and_gradient() do that separately).
        compute_grad : bool
            If False, skip the backward pass entirely (cheaper — used when
            an optimizer only needs a bare energy evaluation).

        Returns
        -------
        energy : float
        grad   : (n_params,) ndarray, or None if compute_grad=False
        """
        h_driver, h_target, dh_drv_dtheta, dh_tgt_dtheta = (
            self._compute_driving_and_jacobian(parameters)
        )
        dt = self.dt

        # ── forward pass ──────────────────────────────────────────────────────
        # Standard piecewise-constant-Hamiltonian propagation: freeze H(t)
        # over each small interval dt and apply exp(-i dt H) exactly via
        # scipy's Krylov-subspace expm_multiply (avoids ever forming the
        # dense matrix exponential, which is essential once dim gets large).
        psi = self._psi_init.copy()
        psi_fwd = [psi.copy()]  # psi_fwd[i] = |ψ_i⟩ before step i

        for i in range(self.nsteps):
            H_t = h_driver[i] * self._H_driver + h_target[i] * self._H_target
            psi = expm_multiply(-1j * dt * H_t, psi)
            psi_fwd.append(psi.copy())

        psi_final = psi_fwd[-1]
        self.psi = psi_final.copy()
        energy = (psi_final.conj() @ self._H_ref @ psi_final).real

        if not compute_grad:
            return energy, None

        # ── backward pass ─────────────────────────────────────────────────────
        # initial co-state: |χ_T⟩ = H_ref |ψ_T⟩
        # The co-state chi plays the role of an "error signal" that we
        # propagate backward in time using the ADJOINT (time-reversed)
        # unitary: since H is Hermitian, the adjoint of exp(-i dt H) is
        # exp(+i dt H), so we simply flip the sign in the exponent — no
        # separate adjoint machinery required.
        chi = self._H_ref @ psi_final

        dE_dh_drv = np.zeros(self.nsteps)
        dE_dh_tgt = np.zeros(self.nsteps)

        for i in reversed(range(self.nsteps)):
            H_t = h_driver[i] * self._H_driver + h_target[i] * self._H_target

            # propagate co-state one step backward (adjoint = +1j for Hermitian H)
            chi = expm_multiply(+1j * dt * H_t, chi)

            psi_i = psi_fwd[i]

            # GRAPE: dE/dh_x_i = -2 dt Im[ ⟨ψ_i | H_x | χ_i⟩ ]
            # This is the standard GRAPE per-timestep gradient formula: the
            # sensitivity of the final energy to a small perturbation of the
            # driving strength h_x at step i, expressed purely in terms of
            # the forward state psi_i and backward co-state chi at that same
            # step — no need to re-run the whole propagation for each
            # parameter.
            dE_dh_drv[i] = -2.0 * dt * (psi_i.conj() @ self._H_driver @ chi).imag
            dE_dh_tgt[i] = -2.0 * dt * (psi_i.conj() @ self._H_target @ chi).imag

        # ── chain rule through schedule ───────────────────────────────────────
        # grad[k] = Σ_i [ dE/dh_drv_i * ∂h_drv_i/∂θ_k
        #               + dE/dh_tgt_i * ∂h_tgt_i/∂θ_k ]
        # This is where the per-timestep GRAPE gradient (dE_dh_drv/dh_tgt,
        # length nsteps) gets converted into a per-PARAMETER gradient (grad,
        # length n_params) using the ansatz Jacobian computed at the very
        # top of this function — a single matrix-vector product per schedule
        # branch, thanks to _compute_driving_and_jacobian doing the
        # analytic differentiation up front.
        grad = dh_drv_dtheta @ dE_dh_drv + dh_tgt_dtheta @ dE_dh_tgt  # (n_params,)

        return energy, grad

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface — identical to JaxSchedulerModel
    # ─────────────────────────────────────────────────────────────────────────

    def forward_and_gradient(self, parameters: np.ndarray):
        """
        Returns (energy, grad) together — pass as jac=True to scipy minimize.

        Also updates self.parameters/self.energy and increments
        self.run_number, so this is the method to call from an optimizer
        (as opposed to _forward_and_grad, which is side-effect-light and
        intended for internal use).
        """
        self.parameters = parameters
        energy, grad = self._forward_and_grad(parameters, compute_grad=True)
        self.energy = energy
        self.run_number += 1
        return energy, grad

    def forward(self, parameters: np.ndarray) -> float:
        """Energy-only evaluation (no backward pass) — cheaper than forward_and_gradient."""
        self.parameters = parameters
        energy, _ = self._forward_and_grad(parameters, compute_grad=False)
        self.energy = energy
        self.run_number += 1
        return energy

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Gradient-only evaluation. Note: unlike forward()/forward_and_gradient(),
        this does NOT update self.parameters/self.energy/self.run_number."""
        _, grad = self._forward_and_grad(parameters, compute_grad=True)
        return grad

    def callback(self, *args):
        """
        Optimizer callback (invoked once per ACCEPTED L-BFGS-B iteration,
        not per internal line-search evaluation). Appends a snapshot of the
        current energy/parameters/schedule/state to the history_* lists and
        prints the current energy for live monitoring.
        """
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters.copy())
        self.history_drivings.append(self.get_driving())
        if self.psi is not None:
            self.history_psi.append(self.psi.copy())
        self.history_run.append(self.run_number)
        print(self.energy)

    def load(self, parameters: np.ndarray):
        """Overwrite self.parameters with a previously saved vector (e.g. to
        resume/inspect a converged optimization). Raises on shape mismatch."""
        if parameters.shape[0] == self.parameters.shape[0]:
            self.parameters = parameters.copy()
        else:
            raise ValueError(
                f"Shape mismatch: got {parameters.shape[0]}, "
                f"expected {self.parameters.shape[0]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class SparseGRAPETrainer:
    """
    Drop-in replacement for JaxTrainer.

    Uses jac=True so scipy gets (energy, grad) in a single call,
    avoiding a duplicate forward pass per iteration.
    """

    def __init__(
        self,
        model: SparseGRAPEModel,
        maxiter: int = 500,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
        tol: float = 1e-3,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        model : SparseGRAPEModel
            The model to optimize (mutated in place — run() leaves
            model.parameters/psi/history_* at the optimizer's final state).
        maxiter, ftol, gtol : passed straight through to scipy's L-BFGS-B
            (see scipy.optimize.minimize options for exact semantics).
        tol : float
            Currently unused by run() (L-BFGS-B is configured via ftol/gtol
            instead) — kept for interface parity with JaxTrainer.
        verbose : bool
            If True, prints progress every accepted iteration (via
            model.callback) and runs the finite-difference gradient check
            described in run().
        """
        self.model = model
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose

    def run(self) -> dict:
        """
        Run L-BFGS-B with GRAPE gradients.
        Returns dict with identical keys to JaxTrainer.run().

        Before optimizing, if verbose, does a one-parameter finite-difference
        sanity check: perturbs parameter 0 by eps and compares the resulting
        finite-difference slope to the analytic GRAPE gradient at that same
        entry. This is a cheap way to catch a broken Jacobian (e.g. after
        editing _compute_driving_and_jacobian) before spending time on a
        full optimization run — a large relative error here means something
        is wrong with the analytic gradient, not with the optimizer.
        """
        # ── gradient check at initial point ───────────────────────────────────
        p0 = self.model.parameters.copy()
        e0, g0 = self.model.forward_and_gradient(p0)
        if self.verbose:
            print(f"  Initial energy    : {e0:.6f}")
            print(f"  Gradient norm     : {np.linalg.norm(g0):.6e}")
            # finite difference check on first parameter
            eps = 1e-5
            p_plus = p0.copy()
            p_plus[0] += eps
            e_plus, _ = self.model.forward_and_gradient(p_plus)
            fd = (e_plus - e0) / eps
            print(f"  FD  grad[0]       : {fd:.6e}")
            print(f"  GRAPE grad[0]     : {g0[0]:.6e}")
            print(f"  Relative error    : {abs(fd - g0[0]) / (abs(fd) + 1e-15):.3e}")
        # reset state after gradient check
        self.model.forward_and_gradient(p0)

        res = minimize(
            self.model.forward_and_gradient,
            self.model.parameters,
            jac=True,  # forward_and_gradient returns (f, grad) together
            method="L-BFGS-B",
            callback=self.model.callback if self.verbose else None,
            options={
                "maxiter": self.maxiter,
                "ftol": self.ftol,
                "gtol": self.gtol,
            },
        )

        # sync final state
        self.model.forward_and_gradient(res.x)
        h_driver, h_target = self.model.get_driving()

        if self.verbose:
            print(f"\nOptimization success : {res.success}")
            print(f"Final energy         : {res.fun:.6f}")
            print(f"Message              : {res.message}")

        return {
            "success": bool(res.success),
            "message": res.message,
            "n_iterations": int(res.nit),
            "n_evals": int(res.nfev),
            "energy": float(res.fun),
            "parameters": np.array(res.x),
            "psi": self.model.psi.copy(),
            "h_driver": h_driver,
            "h_target": h_target,
            "time": self.model.time.copy(),
            "history_energy": list(self.model.history),
            "history_parameters": [p.copy() for p in self.model.history_parameters],
            "history_drivings": self.model.history_drivings,
            "history_psi": [p.copy() for p in self.model.history_psi],
        }
