"""
src/schedule_utils.py

Gradient-FREE counterpart to src/sparse_grape_method.py.

`Schedule` defines the same family of schedule ansätze (power law,
Fourier/F-CRAB, LZS) and `SchedulerModel` propagates them exactly like
`SparseGRAPEModel` does (piecewise-constant Hamiltonian, sparse
`expm_multiply`) — but neither class computes an analytic gradient. There
is no `_compute_driving_and_jacobian`, no backward/co-state pass: only
`get_driving()` (schedule only) and `forward()` (energy only).

That makes this module the natural home for a DERIVATIVE-FREE optimizer
(`SchedulerTrainer`, added at the bottom of this file) — useful as:
  - a quick, gradient-free baseline/sanity-check before investing in GRAPE,
  - a cheap way to explore an ansatz GRAPE doesn't (yet) support,
  - a coarse starting point whose converged parameters can seed a
    subsequent SparseGRAPETrainer run for fine refinement.

IMPORTANT DIFFERENCES FROM src/sparse_grape_method.py (read before treating
results from the two modules as interchangeable):

1. `SchedulerModel` now takes `initial_state` as an explicit constructor
   argument, exactly like `SparseGRAPEModel` — the caller is responsible
   for supplying the correct psi(t=0) (e.g. `psi_init_s` in
   study_1d_ising.py: the all-plus state projected into the Z2 symmetric
   sector, NOT a naive uniform superposition over the truncated sector
   basis — those are different states; see the Z2SymmetricSector vs
   Sector distinction in src/utils.py). Passing the SAME `initial_state`
   here and to SparseGRAPEModel is what makes a SchedulerTrainer result a
   valid warm start for a subsequent SparseGRAPETrainer run. (Previously
   this class hardcoded a plain `np.ones(dim)/sqrt(dim)` inside forward(),
   which is only correct for an unprojected transverse-field driver — for
   a symmetric-sector-projected system it silently used the wrong state.
   No existing call site in this repo depended on the old default, so this
   was changed to a required argument rather than a defaulted one.)

2. The Fourier/F-CRAB correction here is a MEAN over the `dim` frequency
   components (`np.mean(matrix_driver, axis=0)`), whereas
   `SparseGRAPEModel`'s Fourier branch is a plain SUM (matrix-vector
   product). Same-magnitude parameters therefore produce a ~dim-times
   smaller correction amplitude here than in the GRAPE module.

3. `Schedule.load()` calls `exit()` on a shape mismatch (kills the whole
   process/kernel) rather than raising an exception like
   `SparseGRAPEModel.load()` does — be careful calling this from a
   notebook.

4. `SchedulerModel.depolarization_option()` only stores
   `noise_option`/`noise_coupling` as attributes; `forward()` never reads
   them. Noise is not actually wired into the propagation yet — this is a
   hook for future work, not a working feature.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply


def configuration(res, energy, grad_energy):
    """
    Quick console summary after an optimization run.

    Kept for interface parity with older gradient-based scripts that pass
    a (res, energy, grad_energy) triple; note `grad_energy` is meaningless
    for the gradient-free path in this file (SchedulerTrainer does not
    produce one) — only call this helper from gradient-based code, or pass
    a dummy array if you want the same printout format here.
    """
    print("Optimization Success=", res.success)
    print(f"energy={energy:.5f}")
    print(f"average gradient={np.average(grad_energy):.5f} \n")


class Schedule:
    """
    Decodes a flat parameter vector theta into a schedule (h_driver(t),
    h_target(t)) for one of three ansätze — power law, Fourier/F-CRAB, or
    the LZS M-plateau interferometer. Pure bookkeeping/math, no physics
    (no Hamiltonians) — see SchedulerModel below for the class that
    actually propagates a wavefunction under the resulting schedule.
    """

    def __init__(
        self,
        tf: float,
        type: str,
        number_of_parameters: int,
        nsteps: int,
        seed: Optional[int] = None,
        mode: Optional[str] = "annealing ansatz",
        random: Optional[bool] = False,
    ):
        """
        Parameters
        ----------
        tf : float
            Total annealing time.
        type : str
            Ansatz choice — 'power law', 'F-CRAB', 'fourier', or 'LZS'.
        number_of_parameters : int
            The "dim" knob for the chosen ansatz. For 'LZS' this is M (the
            number of plateaus), NOT the total parameter count — see the
            sizing block below.
        nsteps : int
            Number of points in the (uniform) time discretization.
        seed : int, optional
            Present for interface parity; note the actual random draws
            below (`np.random.uniform(...)`) use the global numpy RNG, not
            a seeded `Generator` built from this argument — so `seed` is
            currently NOT wired up to reproducibility here (unlike
            SparseGRAPEModel, which uses `np.random.default_rng(seed)`).
        mode : str, optional
            'annealing ansatz' (linear ramp envelope × correction) is the
            only mode actually treated differently in comments; in
            practice this attribute is stored but not read anywhere in
            get_driving() below — the ramp-envelope formula is always
            applied for power-law/Fourier types regardless of `mode`.
        random : bool
            If True, initialize parameters ~ Uniform(-2, 2) instead of
            all-zeros (a wider range than SparseGRAPEModel's
            Uniform(-0.5, 0.5) default).
        """
        self.tf = tf
        self.nsteps = nsteps
        self.type = type
        self.mode = mode
        self.time = np.linspace(0, self.tf, nsteps)
        self.number_parameters = number_of_parameters
        self.seed = seed

        dim = number_of_parameters  # alias for clarity below

        # ── parameter vector sizing ──────────────────────────────────────────
        # Both 'annealing ansatz' and free mode optimise driver AND target.
        # The mode only affects the *shape* of the schedule (linear ramp
        # envelope vs fully free), not how many parameters are used.
        #   F-CRAB / fourier:  4*dim  (sin+cos for driver, sin+cos for target)
        #   power law:         2*dim  (coefficients for driver and target)
        #   LZS:                3*dim+1  (dim = M plateaus/interferometer arms;
        #                       see get_driving() for the parameter layout).
        #                       LZS parameterizes s(t) directly (boundary
        #                       conditions s(0)=0, s(tf)=1 built in), so
        #                       'mode' has no effect for this type.
        if self.type in ("F-CRAB", "fourier"):
            n_params = 4 * dim
        elif self.type == "LZS":
            n_params = 3 * dim + 1
        else:  # power law
            n_params = 2 * dim

        self.parameters = np.zeros(n_params)
        if random:
            self.parameters = np.random.uniform(-2, 2, size=n_params)

        # ── random frequencies (fix: dim was undefined in 'fourier' branch) ──
        # NOTE: both 'F-CRAB' and 'fourier' apply the SAME randomised-
        # frequency perturbation here. In the original CRAB/F-CRAB
        # distinction, only F-CRAB ("free" CRAB) is supposed to randomise
        # frequencies — plain 'fourier' would normally use the exact
        # harmonic grid 2*pi*k/tf. As written, choosing type='fourier'
        # behaves identically to 'F-CRAB' (both get randomised omegas, and
        # both draw from the unseeded global RNG — see the `seed` note
        # above). Left as-is here since changing it would alter existing
        # results; flagging in case a genuinely non-randomised 'fourier'
        # variant was intended.
        if self.type == "F-CRAB":
            self.omegas = (
                2
                * np.pi
                * np.arange(1, dim + 1)
                * (1 + np.random.uniform(-0.5, 0.5, dim))
                / self.tf
            )
        if self.type == "fourier":
            self.omegas = (
                2
                * np.pi
                * np.arange(1, dim + 1)
                * (1 + np.random.uniform(-0.5, 0.5, dim))
                / self.tf
            )

    # ─────────────────────────────────────────────────────────────────────────
    def get_driving(self) -> np.ndarray:
        """
        Decode self.parameters into (h_driver, h_target) at every point of
        self.time, for whichever ansatz self.type selects. Pure function of
        self.parameters/self.type — no history side effects (contrast with
        SchedulerModel.forward(), which also propagates a wavefunction).

        Returns
        -------
        h_driver, h_target : each (nsteps,) ndarray
        """
        dim = self.number_parameters
        t = self.time
        tf = self.tf

        # ── compute Fourier / power-law correction matrices ──────────────────
        if self.type == "power law":
            # Polynomial correction basis (t/tf)^(k+1), k=1..dim — a smooth,
            # low-frequency-biased alternative to the Fourier ansatz below.
            exponents = np.arange(1, dim + 1)[:, None]  # (dim, 1)
            basis = (t[None, :] / tf) ** exponents  # (dim, nsteps)
            matrix_driver = self.parameters[:dim, None] * basis
            matrix_target = self.parameters[dim : 2 * dim, None] * basis
            # falls through to the shared "build physical schedules" block
            # at the bottom of this method

        elif self.type in ("F-CRAB", "fourier"):
            # parameters = [a_drv(dim), b_drv(dim), a_tgt(dim), b_tgt(dim)]
            sin_basis = np.sin(t[None, :] * self.omegas[:, None])  # (dim, nsteps)
            cos_basis = np.cos(t[None, :] * self.omegas[:, None])
            matrix_driver = (
                self.parameters[:dim, None] * sin_basis
                + self.parameters[dim : 2 * dim, None] * cos_basis
            )
            matrix_target = (
                self.parameters[2 * dim : 3 * dim, None] * sin_basis
                + self.parameters[3 * dim : 4 * dim, None] * cos_basis
            )

            # NOTE: mean, not sum, over the dim frequency components — this
            # is a scale convention DIFFERENT from SparseGRAPEModel's
            # Fourier branch (plain sum via matrix-vector product). Same
            # parameter magnitudes give a ~dim-times smaller correction
            # amplitude here. See module docstring point 2.
            correction_driver = np.mean(matrix_driver, axis=0)  # (nsteps,)
            correction_target = np.mean(matrix_target, axis=0)

            # linear ramp envelope × (1 + correction) — early return since
            # this branch's formula is identical to the shared block below
            # but was written out separately (kept as in the source).
            h_driver = (1 - t / tf) * ((1 + correction_driver))
            h_target = (t / tf) * ((1 + correction_target))

            return h_driver, h_target

        elif self.type == "LZS":
            # ── Landau-Zener-Stückelberg interference ansatz ────────────────
            # Generalization of Werner, Jonsson, García-Sáez, Riera & Albas
            # (2026) from a single interferometer arm (M=1, 7 params) to M
            # plateaus/arms. Directly parameterizes s(t) (not a Fourier
            # correction on top of a linear ramp), with:
            #   - (2M+1) segment durations: ramp0, plateau1, ramp1, ..., rampM
            #   - M plateau heights s_1...s_M in (0,1)
            # Reparametrized via softplus/sigmoid so any raw parameter vector
            # from an unconstrained optimizer maps to a valid schedule
            # (positive durations summing to tf, plateau heights in (0,1)).
            # This is exactly the schedule shape computed (without a
            # Jacobian) inside SparseGRAPEModel._compute_driving_and_jacobian
            # for type='LZS' — see that file for a fully worked walkthrough
            # of the waypoint/segment bookkeeping below.
            M = dim
            n_seg = 2 * M + 1
            raw_durations = self.parameters[:n_seg]
            raw_splateaus = self.parameters[n_seg : n_seg + M]

            # softplus -> strictly positive, normalized to sum to tf
            durations = np.log1p(np.exp(raw_durations))
            durations = durations / durations.sum() * tf
            t_bounds = np.concatenate(([0.0], np.cumsum(durations)))
            t_bounds[-1] = tf  # guard against floating-point drift

            # sigmoid -> plateau heights in (0,1); waypoints include the
            # fixed endpoints s(0)=0 and s(tf)=1
            s_plateaus = 1.0 / (1.0 + np.exp(-raw_splateaus))
            s_way = np.concatenate(([0.0], s_plateaus, [1.0]))  # length M+2

            s = np.zeros_like(t)
            for seg in range(n_seg):
                t0, t1 = t_bounds[seg], t_bounds[seg + 1]
                mask = (t >= t0) & (t <= t1)
                if seg % 2 == 0:
                    # ramp segment: linear interpolation between flanking waypoints
                    k = seg // 2
                    s_start, s_end = s_way[k], s_way[k + 1]
                    denom = (t1 - t0) if t1 > t0 else 1.0
                    frac = (t[mask] - t0) / denom
                    s[mask] = s_start + (s_end - s_start) * frac
                else:
                    # plateau segment: constant value
                    k = (seg + 1) // 2
                    s[mask] = s_way[k]

            h_driver = 1.0 - s
            h_target = s
            return h_driver, h_target

        else:
            raise ValueError(f"Unknown schedule type: '{self.type}'")

        # ── build physical schedules (power law / non-LZS path) ──────────────
        # Only reached by the 'power law' branch (Fourier/F-CRAB and LZS
        # both return early above).
        # 'annealing ansatz': linear ramp envelope x (1 + Fourier correction)
        # 'free': same formula, no constraint — optimizer is free to deform both
        correction_driver = np.mean(matrix_driver, axis=0)  # (nsteps,)
        correction_target = np.mean(matrix_target, axis=0)

        h_driver = (1 - t / tf) * ((1 + correction_driver))
        h_target = (t / tf) * ((1 + correction_target))

        return h_driver, h_target

    # ─────────────────────────────────────────────────────────────────────────
    def get_lzs_waypoints(self):
        """
        Diagnostic helper (LZS type only): returns (t_bounds, s_way) — the
        segment boundary times and s-values actually used by get_driving(),
        decoded from the current self.parameters. Useful for overlaying the
        AC location / gap data on top of the schedule during tuning.
        Identical decoding logic to SparseGRAPEModel.get_lzs_waypoints().
        """
        if self.type != "LZS":
            raise ValueError("get_lzs_waypoints() only valid for type='LZS'")
        dim = self.number_parameters
        M = dim
        n_seg = 2 * M + 1
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
    def load(self, parameters: np.ndarray):
        """
        Overwrite self.parameters with a previously saved vector.

        CAUTION: on a shape mismatch this calls exit() — which terminates
        the whole Python process (or kills a Jupyter kernel) rather than
        raising a catchable exception. SparseGRAPEModel.load() raises
        ValueError instead; consider aligning the two if this is ever
        called somewhere a hard process exit is unacceptable (e.g. inside
        a long-running notebook session or a loop over many instances).
        """
        if parameters.shape[0] == self.parameters.shape[0]:
            self.parameters = parameters
        else:
            print(
                f"Shape mismatch: got {parameters.shape[0]}, "
                f"expected {self.parameters.shape[0]}"
            )
            exit()


# ─────────────────────────────────────────────────────────────────────────────
class SchedulerModel(Schedule):
    """
    Adds physical propagation on top of Schedule: given the three sparse
    Hamiltonians (driver, target, reference) it evolves a wavefunction
    under H(t) = h_driver(t)*initial_hamiltonian + h_target(t)*target_hamiltonian
    and evaluates the energy w.r.t. reference_hamiltonian at t=tf.

    Energy-only — there is no analytic gradient here (contrast with
    SparseGRAPEModel, which additionally runs a backward/co-state pass to
    get dE/dtheta for free). Pair this class with the gradient-free
    SchedulerTrainer defined at the bottom of this file.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        target_hamiltonian: scipy.sparse.spmatrix,
        initial_hamiltonian: scipy.sparse.spmatrix,
        reference_hamiltonian: scipy.sparse.spmatrix,
        tf: float,
        number_of_parameters: int,
        nsteps: int,
        type: str,
        seed: int,
        mode: Optional[str] = "annealing ansatz",
        random: Optional[bool] = False,
    ):
        """
        Parameters
        ----------
        initial_state : (dim,) array-like
            psi(t=0). MUST be consistent with whatever Hilbert space
            `initial_hamiltonian`/`target_hamiltonian` live in — e.g. if
            those are projected into a symmetry sector (see
            src/utils.py Z2SymmetricSector), this must be the correctly
            projected initial state (PS.project(...)), not a naive
            uniform superposition over the sector's truncated basis
            index — those are NOT the same state. Pass the identical
            array here and to SparseGRAPEModel if you want a
            SchedulerTrainer result to be a valid GRAPE warm start.
        target_hamiltonian, initial_hamiltonian : sparse (dim, dim)
            H_target (problem Hamiltonian, coefficient h_target(t)) and
            H_driver (mixer Hamiltonian, coefficient h_driver(t)).
        reference_hamiltonian : sparse (dim, dim)
            Hamiltonian the final energy is evaluated against, i.e.
            E = <psi(tf)| reference_hamiltonian |psi(tf)>.
        tf, number_of_parameters, nsteps, type, seed, mode, random :
            forwarded straight to Schedule.__init__ — see its docstring.
        """
        self.target_hamiltonian = target_hamiltonian
        self.initial_hamiltonian = initial_hamiltonian
        self.reference_hamiltonian = reference_hamiltonian
        self._psi_init = np.asarray(initial_state, dtype=complex)

        super().__init__(
            tf=tf,
            type=type,
            number_of_parameters=number_of_parameters,
            nsteps=nsteps,
            seed=seed,
            mode=mode,
            random=random,
        )

        self.energy = 1000.0
        self.psi = None

        # memory
        self.history = []
        self.history_psi = []
        self.history_drivings = []
        self.history_parameters = []
        self.history_run = []
        self.run_number = 0

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, parameters):
        """
        Energy-only forward propagation at the given parameters.

        Same piecewise-constant-Hamiltonian / sparse expm_multiply scheme
        as SparseGRAPEModel._forward_and_grad's forward pass:
          - psi_init is the caller-supplied `initial_state` (stored as
            self._psi_init in __init__) — same convention as
            SparseGRAPEModel, so a converged schedule here starts from the
            same physical state as a GRAPE run given the same
            initial_state;
          - there is no backward pass / gradient — this is the method a
            derivative-free optimizer (SchedulerTrainer, below) calls
            directly as its objective function.

        Side effects: updates self.parameters, self.energy, self.psi, and
        increments self.run_number (mirrors SparseGRAPEModel.forward()).

        Parameters
        ----------
        parameters : (n_params,) ndarray
            theta to evaluate at.

        Returns
        -------
        energy : float
        """
        dt = self.time[1] - self.time[0]
        self.parameters = parameters

        hamiltonians = [self.initial_hamiltonian, self.target_hamiltonian]
        h_driver, h_target = self.get_driving()  # pre-compute once
        schedules = [h_driver, h_target]

        # sanity check: initial_state must live in the same Hilbert space
        # as the Hamiltonians (a mismatch here usually means the wrong
        # sector projection was passed in — see the initial_state note in
        # __init__'s docstring).
        dim = self.initial_hamiltonian.shape[0]
        if self._psi_init.shape[0] != dim:
            raise ValueError(
                f"initial_state has dim {self._psi_init.shape[0]}, but "
                f"initial_hamiltonian has dim {dim} — check that both are "
                f"projected into the same (sub)space."
            )
        psi = self._psi_init.copy()
        for i in range(self.nsteps):
            # H(t_i) = h_driver_i * H_driver + h_target_i * H_target, built
            # fresh each step from the two schedules and Hamiltonians.
            time_hamiltonian = sum(schedules[r][i] * hamiltonians[r] for r in range(2))
            psi = expm_multiply(-1j * dt * time_hamiltonian, psi)

        # .real avoids ComplexWarning and satisfies scipy's scalar requirement
        self.energy = (psi.conjugate() @ self.reference_hamiltonian.dot(psi)).real
        self.psi = psi
        self.run_number += 1

        return self.energy

    # ─────────────────────────────────────────────────────────────────────────
    def callback(self, *args):
        """
        Optimizer callback — appends a snapshot of the current
        energy/parameters/schedule/state to the history_* lists and prints
        the current energy. Signature accepts arbitrary *args since
        different scipy.optimize methods invoke callbacks with different
        call signatures (e.g. just `xk`, or an OptimizeResult).
        """
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters.copy())
        self.history_drivings.append(self.get_driving())
        self.history_psi.append(self.psi.copy())
        self.history_run.append(self.run_number)
        print(self.energy)

    # ─────────────────────────────────────────────────────────────────────────
    def depolarization_option(self, noise_option: bool, noise_coupling: float):
        """
        Stores a noise flag/coupling on the instance. NOT currently wired
        into forward() — calling this has no effect on the propagation
        yet. Placeholder hook for a future noisy-evolution extension.
        """
        self.depolarization_coupling = noise_coupling
        self.noise_option = noise_option


# ─────────────────────────────────────────────────────────────────────────────
class SchedulerTrainer:
    """
    Gradient-FREE trainer for SchedulerModel — the derivative-free
    counterpart to SparseGRAPETrainer.

    Use this when:
      - GRAPE's analytic gradient isn't needed (small system, quick
        exploratory run, or you just want a fast sanity-check on whether a
        given ansatz/tf/number_of_parameters combination can reach a
        reasonable energy at all before investing in the GRAPE machinery);
      - you want a schedule type SparseGRAPEModel doesn't (yet) implement;
      - you want a cheap, rough starting point whose converged parameters
        can seed a subsequent SparseGRAPETrainer run for exact-gradient
        fine refinement (pass the SAME initial_state to both models — see
        the Fourier-scaling caveat documented at the top of this file
        before doing that, since that part still differs between the two
        modules).

    SchedulerModel.forward() returns only an energy — no gradient — so
    this trainer uses one of scipy's derivative-free local optimizers
    (Nelder-Mead by default, matching the classical dCRAB/simplex approach
    used in the optimal-control literature this project's ansätze are
    drawn from; Powell is offered as a usually-faster alternative for
    smooth, well-behaved landscapes). Every parameter in every ansatz here
    (power law / Fourier / F-CRAB coefficients, and the softplus/sigmoid-
    reparameterized LZS durations & plateau heights) is an UNCONSTRAINED
    real number — the ansatz itself enforces positivity/bounds internally
    — so no bounds need to be passed to the optimizer.
    """

    def __init__(
        self,
        model: SchedulerModel,
        maxiter: int = 2000,
        method: str = "Nelder-Mead",
        xatol: float = 1e-6,
        fatol: float = 1e-8,
        adaptive: bool = True,
        verbose: bool = True,
        options: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        model : SchedulerModel
            The model to optimize (mutated in place — run() leaves
            model.parameters/psi/history_* at the optimizer's final state,
            exactly like SparseGRAPETrainer does for SparseGRAPEModel).
        maxiter : int
            Max iterations. Derivative-free methods typically need
            noticeably more iterations than a gradient-based method to
            reach comparable precision — the default here (2000) is set
            higher than SparseGRAPETrainer's default (500) accordingly.
        method : str
            Any gradient-free scipy.optimize.minimize method, e.g.
            'Nelder-Mead' (default, robust simplex search — good first
            choice, especially for noisy or low-dimensional landscapes),
            'Powell' (direction-set search, often converges faster on
            smooth landscapes and scales better with parameter count),
            or 'COBYLA' (supports nonlinear constraints, not needed here
            but available if you extend the ansatz with explicit
            constraints later).
        xatol, fatol : float
            Convergence tolerances passed to Nelder-Mead specifically
            (absolute tolerance on parameters / on function value between
            simplex iterations). Ignored by other methods — pass
            method-specific tolerances via `options` instead if you switch
            methods (e.g. Powell uses `xtol`/`ftol`).
        adaptive : bool
            Nelder-Mead-specific: rescales the simplex operations for the
            problem dimension, which scipy recommends for problems with
            more than a handful of parameters (LZS with several plateaus,
            or a high-dim Fourier series, both qualify). Ignored by other
            methods.
        verbose : bool
            If True, prints progress every accepted iteration (via
            model.callback) and an initial-energy line before optimizing.
        options : dict, optional
            Extra/override entries merged into the options dict passed to
            scipy.optimize.minimize — use this to pass method-specific
            options not covered by the named arguments above (e.g.
            {"xtol": 1e-6, "ftol": 1e-8} for Powell).
        """
        self.model = model
        self.maxiter = maxiter
        self.method = method
        self.xatol = xatol
        self.fatol = fatol
        self.adaptive = adaptive
        self.verbose = verbose
        self.options = options or {}

    def run(self) -> dict:
        """
        Run the configured derivative-free optimizer on model.forward.
        Returns a dict with the SAME keys as SparseGRAPETrainer.run() /
        JaxTrainer.run(), so downstream plotting/analysis code can treat
        the two trainers' outputs interchangeably (aside from the
        psi_init / Fourier-scaling caveats noted at the top of this file).
        """
        p0 = self.model.parameters.copy()
        e0 = self.model.forward(p0)
        if self.verbose:
            print(f"  Method            : {self.method}")
            print(f"  Initial energy    : {e0:.6f}")

        # ── assemble method-specific options ────────────────────────────────
        opts = {"maxiter": self.maxiter, "disp": self.verbose}
        if self.method == "Nelder-Mead":
            opts.update(
                {
                    "xatol": self.xatol,
                    "fatol": self.fatol,
                    "adaptive": self.adaptive,
                }
            )
        opts.update(self.options)  # caller overrides always win

        res = minimize(
            self.model.forward,
            self.model.parameters,
            method=self.method,
            callback=self.model.callback if self.verbose else None,
            options=opts,
        )

        # sync final state (mirrors SparseGRAPETrainer.run())
        self.model.forward(res.x)
        h_driver, h_target = self.model.get_driving()

        if self.verbose:
            print(f"\nOptimization success : {res.success}")
            print(f"Final energy         : {res.fun:.6f}")
            print(f"Message              : {res.message}")

        return {
            "success": bool(res.success),
            "message": str(res.message),
            # not every gradient-free method reports nit/nfev the same way
            # as L-BFGS-B — fall back gracefully if a field is absent.
            "n_iterations": int(getattr(res, "nit", -1)),
            "n_evals": int(getattr(res, "nfev", -1)),
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
