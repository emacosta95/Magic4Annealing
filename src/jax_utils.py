import numpy as np
import scipy
from scipy.optimize import minimize
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

# enable float64 — essential for physics accuracy
jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
class JaxSchedule:
    """
    Mirrors the Schedule class in schedule_utils.py.
    Stores parameters as numpy arrays for scipy compatibility,
    converts to jax internally only when needed.
    """

    def __init__(
        self,
        tf: float,
        type: str,
        number_of_parameters: int,
        nsteps: int,
        seed: Optional[int] = None,
        mode: Optional[str] = 'annealing ansatz',
        random: Optional[bool] = False,
    ):
        self.tf = tf
        self.nsteps = nsteps
        self.type = type
        self.mode = mode
        self.time = np.linspace(0, tf, nsteps)
        self.number_parameters = number_of_parameters
        self.seed = seed

        dim = number_of_parameters

        # ── parameter vector sizing (mirrors schedule_utils.py) ───────────────
        if self.type in ('F-CRAB', 'fourier'):
            n_params = 4 * dim
        else:  # power law
            n_params = 2 * dim

        self.parameters = np.zeros(n_params)
        if random:
            rng = np.random.default_rng(seed)
            self.parameters = rng.uniform(-2, 2, size=n_params)

        # ── frequencies ───────────────────────────────────────────────────────
        if self.type in ('F-CRAB', 'fourier'):
            rng = np.random.default_rng(seed)
            self.omegas = (
                2 * np.pi * np.arange(1, dim + 1)
                * (1 + rng.uniform(-0.5, 0.5, dim))
                / self.tf
            )

        # ── precompute jax constants (time, basis) ────────────────────────────
        self._time_jax   = jnp.array(self.time)
        if self.type in ('F-CRAB', 'fourier'):
            _omegas          = jnp.array(self.omegas)
            self._sin_basis  = jnp.sin(jnp.outer(_omegas, self._time_jax))  # (dim, nsteps)
            self._cos_basis  = jnp.cos(jnp.outer(_omegas, self._time_jax))
        elif self.type == 'power law':
            exponents        = jnp.arange(1, dim + 1)[:, None]
            self._pw_basis   = (self._time_jax[None, :] / tf) ** exponents   # (dim, nsteps)

    # ─────────────────────────────────────────────────────────────────────────
    def get_driving(self) -> tuple:
        """Returns (h_driver, h_target) as numpy arrays — drop-in replacement."""
        h_driver_jax, h_target_jax = self._get_driving_jax(
            jnp.array(self.parameters)
        )
        return np.array(h_driver_jax), np.array(h_target_jax)

    def _get_driving_jax(self, parameters: jnp.ndarray) -> tuple:
        """Internal JAX version — differentiable."""
        dim = self.number_parameters
        t   = self._time_jax
        tf  = self.tf

        if self.type in ('F-CRAB', 'fourier'):
            a_drv = parameters[        : dim  ]
            b_drv = parameters[  dim   : 2*dim]
            a_tgt = parameters[2*dim   : 3*dim]
            b_tgt = parameters[3*dim   : 4*dim]
            corr_driver = jnp.mean(
                a_drv[:, None] * self._sin_basis + b_drv[:, None] * self._cos_basis,
                axis=0,
            )
            corr_target = jnp.mean(
                a_tgt[:, None] * self._sin_basis + b_tgt[:, None] * self._cos_basis,
                axis=0,
            )
        else:  # power law
            corr_driver = jnp.mean(
                parameters[:dim, None] * self._pw_basis, axis=0
            )
            corr_target = jnp.mean(
                parameters[dim:2*dim, None] * self._pw_basis, axis=0
            )

        h_driver = (1 - t / tf) * (1 + corr_driver)
        h_target = (t / tf)     * (1 + corr_target)
        return h_driver, h_target

    # ─────────────────────────────────────────────────────────────────────────
    def load(self, parameters: np.ndarray):
        if parameters.shape[0] == self.parameters.shape[0]:
            self.parameters = parameters.copy()
        else:
            raise ValueError(
                f'Shape mismatch: got {parameters.shape[0]}, '
                f'expected {self.parameters.shape[0]}'
            )


# ─────────────────────────────────────────────────────────────────────────────
class JaxSchedulerModel(JaxSchedule):
    """
    JAX equivalent of SchedulerModel.
    - Uses jax.scipy.linalg.expm + lax.scan for the time evolution
    - Provides exact gradients via jax.grad — no finite differences
    - All public outputs (psi, energy, history, get_driving) are numpy arrays
      so the rest of your pipeline works without changes
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
        mode: Optional[str] = 'annealing ansatz',
        random: Optional[bool] = False,
    ):
        self.initial_state        = initial_state
        self.target_hamiltonian   = target_hamiltonian
        self.initial_hamiltonian  = initial_hamiltonian
        self.reference_hamiltonian = reference_hamiltonian

        super().__init__(
            tf=tf,
            type=type,
            number_of_parameters=number_of_parameters,
            nsteps=nsteps,
            seed=seed,
            mode=mode,
            random=random,
        )

        # ── convert sparse hamiltonians to dense jax arrays ───────────────────
        self._H_driver = jnp.array(initial_hamiltonian.toarray(),  dtype=jnp.complex128)
        self._H_target = jnp.array(target_hamiltonian.toarray(),   dtype=jnp.complex128)
        self._H_ref    = jnp.array(reference_hamiltonian.toarray(), dtype=jnp.complex128)
        self._psi_init = jnp.array(initial_state,                  dtype=jnp.complex128)
        self._dt       = jnp.float64(self.time[1] - self.time[0])

        # ── compile forward + gradient ────────────────────────────────────────
        self._forward_jax = jax.jit(self._build_forward())
        self._grad_jax    = jax.jit(jax.grad(self._build_forward()))

        # warm up JIT
        _p = jnp.array(self.parameters)
        self._forward_jax(_p).block_until_ready()
        self._grad_jax(_p).block_until_ready()
        print('JIT compilation done.')

        # ── state ─────────────────────────────────────────────────────────────
        self.energy = 1000.0
        self.psi    = None

        # memory — all stored as numpy for compatibility
        self.history            = []
        self.history_psi        = []
        self.history_drivings   = []
        self.history_parameters = []
        self.history_run        = []
        self.run_number         = 0

    # ─────────────────────────────────────────────────────────────────────────
    def _build_forward(self):
        """Closure that captures jax arrays — returns a pure jax function."""
        H_driver  = self._H_driver
        H_target  = self._H_target
        H_ref     = self._H_ref
        psi_init  = self._psi_init
        dt        = self._dt
        nsteps    = self.nsteps

        # capture schedule internals
        get_driving = self._get_driving_jax

        def forward(parameters):
            h_driver, h_target = get_driving(parameters)

            def step(psi, i):
                H_t = h_driver[i] * H_driver + h_target[i] * H_target
                psi = expm(-1j * dt * H_t) @ psi
                return psi, None

            psi_final, _ = jax.lax.scan(step, psi_init, jnp.arange(nsteps))
            energy = (psi_final.conj() @ H_ref @ psi_final).real
            return energy

        return forward

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, parameters: np.ndarray) -> float:
        """
        Drop-in replacement for SchedulerModel.forward.
        Accepts and returns numpy/python scalars — scipy compatible.
        """
        self.parameters = parameters
        p = jnp.array(parameters)

        energy = float(self._forward_jax(p))
        self.energy = energy

        # also update psi by re-running the scan and capturing final state
        self.psi = self._get_final_psi(p)
        self.run_number += 1
        return energy

    def _get_final_psi(self, parameters: jnp.ndarray) -> np.ndarray:
        """Re-run evolution and return final state as numpy array."""
        h_driver, h_target = self._get_driving_jax(parameters)
        H_driver, H_target = self._H_driver, self._H_target
        dt = self._dt

        def step(psi, i):
            H_t = h_driver[i] * H_driver + h_target[i] * H_target
            psi = expm(-1j * dt * H_t) @ psi
            return psi, None

        psi_final, _ = jax.lax.scan(step, self._psi_init, jnp.arange(self.nsteps))
        return np.array(psi_final)

    # ─────────────────────────────────────────────────────────────────────────
    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Exact gradient via jax.grad — pass as jac= to scipy.optimize.minimize."""
        return np.array(self._grad_jax(jnp.array(parameters)), dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    def callback(self, *args):
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters.copy())
        self.history_drivings.append(self.get_driving())       # numpy
        self.history_psi.append(self.psi.copy())
        self.history_run.append(self.run_number)
        print(self.energy)


# ─────────────────────────────────────────────────────────────────────────────
class JaxTrainer:
    """
    Handles optimization of a JaxSchedulerModel.
    Call trainer.run() and get results back as plain numpy/python objects.
    """

    def __init__(
        self,
        model: JaxSchedulerModel,
        maxiter: int = 1000,
        tol: float = 1e-6,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
        verbose: bool = True,
    ):
        self.model   = model
        self.maxiter = maxiter
        self.tol     = tol
        self.ftol    = ftol
        self.gtol    = gtol
        self.verbose = verbose
        self.result  = None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> dict:
        """
        Run L-BFGS-B with exact gradients.
        Returns a dict with all results as numpy arrays — no jax objects.
        """
        res = minimize(
            self.model.forward,
            self.model.parameters,
            jac=self.model.gradient,
            method='L-BFGS-B',
            tol=self.tol,
            callback=self.model.callback if self.verbose else None,
            options={
                'maxiter': self.maxiter,
                'ftol'   : self.ftol,
                'gtol'   : self.gtol,
            },
        )
        self.result = res

        # final forward pass to sync model state
        self.model.forward(res.x)
        h_driver, h_target = self.model.get_driving()

        if self.verbose:
            print(f'\nOptimization success : {res.success}')
            print(f'Final energy         : {res.fun:.6f}')
            print(f'Message              : {res.message}')

        return {
            # optimization results
            'success'      : bool(res.success),
            'message'      : res.message,
            'n_iterations' : int(res.nit),
            'n_evals'      : int(res.nfev),
            # physics results — all numpy
            'energy'       : float(res.fun),
            'parameters'   : np.array(res.x),
            'psi'          : self.model.psi.copy(),          # numpy complex128
            'h_driver'     : h_driver,                       # numpy float64
            'h_target'     : h_target,
            'time'         : self.model.time.copy(),
            # full history
            'history_energy'     : list(self.model.history),
            'history_parameters' : [p.copy() for p in self.model.history_parameters],
            'history_drivings'   : self.model.history_drivings,
            'history_psi'        : [p.copy() for p in self.model.history_psi],
        }