import numpy as np
import scipy
from scipy.optimize import minimize
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from functools import partial

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





jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# Core idea:
# Every n-qubit Pauli P = i^k * X^a Z^b  where a,b in {0,1}^n
# <psi|P|psi> can be computed without building the full 2^n x 2^n matrix.
# For a state psi in the computational basis:
#
#   <psi|X^a Z^b|psi> = sum_{x} psi*(x) * (-1)^{x.b} * psi(x XOR a)
#
# where x is a computational basis index, x.b is the bitwise dot product,
# and x XOR a is the bit-flip by a.
# This is O(2^n) per Pauli — but we can vectorize over all 4^n Paulis at once.
# ─────────────────────────────────────────────────────────────────────────────

def _build_binary_reps(n: int):
    """
    Build all 4^n Pauli labels as (a, b) pairs where a,b in {0,1}^n.
    Returns:
        a_vecs : (4^n, n) int8 array — X part of each Pauli
        b_vecs : (4^n, n) int8 array — Z part of each Pauli
    """
    n_paulis = 4 ** n
    # enumerate all (a, b) pairs in base-4: digit k -> (a_k, b_k)
    # 0=I(00), 1=X(10), 2=Y(11), 3=Z(01)
    pauli_map = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.int8)  # (4, 2)

    indices = np.arange(n_paulis, dtype=np.int32)
    a_vecs  = np.zeros((n_paulis, n), dtype=np.int8)
    b_vecs  = np.zeros((n_paulis, n), dtype=np.int8)

    tmp = indices.copy()
    for k in range(n - 1, -1, -1):
        digit         = tmp % 4
        a_vecs[:, k]  = pauli_map[digit, 0]
        b_vecs[:, k]  = pauli_map[digit, 1]
        tmp           = tmp // 4

    return a_vecs, b_vecs


def _build_xor_table(n: int):
    """
    Precompute XOR table: for each basis index x (0..2^n-1) and each Pauli a,
    we need x XOR a_int where a_int = sum_k a_k * 2^(n-1-k).
    Returns a_int : (4^n,) int32 array — integer representation of X parts.
    """
    a_vecs, b_vecs = _build_binary_reps(n)
    n_paulis = 4 ** n
    dim      = 2 ** n

    # integer representation of a (X part) — for XOR
    powers   = (2 ** np.arange(n - 1, -1, -1)).astype(np.int32)
    a_int    = (a_vecs @ powers).astype(np.int32)   # (4^n,)

    # integer dot product b.x for all x and all Paulis
    # b_dot_x[p, x] = sum_k b_vecs[p,k] * bit_k(x)   mod 2
    # We'll compute this as a (4^n, 2^n) binary matrix
    x_vals   = np.arange(dim, dtype=np.int32)        # (2^n,)
    # bit_matrix[x, k] = k-th bit of x
    bit_matrix = ((x_vals[:, None] >> np.arange(n-1, -1, -1)[None, :]) & 1).astype(np.int8)
    # b_dot_x[p, x] = (b_vecs[p] . bit_matrix[x]) mod 2
    b_dot_x  = (b_vecs @ bit_matrix.T) % 2           # (4^n, 2^n)
    signs    = 1 - 2 * b_dot_x                        # (4^n, 2^n): +1 or -1

    return a_int, signs


class SREJax:
    """
    Stabilizer Rényi Entropy at n=2 using JAX.

    M_2(psi) = -log(sum_P <psi|P|psi>^4) - n*log(2)

    Algorithm:
        For each Pauli P = X^a Z^b:
            <psi|P|psi> = sum_x psi*(x) * (-1)^{x.b} * psi(x XOR a)
                        = sum_x conj(psi[x]) * sign[x] * psi[x XOR a]

    This vectorizes over all 4^n Paulis simultaneously using precomputed
    XOR indices and sign tables — no matrix exponentiation, no dense Pauli
    tensor, memory scales as O(4^n + 2^n) not O(4^n * 2^n * 2^n).

    Parameters
    ----------
    n_qubits : int
    batch_size : int
        Number of Paulis to process per JAX call. Tune to fit GPU/CPU memory.
        Default 4096 works well for n=10 on CPU.
    """

    def __init__(self, n_qubits: int, batch_size: int = 4096):
        self.n          = n_qubits
        self.dim        = 2 ** n_qubits
        self.n_paulis   = 4 ** n_qubits
        self.batch_size = batch_size

        print(f'Building Pauli tables for n={n_qubits} ({self.n_paulis} Paulis)...')
        a_int, signs = _build_xor_table(n_qubits)

        # store as jax arrays
        self._a_int  = jnp.array(a_int,  dtype=jnp.int32)    # (4^n,)
        self._signs  = jnp.array(signs,  dtype=jnp.float64)   # (4^n, 2^n)
        self._x_idx  = jnp.arange(self.dim, dtype=jnp.int32)  # (2^n,)
        print('Done.')

    # ─────────────────────────────────────────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def _xi_batch(self, psi: jnp.ndarray, a_int_batch: jnp.ndarray,
                  signs_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Compute <psi|P|psi> for a batch of Paulis.
        psi          : (2^n,) complex
        a_int_batch  : (batch,) int32
        signs_batch  : (batch, 2^n) float64
        Returns      : (batch,) float64
        """
        # psi[x XOR a] for each Pauli in batch: shape (batch, 2^n)
        x_xor_a = jnp.bitwise_xor(
            self._x_idx[None, :], a_int_batch[:, None]
        )                                               # (batch, 2^n)
        psi_flipped = psi[x_xor_a]                    # (batch, 2^n) complex

        # <psi|P|psi> = sum_x conj(psi[x]) * sign[x] * psi[x XOR a]
        xi = jnp.einsum(
            'x,px,px->p',
            psi.conj(),
            signs_batch,
            psi_flipped,
        ).real                                          # (batch,) float64

        return xi

    # ─────────────────────────────────────────────────────────────────────────
    def characteristic_function(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute Xi(P) = <psi|P|psi> for all 4^n Paulis.
        Returns numpy array of shape (4^n,).
        """
        psi = jnp.array(psi / np.linalg.norm(psi), dtype=jnp.complex128)
        xi  = np.zeros(self.n_paulis, dtype=np.float64)

        for start in range(0, self.n_paulis, self.batch_size):
            end   = min(start + self.batch_size, self.n_paulis)
            xi[start:end] = np.array(self._xi_batch(
                psi,
                self._a_int[start:end],
                self._signs[start:end],
            ))

        return xi

    # ─────────────────────────────────────────────────────────────────────────
    def __call__(self, psi: np.ndarray) -> float:
        """
        M_2(psi) = -log(sum_P Xi(P)^4) - n*log(2)
        """
        xi = self.characteristic_function(psi)
        return float(-np.log(np.sum(xi ** 4)) + self.n * np.log(2))

    # ─────────────────────────────────────────────────────────────────────────
    def along_path(self, psi_history: np.ndarray,
                   verbose: bool = True) -> np.ndarray:
        """
        Compute M_2 along a full annealing trajectory.

        psi_history : (nsteps, 2^n) complex array
        Returns     : (nsteps,) float64 array
        """
        nsteps = psi_history.shape[0]
        m2     = np.zeros(nsteps)

        for i in range(nsteps):
            m2[i] = self(psi_history[i])
            if verbose and i % 10 == 0:
                print(f'  SRE step {i}/{nsteps}: M2={m2[i]:.4f}')

        return m2

    # ─────────────────────────────────────────────────────────────────────────
    def along_path_fast(self, psi_history: np.ndarray,
                        verbose: bool = True) -> np.ndarray:
        """
        Faster version: processes all time steps together per Pauli batch.
        Better cache usage when nsteps is large.

        psi_history : (nsteps, 2^n) complex array
        Returns     : (nsteps,) float64 array
        """
        nsteps = psi_history.shape[0]
        norms  = np.linalg.norm(psi_history, axis=1, keepdims=True)
        psi_n  = jnp.array(psi_history / norms, dtype=jnp.complex128)

        # sum_P xi^4 accumulated over batches
        sum_xi4 = np.zeros(nsteps, dtype=np.float64)

        for start in range(0, self.n_paulis, self.batch_size):
            end          = min(start + self.batch_size, self.n_paulis)
            a_batch      = self._a_int[start:end]     # (batch,)
            signs_batch  = self._signs[start:end]     # (batch, 2^n)

            # compute xi for all time steps and this Pauli batch
            # xi[t, p] = <psi_t|P_p|psi_t>
            xi_batch = np.array(jax.vmap(
                lambda psi: self._xi_batch(psi, a_batch, signs_batch)
            )(psi_n))                                  # (nsteps, batch)

            sum_xi4 += np.sum(xi_batch ** 4, axis=1)  # (nsteps,)

            if verbose and start % (self.batch_size * 10) == 0:
                print(f'  Pauli batch {start}/{self.n_paulis}')

        m2 = -np.log(sum_xi4) + self.n * np.log(2)
        return m2

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