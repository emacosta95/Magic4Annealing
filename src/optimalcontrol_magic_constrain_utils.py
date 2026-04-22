import numpy as np
import scipy
from scipy.optimize import minimize
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from functools import partial

jax.config.update("jax_enable_x64", True)


# ── Pauli table builders ──────────────────────────────────────────────────────
def _build_binary_reps(n: int):
    n_paulis = 4**n
    pauli_map = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int8)
    indices = np.arange(n_paulis, dtype=np.int32)
    a_vecs = np.zeros((n_paulis, n), dtype=np.int8)
    b_vecs = np.zeros((n_paulis, n), dtype=np.int8)
    tmp = indices.copy()
    for k in range(n - 1, -1, -1):
        digit = tmp % 4
        a_vecs[:, k] = pauli_map[digit, 0]
        b_vecs[:, k] = pauli_map[digit, 1]
        tmp = tmp // 4
    return a_vecs, b_vecs


def _build_xor_table(n: int):
    a_vecs, b_vecs = _build_binary_reps(n)
    dim = 2**n
    powers = (2 ** np.arange(n - 1, -1, -1)).astype(np.int32)
    a_int = (a_vecs @ powers).astype(np.int32)
    x_vals = np.arange(dim, dtype=np.int32)
    bit_mat = ((x_vals[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(
        np.int8
    )
    b_dot_x = (b_vecs @ bit_mat.T) % 2
    signs = 1 - 2 * b_dot_x  # (4^n, 2^n)
    return a_int, signs


# ─────────────────────────────────────────────────────────────────────────────
class JaxSchedule:

    def __init__(
        self,
        tf,
        type,
        number_of_parameters,
        nsteps,
        seed=None,
        mode="annealing ansatz",
        random=False,
    ):
        self.tf = tf
        self.nsteps = nsteps
        self.type = type
        self.mode = mode
        self.time = np.linspace(0, tf, nsteps)
        self.number_parameters = number_of_parameters
        self.seed = seed
        dim = number_of_parameters

        if self.type in ("F-CRAB", "fourier"):
            n_params = 4 * dim
        else:
            n_params = 2 * dim

        self.parameters = np.zeros(n_params)
        if random:
            rng = np.random.default_rng(seed)
            self.parameters = rng.uniform(-2, 2, size=n_params)

        if self.type in ("F-CRAB", "fourier"):
            rng = np.random.default_rng(seed)
            self.omegas = (
                2
                * np.pi
                * np.arange(1, dim + 1)
                * (1 + rng.uniform(-0.5, 0.5, dim))
                / self.tf
            )

        self._time_jax = jnp.array(self.time)
        if self.type in ("F-CRAB", "fourier"):
            _omegas = jnp.array(self.omegas)
            self._sin_basis = jnp.sin(jnp.outer(_omegas, self._time_jax))
            self._cos_basis = jnp.cos(jnp.outer(_omegas, self._time_jax))
        elif self.type == "power law":
            exponents = jnp.arange(1, dim + 1)[:, None]
            self._pw_basis = (self._time_jax[None, :] / tf) ** exponents

    def get_driving(self):
        h_driver_jax, h_target_jax = self._get_driving_jax(jnp.array(self.parameters))
        return np.array(h_driver_jax), np.array(h_target_jax)

    def _get_driving_jax(self, parameters):
        dim = self.number_parameters
        t = self._time_jax
        tf = self.tf
        if self.type in ("F-CRAB", "fourier"):
            a_drv = parameters[:dim]
            b_drv = parameters[dim : 2 * dim]
            a_tgt = parameters[2 * dim : 3 * dim]
            b_tgt = parameters[3 * dim : 4 * dim]
            corr_driver = jnp.mean(
                a_drv[:, None] * self._sin_basis + b_drv[:, None] * self._cos_basis,
                axis=0,
            )
            corr_target = jnp.mean(
                a_tgt[:, None] * self._sin_basis + b_tgt[:, None] * self._cos_basis,
                axis=0,
            )
        else:
            corr_driver = jnp.mean(parameters[:dim, None] * self._pw_basis, axis=0)
            corr_target = jnp.mean(
                parameters[dim : 2 * dim, None] * self._pw_basis, axis=0
            )
        h_driver = (1 - t / tf) * (1 + corr_driver)
        h_target = (t / tf) * (1 + corr_target)
        return h_driver, h_target

    def load(self, parameters):
        if parameters.shape[0] == self.parameters.shape[0]:
            self.parameters = parameters.copy()
        else:
            raise ValueError(
                f"Shape mismatch: got {parameters.shape[0]}, "
                f"expected {self.parameters.shape[0]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class JaxSchedulerModelMagic(JaxSchedule):
    """
    Optimal control model with integrated magic penalty in the loss:

        L(θ) = E(θ) + λ * (1/T) Σ_t M_2(ψ(t))

    M_2 is computed at every time step inside lax.scan so gradients flow
    through the full trajectory. Set lambda_magic=0.0 to recover the
    standard energy-only loss with no overhead.
    """

    def __init__(
        self,
        initial_state,
        target_hamiltonian,
        initial_hamiltonian,
        reference_hamiltonian,
        tf,
        number_of_parameters,
        nsteps,
        type,
        seed,
        lambda_magic: float = 0.0,
        magic_batch_size: int = 4096,
        mode="annealing ansatz",
        random=False,
    ):
        self.initial_state = initial_state
        self.target_hamiltonian = target_hamiltonian
        self.initial_hamiltonian = initial_hamiltonian
        self.reference_hamiltonian = reference_hamiltonian
        self.lambda_magic = lambda_magic

        super().__init__(
            tf=tf,
            type=type,
            number_of_parameters=number_of_parameters,
            nsteps=nsteps,
            seed=seed,
            mode=mode,
            random=random,
        )

        self._H_driver = jnp.array(initial_hamiltonian.toarray(), dtype=jnp.complex128)
        self._H_target = jnp.array(target_hamiltonian.toarray(), dtype=jnp.complex128)
        self._H_ref = jnp.array(reference_hamiltonian.toarray(), dtype=jnp.complex128)
        self._psi_init = jnp.array(initial_state, dtype=jnp.complex128)
        self._dt = jnp.float64(self.time[1] - self.time[0])

        # ── qubit counts ──────────────────────────────────────────────────────
        # Hamiltonians live in the sector: dim_sector = 2^(n-1) = 32
        # SRE must be computed in the full space: 2^n = 64, n_qubits = 6
        dim_sector = initial_hamiltonian.shape[0]  # 32
        n_qubits = int(np.log2(dim_sector)) + 1  # 6
        self.n_qubits = n_qubits
        self._full_dim = 2**n_qubits  # 64
        self._sector_idx = jnp.arange(dim_sector, dtype=jnp.int32)  # [0..31]

        # ── Pauli tables for SRE (built for full 2^n space) ───────────────────
        n_paulis = 4**n_qubits  # 4096
        magic_batch_size = min(magic_batch_size, n_paulis)  # cap to n_paulis

        print(
            f"Building Pauli tables for n={n_qubits} ({n_paulis} Paulis, "
            f"sector dim={dim_sector}, batch={magic_batch_size})..."
        )
        a_int_np, signs_np = _build_xor_table(n_qubits)  # full n=6

        n_batches = (n_paulis + magic_batch_size - 1) // magic_batch_size
        pad = n_batches * magic_batch_size - n_paulis
        if pad > 0:
            a_int_np = np.concatenate([a_int_np, np.zeros(pad, dtype=np.int32)])
            signs_np = np.concatenate(
                [signs_np, np.zeros((pad, self._full_dim), dtype=np.float64)]
            )

        self._a_int = jnp.array(a_int_np, dtype=jnp.int32)
        self._signs = jnp.array(signs_np, dtype=jnp.float64)
        self._x_idx = jnp.arange(self._full_dim, dtype=jnp.int32)  # [0..63]
        self._n_paulis = n_paulis
        self._n_batches = n_batches
        self._magic_batch = magic_batch_size
        print("Done.")

        # ── compile forward + gradient ────────────────────────────────────────
        self._forward_jax = jax.jit(self._build_forward())
        self._grad_jax = jax.jit(jax.grad(self._build_forward()))

        _p = jnp.array(self.parameters)
        self._forward_jax(_p).block_until_ready()
        self._grad_jax(_p).block_until_ready()
        print("JIT compilation done.")

        # ── state ─────────────────────────────────────────────────────────────
        self.energy = 1000.0
        self.magic = 0.0
        self.psi = None

        self.history = []
        self.history_magic = []
        self.history_psi = []
        self.history_drivings = []
        self.history_parameters = []
        self.history_run = []
        self.run_number = 0

    # ── SRE: pure JAX, differentiable ────────────────────────────────────────
    def _sre_jax(self, psi: jnp.ndarray) -> jnp.ndarray:
        psi_full = jnp.zeros(self._full_dim, dtype=jnp.complex128)
        psi_full = psi_full.at[self._sector_idx].set(psi)

        a_int = self._a_int
        signs = self._signs
        x_idx = self._x_idx
        B = self._magic_batch
        n_batches = self._n_batches

        def batch_sum_xi4(carry, batch_idx):
            start = jnp.int32(batch_idx * B)
            a_batch = jax.lax.dynamic_slice(a_int, (start,), (B,))
            s_batch = jax.lax.dynamic_slice(
                signs, (start, jnp.int32(0)), (B, signs.shape[1])
            )
            x_xor_a = jnp.bitwise_xor(x_idx[None, :], a_batch[:, None])
            psi_flip = psi_full[x_xor_a]
            xi = jnp.einsum("x,px,px->p", psi_full.conj(), s_batch, psi_flip).real
            return carry + jnp.sum(xi**4), None

        sum_xi4, _ = jax.lax.scan(
            batch_sum_xi4,
            jnp.float64(0.0),
            jnp.arange(n_batches, dtype=jnp.int32),
        )
        return -jnp.log(sum_xi4) + self.n_qubits * jnp.log(2.0)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_forward(self):
        """
        Loss = E(ψ_final) + λ * mean_t[ M_2(ψ(t)) ]

        The time scan accumulates M_2 at every step as a scan output,
        so gradients flow through the full trajectory automatically.
        """
        H_driver = self._H_driver
        H_target = self._H_target
        H_ref = self._H_ref
        psi_init = self._psi_init
        dt = self._dt
        nsteps = self.nsteps
        lambda_m = self.lambda_magic
        get_driving = self._get_driving_jax
        sre = self._sre_jax

        def forward(parameters):
            h_driver, h_target = get_driving(parameters)

            def step(psi, i):
                H_t = h_driver[i] * H_driver + h_target[i] * H_target
                psi_new = expm(-1j * dt * H_t) @ psi
                psi_n = psi_new / jnp.linalg.norm(psi_new)
                m2_i = sre(psi_n) if lambda_m != 0.0 else jnp.float64(0.0)
                return psi_new, m2_i

            psi_final, m2_trajectory = jax.lax.scan(step, psi_init, jnp.arange(nsteps))

            psi_n = psi_final / jnp.linalg.norm(psi_final)
            energy = (psi_n.conj() @ H_ref @ psi_n).real
            magic = jnp.mean(m2_trajectory) if lambda_m != 0.0 else jnp.float64(0.0)
            return energy + lambda_m * magic

        return forward

    # ─────────────────────────────────────────────────────────────────────────
    def _get_energy_magic_psi(self, parameters: jnp.ndarray):
        """
        Re-run evolution to extract energy, mean M_2, full M_2 trajectory,
        and final state separately — for logging, not on the gradient path.
        """
        h_driver, h_target = self._get_driving_jax(parameters)

        def step(psi, i):
            H_t = h_driver[i] * self._H_driver + h_target[i] * self._H_target
            psi_new = expm(-1j * self._dt * H_t) @ psi
            psi_n = psi_new / jnp.linalg.norm(psi_new)
            m2_i = self._sre_jax(psi_n)
            return psi_new, m2_i

        psi_final, m2_trajectory = jax.lax.scan(
            step, self._psi_init, jnp.arange(self.nsteps)
        )
        psi_n = psi_final / jnp.linalg.norm(psi_final)
        energy = float((psi_n.conj() @ self._H_ref @ psi_n).real)
        magic = float(jnp.mean(m2_trajectory))
        return energy, magic, np.array(m2_trajectory), np.array(psi_n)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, parameters: np.ndarray) -> float:
        self.parameters = parameters
        p = jnp.array(parameters)
        loss = float(self._forward_jax(p))

        energy, magic, m2_traj, psi = self._get_energy_magic_psi(p)
        self.energy = energy
        self.magic = magic
        self.m2_traj = m2_traj  # (nsteps,) — available for inspection
        self.psi = psi
        self.run_number += 1
        return loss

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return np.array(self._grad_jax(jnp.array(parameters)), dtype=np.float64)

    def callback(self, *args):
        self.history.append(self.energy)
        self.history_magic.append(self.magic)
        self.history_parameters.append(self.parameters.copy())
        self.history_drivings.append(self.get_driving())
        self.history_psi.append(self.psi.copy())
        self.history_run.append(self.run_number)
        loss = self.energy + self.lambda_magic * self.magic
        print(f"E={self.energy:.6f}  " f"<M2>={self.magic:.4f}  " f"loss={loss:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
class JaxTrainerMagic:
    """Trainer for JaxSchedulerModelMagic."""

    def __init__(
        self,
        model: JaxSchedulerModelMagic,
        maxiter: int = 1000,
        tol: float = 1e-6,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
        verbose: bool = True,
    ):
        self.model = model
        self.maxiter = maxiter
        self.tol = tol
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose

    def run(self) -> dict:
        res = minimize(
            self.model.forward,
            self.model.parameters,
            jac=self.model.gradient,
            method="L-BFGS-B",
            tol=self.tol,
            callback=self.model.callback if self.verbose else None,
            options={
                "maxiter": self.maxiter,
                "ftol": self.ftol,
                "gtol": self.gtol,
            },
        )

        self.model.forward(res.x)
        h_driver, h_target = self.model.get_driving()

        if self.verbose:
            print(f"\nOptimization success : {res.success}")
            print(f"Final energy         : {self.model.energy:.6f}")
            print(f"Final <M2>           : {self.model.magic:.4f}")
            print(f"lambda_magic         : {self.model.lambda_magic}")
            print(f"Message              : {res.message}")

        return {
            "success": bool(res.success),
            "message": res.message,
            "n_iterations": int(res.nit),
            "n_evals": int(res.nfev),
            "energy": float(self.model.energy),
            "magic": float(self.model.magic),  # mean M_2
            "m2_trajectory": self.model.m2_traj.copy(),  # (nsteps,)
            "lambda_magic": self.model.lambda_magic,
            "parameters": np.array(res.x),
            "psi": self.model.psi.copy(),
            "h_driver": h_driver,
            "h_target": h_target,
            "time": self.model.time.copy(),
            "history_energy": list(self.model.history),
            "history_magic": list(self.model.history_magic),
            "history_parameters": [p.copy() for p in self.model.history_parameters],
            "history_drivings": self.model.history_drivings,
            "history_psi": [p.copy() for p in self.model.history_psi],
        }
