# src/sparse_grape.py
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

Schedule structure (same as JaxSchedule):
    h_driver(t_i) = ramp_drv(t_i) * (1 + correction_drv(t_i))
    h_target(t_i) = ramp_tgt(t_i) * (1 + correction_tgt(t_i))

    where ramp_drv = (1 - t/tf),  ramp_tgt = t/tf

GRAPE formula:
    dE/dh_drv_i = -2 dt Im[ <χ_i | H_driver | ψ_i> ]
    dE/dh_tgt_i = -2 dt Im[ <χ_i | H_target | ψ_i> ]

    grad[k] = Σ_i dE/dh_drv_i * dh_drv_i/dθ_k
            + Σ_i dE/dh_tgt_i * dh_tgt_i/dθ_k

    where |χ_i> is the co-state propagated backward from |χ_T> = H_ref |ψ_T>.

Usage in study_1d_ising.py:
    from src.sparse_grape import SparseGRAPEModel, SparseGRAPETrainer

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
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
from typing import Optional


# ── softplus and its derivative ───────────────────────────────────────────────
def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), numerically stable."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Derivative of softplus = sigmoid(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


_SOFTPLUS_1 = _softplus(np.ones(1))[0]  # softplus(1) ≈ 1.3133


# ─────────────────────────────────────────────────────────────────────────────
class SparseGRAPEModel:
    """
    Sparse GRAPE optimal control model.
    Drop-in replacement for JaxSchedulerModel — identical public interface.
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
        self.tf = tf
        self.nsteps = nsteps
        self.type = type
        self.number_parameters = number_of_parameters
        self.time = np.linspace(0, tf, nsteps)
        self.dt = self.time[1] - self.time[0]

        dim = number_of_parameters
        t = self.time

        # ── parameter sizing ──────────────────────────────────────────────────
        if type in ("fourier", "F-CRAB"):
            # sin only: [a_drv(dim), a_tgt(dim)]
            n_params = 2 * dim
        elif type in ("positive fourier", "squared fourier"):
            # sin + cos: [a_drv(dim), b_drv(dim), a_tgt(dim), b_tgt(dim)]
            n_params = 4 * dim
        elif type == "power law":
            # [c_drv(dim), c_tgt(dim)]
            n_params = 2 * dim
        else:
            raise ValueError(
                f"Unknown schedule type '{type}'. "
                f"Choose from: fourier, F-CRAB, positive fourier, "
                f"squared fourier, power law."
            )

        self.parameters = np.zeros(n_params)
        if random:
            rng = np.random.default_rng(seed)
            self.parameters = rng.uniform(-0.5, 0.5, size=n_params)

        # ── basis functions ───────────────────────────────────────────────────
        if type in ("fourier", "F-CRAB", "positive fourier", "squared fourier"):
            if type == "F-CRAB":
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

        # ── sparse Hamiltonians (kept sparse throughout) ───────────────────────
        self._H_driver = initial_hamiltonian.astype(complex)
        self._H_target = target_hamiltonian.astype(complex)
        self._H_ref = reference_hamiltonian.astype(complex)
        self._psi_init = initial_state.astype(complex)

        # ── state tracking (mirrors JaxSchedulerModel) ────────────────────────
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

        return h_driver, h_target, dh_drv, dh_tgt

    # ─────────────────────────────────────────────────────────────────────────
    def get_driving(self, parameters=None) -> tuple:
        """Returns (h_driver, h_target) as numpy arrays."""
        if parameters is None:
            parameters = self.parameters
        h_drv, h_tgt, _, _ = self._compute_driving_and_jacobian(parameters)
        return h_drv, h_tgt

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
        """
        h_driver, h_target, dh_drv_dtheta, dh_tgt_dtheta = (
            self._compute_driving_and_jacobian(parameters)
        )
        dt = self.dt

        # ── forward pass ──────────────────────────────────────────────────────
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
        chi = self._H_ref @ psi_final

        dE_dh_drv = np.zeros(self.nsteps)
        dE_dh_tgt = np.zeros(self.nsteps)

        for i in reversed(range(self.nsteps)):
            H_t = h_driver[i] * self._H_driver + h_target[i] * self._H_target

            # propagate co-state one step backward (adjoint = +1j for Hermitian H)
            chi = expm_multiply(+1j * dt * H_t, chi)

            psi_i = psi_fwd[i]

            # GRAPE: dE/dh_x_i = -2 dt Im[ ⟨ψ_i | H_x | χ_i⟩ ]
            dE_dh_drv[i] = -2.0 * dt * (psi_i.conj() @ self._H_driver @ chi).imag
            dE_dh_tgt[i] = -2.0 * dt * (psi_i.conj() @ self._H_target @ chi).imag

        # ── chain rule through schedule ───────────────────────────────────────
        # grad[k] = Σ_i [ dE/dh_drv_i * ∂h_drv_i/∂θ_k
        #               + dE/dh_tgt_i * ∂h_tgt_i/∂θ_k ]
        grad = dh_drv_dtheta @ dE_dh_drv + dh_tgt_dtheta @ dE_dh_tgt  # (n_params,)

        return energy, grad

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface — identical to JaxSchedulerModel
    # ─────────────────────────────────────────────────────────────────────────

    def forward_and_gradient(self, parameters: np.ndarray):
        """Returns (energy, grad) together — pass as jac=True to scipy minimize."""
        self.parameters = parameters
        energy, grad = self._forward_and_grad(parameters, compute_grad=True)
        self.energy = energy
        self.run_number += 1
        return energy, grad

    def forward(self, parameters: np.ndarray) -> float:
        self.parameters = parameters
        energy, _ = self._forward_and_grad(parameters, compute_grad=False)
        self.energy = energy
        self.run_number += 1
        return energy

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        _, grad = self._forward_and_grad(parameters, compute_grad=True)
        return grad

    def callback(self, *args):
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters.copy())
        self.history_drivings.append(self.get_driving())
        if self.psi is not None:
            self.history_psi.append(self.psi.copy())
        self.history_run.append(self.run_number)
        print(self.energy)

    def load(self, parameters: np.ndarray):
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
        self.model = model
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose

    def run(self) -> dict:
        """
        Run L-BFGS-B with GRAPE gradients.
        Returns dict with identical keys to JaxTrainer.run().
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
