"""
NambuGRAPEModel: drop-in replacement for SparseGRAPEModel on the 1d Ising
chain / frustrated ring, using the free-fermion (BdG) backend.

Reused UNCHANGED from SparseGRAPEModel: every schedule ansatz (fourier,
F-CRAB, positive/squared fourier, power law, LZS), its analytic Jacobian
(_compute_driving_and_jacobian), callback/history, load(), and
SparseGRAPETrainer.  Only _forward_and_grad is swapped:

    state   : W1 [2l, l]  instead of psi [2^L]
    energy  : tr(W1^dag H_ref W1)               (verified vs ED)
    gradient: NambuIsing1D.grape_energy_and_grad  (exact Daleckii-Krein
              derivative of each step, not first order in dt)

Cost O(nsteps * l^3) vs O(nsteps * 2^L).

Usage:
    from src.free_fermions_utils import NambuIsing1D
    from src.nambu_grape import NambuGRAPEModel
    from src.sparse_grape_method import SparseGRAPETrainer

    nambu = NambuIsing1D.frustrated_ring(N)
    model = NambuGRAPEModel(nambu, tf=tau, number_of_parameters=M,
                            nsteps=nsteps, type="LZS")
    out = SparseGRAPETrainer(model, maxiter=200).run()
    # out["psi"] is W1 [2l, l]: feed it to nambu.majorana_covariance,
    # nambu.residual_energy, nambu.level_probabilities, ...
"""

import numpy as np
import scipy.sparse as sp

from src.free_fermions_utils import NambuIsing1D
from src.sparse_grape_method import SparseGRAPEModel


class NambuGRAPEModel(SparseGRAPEModel):
    def __init__(
        self,
        nambu: NambuIsing1D,
        tf: float,
        number_of_parameters: int,
        nsteps: int,
        type: str = "fourier",
        seed: int = 42,
        mode=None,
        random: bool = False,
        bounds_opt: bool = False,
        h_ref=(0.0, 1.0),
    ):
        """
        nambu : NambuIsing1D (couplings, pbc/parity already fixed).
        h_ref : (a, b) -> H_ref = a*H_driver + b*H_target  (default: H_target).
        Other args: exactly as SparseGRAPEModel.
        """
        # parent builds schedule / basis / Jacobian machinery; its sparse
        # Hamiltonians are not used here -> 1x1 placeholders
        one = sp.identity(1, format="csr")
        super().__init__(
            initial_state=np.ones(1),
            target_hamiltonian=one,
            initial_hamiltonian=one,
            reference_hamiltonian=one,
            tf=tf,
            number_of_parameters=number_of_parameters,
            nsteps=nsteps,
            type=type,
            seed=seed,
            mode=mode,
            random=random,
            bounds_opt=bounds_opt,
        )
        self.nambu = nambu
        self.h_ref = tuple(h_ref)
        # fixed initial state = driver ground state (parameter-independent,
        # as psi_init in the sparse version)
        _, self._w_init = nambu.diagonalize(1.0, 0.0)

    def _forward_and_grad(self, parameters: np.ndarray, compute_grad: bool = True):
        h_driver, h_target, dh_drv_dtheta, dh_tgt_dtheta = (
            self._compute_driving_and_jacobian(parameters)
        )
        if not compute_grad:
            w, _ = self.nambu.evolve(h_driver, h_target, self.dt, w0=self._w_init)
            w1 = w[:, : self.nambu.l]
            hr = self.nambu.hamiltonian(*self.h_ref)
            self.psi = w1.copy()
            return float(np.real(np.trace(w1.conj().T @ hr @ w1))), None

        energy, dE_dh_drv, dE_dh_tgt, w1 = self.nambu.grape_energy_and_grad(
            h_driver,
            h_target,
            self.dt,
            h_ref=self.h_ref,
            w0=self._w_init,
            return_state=True,
        )
        self.psi = w1.copy()
        grad = dh_drv_dtheta @ dE_dh_drv + dh_tgt_dtheta @ dE_dh_tgt
        return energy, grad
