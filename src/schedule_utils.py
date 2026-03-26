import numpy as np
from typing import List, Dict, Callable, Optional
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import scipy.sparse as sp


def configuration(res, energy, grad_energy):
    print('Optimization Success=', res.success)
    print(f'energy={energy:.5f}')
    print(f'average gradient={np.average(grad_energy):.5f} \n')


class Schedule:
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
        if self.type in ('F-CRAB', 'fourier'):
            n_params = 4 * dim
        else:  # power law
            n_params = 2 * dim

        self.parameters = np.zeros(n_params)
        if random:
            self.parameters = np.random.uniform(-2, 2, size=n_params)

        # ── random frequencies (fix: dim was undefined in 'fourier' branch) ──
        if self.type == 'F-CRAB':
            self.omegas = (
                2 * np.pi * np.arange(1, dim + 1)
                * (1 + np.random.uniform(-0.5, 0.5, dim))
                / self.tf
            )
        if self.type == 'fourier':
            self.omegas = (
                2 * np.pi * np.arange(1, dim + 1)
                * (1 + np.random.uniform(-0.5, 0.5, dim))
                / self.tf
            )

    # ─────────────────────────────────────────────────────────────────────────
    def get_driving(self) -> np.ndarray:
        dim = self.number_parameters
        t = self.time
        tf = self.tf

        # ── compute Fourier / power-law correction matrices ──────────────────
        if self.type == 'power law':
            exponents = np.arange(1, dim + 1)[:, None]        # (dim, 1)
            basis = (t[None, :] / tf) ** exponents             # (dim, nsteps)
            matrix_driver = self.parameters[:dim, None] * basis
            matrix_target = self.parameters[dim:2*dim, None] * basis

        elif self.type in ('F-CRAB', 'fourier'):
            # parameters = [a_drv(dim), b_drv(dim), a_tgt(dim), b_tgt(dim)]
            sin_basis = np.sin(t[None, :] * self.omegas[:, None])  # (dim, nsteps)
            cos_basis = np.cos(t[None, :] * self.omegas[:, None])
            matrix_driver = (
                self.parameters[:dim, None] * sin_basis
                + self.parameters[dim:2*dim, None] * cos_basis
            )
            matrix_target = (
                self.parameters[2*dim:3*dim, None] * sin_basis
                + self.parameters[3*dim:4*dim, None] * cos_basis
            )
        else:
            raise ValueError(f"Unknown schedule type: '{self.type}'")

        # ── build physical schedules ─────────────────────────────────────────
        # 'annealing ansatz': linear ramp envelope x (1 + Fourier correction)
        # 'free': same formula, no constraint — optimizer is free to deform both
        correction_driver = np.mean(matrix_driver, axis=0)   # (nsteps,)
        correction_target = np.mean(matrix_target, axis=0)

        h_driver = (1 - t / tf) * (1 + correction_driver)
        h_target = (t / tf)     * (1 + correction_target)

        return h_driver, h_target

    # ─────────────────────────────────────────────────────────────────────────
    def load(self, parameters: np.ndarray):
        if parameters.shape[0] == self.parameters.shape[0]:
            self.parameters = parameters
        else:
            print(
                f'Shape mismatch: got {parameters.shape[0]}, '
                f'expected {self.parameters.shape[0]}'
            )
            exit()


# ─────────────────────────────────────────────────────────────────────────────
class SchedulerModel(Schedule):
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
        self.target_hamiltonian = target_hamiltonian
        self.initial_hamiltonian = initial_hamiltonian
        self.reference_hamiltonian = reference_hamiltonian
        self.initial_state = initial_state

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
        psi = self.initial_state.copy()
        dt = self.time[1] - self.time[0]
        self.parameters = parameters

        hamiltonians = [self.initial_hamiltonian, self.target_hamiltonian]
        h_driver, h_target = self.get_driving()          # pre-compute once
        schedules = [h_driver, h_target]

        for i in range(self.nsteps):
            time_hamiltonian = sum(
                schedules[r][i] * hamiltonians[r] for r in range(2)
            )
            psi = expm_multiply(-1j * dt * time_hamiltonian, psi)

        # .real avoids ComplexWarning and satisfies scipy's scalar requirement
        self.energy = (
            psi.conjugate() @ self.reference_hamiltonian.dot(psi)
        ).real
        self.psi = psi
        self.run_number += 1

        return self.energy

    # ─────────────────────────────────────────────────────────────────────────
    def callback(self, *args):
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters.copy())
        self.history_drivings.append(self.get_driving())
        self.history_psi.append(self.psi.copy())
        self.history_run.append(self.run_number)
        print(self.energy)

    # ─────────────────────────────────────────────────────────────────────────
    def depolarization_option(self, noise_option: bool, noise_coupling: float):
        self.depolarization_coupling = noise_coupling
        self.noise_option = noise_option