# src/fidelity_robustness.py
"""
Robustness of the final-state fidelity against local fields: exact (the
evolution is re-run for every value of the fields) and to second order (the
augmented-matrix / Van Loan method). Spin basis, full 2^N space.

Perturbed evolution
-------------------
    H_lam(t) = (1 - s(t)) H_D + s(t) H_P + sum_k lam_k f_k(t) V_k

    V_k : the 3N single-site Pauli operators, ordered BY AXIS,
              k = a N + i,    a = 0, 1, 2 <-> x, y, z,    i = 0..N-1
          so lam = [x_0..x_{N-1}, y_0..y_{N-1}, z_0..z_{N-1}], a (3, N) array
          flattened. Pauli normalization (not spin-1/2), site 0 = most
          significant bit (SpinOperator, as in src.annealing_utils).
    f_k : optional time profile, 1 by default (static field).

Exact fidelity (method="exact", the default)
--------------------------------------------
|psi_lam(T)> from the same midpoint-rule evolution, and its population in a
reference subspace with orthonormal basis {|r_n>}:

    F(lam) = sum_n |<r_n|psi_lam(T)>|^2

    "ground_space"     : the ground space of the UNPERTURBED H_P. It is
                         degenerate (|s> and its global flip |s_bar> for the
                         ZZ models here), so this is the probability of
                         reading out a ground state (the default)
    "ground_symmetric" : its Pi = prod_i sigma^x_i = +1 part, (|s> +
                         |s_bar>)/sqrt(2): the ground state of the Z2 +1
                         sector the unperturbed evolution aims at. Also
                         sensitive to the relative phase of |s> and |s_bar>
    "final_state"      : the unperturbed final state |psi_0(T)>

F(0) is not 1 for the ground-state references: it is the unperturbed
success probability of the schedule.

Second order (method="second_order")
------------------------------------
Only with respect to the UNPERTURBED final state (the expansion about the
ground state would need the second derivative of the state):

    F(lam) = |<psi_0(T)|psi_lam(T)>|^2  ~  1 - sum_ij lam_i lam_j Re G_ij
    G_ij   = <d_i|d_j> - <d_i|psi_0(T)><psi_0(T)|d_j>
    |d_k>  = d|psi_lam(T)> / d lam_k   at lam = 0

G = T^2 Cov(Vbar_i, Vbar_j) is the quantum geometric tensor of the final
state. It is Hermitian; only Re G (real, symmetric, positive semidefinite)
enters the fidelity.

The derivative is taken of the DISCRETE midpoint-rule evolution of
src/time_grid.py, not of the continuous equation, so G is the second
derivative of the fidelity a brute-force re-evolution on the same grid would
give, with no O(dt) error. One forward pass over x = (d_1, .., d_K, psi):

    x_{n+1} = exp(-i dt Haug_n) x_n,      |d_k(0)> = 0,  |psi(0)> = psi_0

    Haug_n = 1_{K+1} (x) H(tbar_n) + sum_k f_k(tbar_n) E_{k,K} (x) V_k

whose upper-right blocks are the Frechet derivatives of the step propagator.

sigma^y_i and sigma^z_i anticommute with Pi = prod_i sigma^x_i and take the
state out of the Z2 +1 sector of the build scripts: everything here lives in
the full space, with H_D, H_P and psi_0 NOT projected.
"""

import numpy as np
import scipy.sparse as sp
from ManyBodyQutip.qutip_class import SpinOperator
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

from src.time_grid import make_time_grids

AXES = ("x", "y", "z")
# named reference subspaces of the exact fidelity (reference_states)
REFERENCES = ("ground_space", "ground_symmetric", "final_state")


def local_pauli_operators(nqubits: int):
    """
    The 3N single-site Pauli operators in the full 2^N space, ordered by axis
    (index a * nqubits + i for axis a on site i).

    Returns (operators, labels): a list of sparse matrices and the matching
    list of strings "x_0", ..., "x_{N-1}", "y_0", ..., "z_{N-1}".
    """
    operators, labels = [], []
    for axis in AXES:
        for i in range(nqubits):
            operators.append(
                SpinOperator([(axis, i)], coupling=[1], size=nqubits, verbose=1)
                .qutip_op.data.as_scipy()
                .tocsr()
            )
            labels.append(f"{axis}_{i}")
    return operators, labels


class FidelityRobustness:
    """
    Fidelity of one annealing evolution under the 3N local fields
    lam_k sigma^a_i: exact (final_states, population, fidelity) and to second
    order through the robustness matrix G (compute, infidelity).

    Parameters
    ----------
    nqubits            : number of qubits N
    driver_hamiltonian : sparse (2^N, 2^N), full space
    target_hamiltonian : sparse (2^N, 2^N), full space
    initial_state      : (2^N,) full-space |psi(0)>
    schedule_ctrl      : (nsteps,) s(t) on the CONTROL grid (cell midpoints);
                         h_driver = 1 - s, h_target = s
    tf                 : total evolution time T
    profiles           : optional (3N, nsteps) time profiles f_k on the control
                         grid. None -> static fields, f_k = 1
    chunk_size         : perturbations propagated together in one pass (the
                         augmented vector has (chunk_size + 1) 2^N entries).
                         None -> all 3N at once. Bounds memory, same result
    verbose            : progress bar and summary

    Attributes (after compute())
    ----------------------------
    G           : (3N, 3N) complex robustness matrix
    cov         : G / tf^2
    dpsi        : (3N, 2^N) derivatives |d_k>, NOT normalized (their norm is
                  the sensitivity)
    psi_final   : (2^N,) unperturbed final state
    lambda_star : 1 / sqrt(max eig Re G), scale of validity of the second
                  order (for |lam| >~ lambda_star / 3 it is off by > ~1 %)

    The exact fidelity does not need compute().
    """

    def __init__(
        self,
        nqubits,
        driver_hamiltonian,
        target_hamiltonian,
        initial_state,
        schedule_ctrl,
        tf,
        profiles=None,
        chunk_size=None,
        verbose=False,
    ):
        self.nqubits = int(nqubits)
        self.dim = 2**self.nqubits
        self.n_perturbations = 3 * self.nqubits

        shape = (self.dim, self.dim)
        if driver_hamiltonian.shape != shape or target_hamiltonian.shape != shape:
            raise ValueError(
                f"Hamiltonians must act on the full space {shape} (not on a Z2 "
                f"sector), got {driver_hamiltonian.shape} and "
                f"{target_hamiltonian.shape}"
            )
        self.driver_hamiltonian = sp.csr_matrix(driver_hamiltonian, dtype=complex)
        self.target_hamiltonian = sp.csr_matrix(target_hamiltonian, dtype=complex)

        self.initial_state = np.array(initial_state, dtype=complex).ravel()
        if self.initial_state.shape != (self.dim,):
            raise ValueError(
                f"initial_state must have {self.dim} components (full space), "
                f"got {self.initial_state.shape}"
            )

        self.tf = float(tf)
        self.schedule_ctrl = np.array(schedule_ctrl, dtype=float).ravel()
        self.nsteps = len(self.schedule_ctrl)
        self.times, self.times_ctrl, self.dt = make_time_grids(self.tf, self.nsteps)

        if profiles is not None:
            profiles = np.asarray(profiles, dtype=float)
            if profiles.shape != (self.n_perturbations, self.nsteps):
                raise ValueError(
                    f"profiles must have shape "
                    f"{(self.n_perturbations, self.nsteps)}, got {profiles.shape}"
                )
        self.profiles = profiles

        self.chunk_size = (
            self.n_perturbations if chunk_size is None else max(int(chunk_size), 1)
        )
        self.verbose = verbose

        self.operators, self.labels = local_pauli_operators(self.nqubits)
        self._references = {}  # reference subspaces of the exact fidelity

        self.G = None
        self.cov = None
        self.dpsi = None
        self.psi_final = None
        self.lambda_star = None

    # ─────────────────────────────────────────────────────────────────────────
    def _forward_pass(self, first: int, last: int):
        """
        One augmented evolution for the perturbations first..last-1.

        Returns (dpsi, psi_final): (last - first, 2^N) derivatives and the
        unperturbed final state.
        """
        n_block, dim = last - first, self.dim
        size = (n_block + 1) * dim

        # generators -i dt (...) built once: only the scalar coefficients
        # change from step to step
        identity = sp.identity(n_block + 1, format="csr", dtype=complex)
        gen_driver = sp.kron(
            identity, -1j * self.dt * self.driver_hamiltonian, format="csr"
        )
        gen_target = sp.kron(
            identity, -1j * self.dt * self.target_hamiltonian, format="csr"
        )
        # V_k in block (k, n_block): the V_k stacked by rows, shifted to the
        # last block of columns, plus dim empty rows for the psi block
        stacked = sp.vstack(self.operators[first:last], format="csr")
        gen_perturbation = sp.csr_matrix(
            (
                -1j * self.dt * stacked.data,
                stacked.indices + n_block * dim,
                np.concatenate([stacked.indptr, np.full(dim, stacked.nnz)]),
            ),
            shape=(size, size),
        )
        if self.profiles is not None:
            static_data = gen_perturbation.data.copy()
            # perturbation index of every stored entry of gen_perturbation
            entry_k = first + np.repeat(
                np.arange(n_block * dim) // dim, np.diff(stacked.indptr)
            )

        x = np.zeros(size, dtype=complex)
        x[n_block * dim :] = self.initial_state

        steps = range(self.nsteps)
        if self.verbose:
            steps = tqdm(steps, desc=f"Robustness, perturbations {first}..{last - 1}")
        for n in steps:
            if self.profiles is not None:
                gen_perturbation.data = static_data * self.profiles[entry_k, n]
            s_n = self.schedule_ctrl[n]
            x = expm_multiply(
                (1 - s_n) * gen_driver + s_n * gen_target + gen_perturbation, x
            )

        x = x.reshape(n_block + 1, dim)
        return x[:n_block], x[n_block]

    # ─────────────────────────────────────────────────────────────────────────
    def compute(self):
        """Runs the augmented evolution and fills G, cov, dpsi, psi_final and
        lambda_star. Returns G."""
        dpsi = np.zeros((self.n_perturbations, self.dim), dtype=complex)
        for first in range(0, self.n_perturbations, self.chunk_size):
            last = min(first + self.chunk_size, self.n_perturbations)
            dpsi[first:last], psi_final = self._forward_pass(first, last)

        overlap = dpsi.conj() @ psi_final  # <d_i|psi_T>
        self.dpsi = dpsi
        self.psi_final = psi_final
        self.G = dpsi.conj() @ dpsi.T - np.outer(overlap, overlap.conj())
        self.cov = self.G / self.tf**2

        max_eig = np.linalg.eigvalsh(self.G.real)[-1]
        self.lambda_star = 1.0 / np.sqrt(max_eig) if max_eig > 0 else np.inf
        if self.verbose:
            print(
                f"FidelityRobustness: N={self.nqubits}, T={self.tf:g}, "
                f"max eig Re G = {max_eig:.6g}, lambda* = {self.lambda_star:.4g}"
            )
        return self.G

    # ─────────────────────────────────────────────────────────────────────────
    def _as_lambda(self, lam) -> np.ndarray:
        """lam as (..., 3N): accepts (3N,), (3, N) and batches of either."""
        lam = np.asarray(lam, dtype=float)
        if lam.shape[-2:] == (3, self.nqubits):
            lam = lam.reshape(lam.shape[:-2] + (self.n_perturbations,))
        if lam.shape[-1:] != (self.n_perturbations,):
            raise ValueError(
                f"lam must have shape (..., {self.n_perturbations}) or "
                f"(..., 3, {self.nqubits}), got {lam.shape}"
            )
        return lam

    def infidelity(self, lam):
        """
        Second-order 1 - F = lam^T Re(G) lam.

        lam : (3N,) or (3, N) coefficients ordered by axis, or a batch
              (L, 3N) / (L, 3, N). Returns a float, or an (L,) array.
        """
        if self.G is None:
            self.compute()
        lam = self._as_lambda(lam)
        out = np.einsum("...i,ij,...j->...", lam, self.G.real, lam)
        return float(out) if out.ndim == 0 else out

    # ── exact fidelity ───────────────────────────────────────────────────────
    def _evolve_batch(self, lam):
        """|psi_lam(T)> for the rows of lam (L, 3N), propagated together as
        the block-diagonal system 1_L (x) H_0 + sum_k diag(lam[:, k]) (x) V_k."""
        n_batch, dim = len(lam), self.dim
        size = n_batch * dim

        # generators -i dt (...) built once, as in _forward_pass
        identity = sp.identity(n_batch, format="csr", dtype=complex)
        gen_driver = sp.kron(
            identity, -1j * self.dt * self.driver_hamiltonian, format="csr"
        )
        gen_target = sp.kron(
            identity, -1j * self.dt * self.target_hamiltonian, format="csr"
        )
        # one term per perturbation that is switched on in some row of lam
        active = np.flatnonzero(np.any(lam != 0, axis=0))
        terms = []
        for k in active:
            weights = sp.diags(lam[:, k]).tocsr()
            weights.eliminate_zeros()
            terms.append(
                sp.kron(weights, -1j * self.dt * self.operators[k], format="csr")
            )
        gen_perturbation = sp.csr_matrix((size, size), dtype=complex)
        if self.profiles is None:
            for term in terms:
                gen_perturbation = gen_perturbation + term

        x = np.tile(self.initial_state, n_batch)

        steps = range(self.nsteps)
        if self.verbose:
            steps = tqdm(steps, desc=f"Exact evolution, {n_batch} values of lambda")
        for n in steps:
            if self.profiles is not None:
                gen_perturbation = sp.csr_matrix((size, size), dtype=complex)
                for k, term in zip(active, terms):
                    gen_perturbation = gen_perturbation + self.profiles[k, n] * term
            s_n = self.schedule_ctrl[n]
            x = expm_multiply(
                (1 - s_n) * gen_driver + s_n * gen_target + gen_perturbation, x
            )

        return x.reshape(n_batch, dim)

    def final_states(self, lam, batch_size=None):
        """
        Exact |psi_lam(T)>: the midpoint-rule evolution re-run with
        H_0(tbar_n) + sum_k lam_k f_k(tbar_n) V_k.

        lam        : (3N,) or (3, N) coefficients ordered by axis, or a batch
                     (L, 3N) / (L, 3, N)
        batch_size : values of lam propagated together (one expm_multiply per
                     step for all of them). None -> as many as keep
                     batch_size * 2^N <= 2^15

        Returns (2^N,), or (L, 2^N) for a batch.
        """
        lam = self._as_lambda(lam)
        batch_shape = lam.shape[:-1]
        lam = lam.reshape(-1, self.n_perturbations)
        if batch_size is None:
            batch_size = max(1, 2**15 // self.dim)

        states = np.zeros((len(lam), self.dim), dtype=complex)
        for first in range(0, len(lam), batch_size):
            last = first + batch_size
            states[first:last] = self._evolve_batch(lam[first:last])
        return states.reshape(batch_shape + (self.dim,))

    def ground_space(self, atol=1e-9):
        """Orthonormal basis (n, 2^N), one state per row, of the ground space
        of the (unperturbed) target Hamiltonian."""
        hamiltonian = self.target_hamiltonian
        diagonal = hamiltonian.diagonal()
        if (hamiltonian - sp.diags(diagonal)).count_nonzero() == 0:
            # classical target: the ground space is spanned by basis states
            energies = diagonal.real
            idx = np.flatnonzero(energies <= energies.min() + atol)
            basis = np.zeros((len(idx), self.dim), dtype=complex)
            basis[np.arange(len(idx)), idx] = 1.0
            return basis
        energies, vectors = np.linalg.eigh(hamiltonian.toarray())
        return vectors[:, energies <= energies[0] + atol].T.astype(complex)

    def reference_states(self, reference="ground_space"):
        """
        Orthonormal states (n, 2^N), one per row, spanning the subspace the
        exact fidelity is measured against:

            "ground_space"     : the ground space of the target Hamiltonian
            "ground_symmetric" : its Pi = prod_i sigma^x_i = +1 part
            "final_state"      : the unperturbed final state |psi_0(T)>

        or an array of orthonormal states, which is returned as (n, 2^N).
        """
        if not isinstance(reference, str):
            return np.atleast_2d(np.asarray(reference, dtype=complex))
        if reference not in self._references:
            if reference == "ground_space":
                states = self.ground_space()
            elif reference == "ground_symmetric":
                basis = self.ground_space()
                # Pi flips every qubit: component i <-> component 2^N - 1 - i
                symmetrized = (basis + basis[:, ::-1]) / 2
                _, weights, vh = np.linalg.svd(symmetrized, full_matrices=False)
                states = vh[weights > 1e-10]
            elif reference == "final_state":
                if self.psi_final is None:
                    self.psi_final = self.final_states(np.zeros(self.n_perturbations))
                states = self.psi_final[None, :]
            else:
                raise ValueError(
                    "reference must be 'ground_space', 'ground_symmetric', "
                    f"'final_state' or an array of states, got {reference!r}"
                )
            self._references[reference] = states
        return self._references[reference]

    def population(self, states, reference="ground_space"):
        """sum_n |<r_n|psi>|^2 over the reference states (see
        reference_states). states: (2^N,) or (L, 2^N); returns a float or (L,)."""
        reference = self.reference_states(reference)
        out = (np.abs(np.asarray(states) @ reference.conj().T) ** 2).sum(axis=-1)
        return float(out) if out.ndim == 0 else out

    def fidelity(self, lam, method="exact", reference=None):
        """
        Fidelity of the final state perturbed by lam ((3N,), (3, N) or a
        batch of them).

        method = "exact" (default): re-evolves for every lam and returns the
            population of |psi_lam(T)> in `reference` ("ground_space" if None;
            see reference_states). For several references of the same lam,
            evolve once with final_states and call population on the result.
        method = "second_order": 1 - lam^T Re(G) lam (see infidelity). Only
            relative to the unperturbed final state. Not clipped: it leaves
            [0, 1] once lam is far beyond lambda_star.
        """
        if method == "second_order":
            if reference is not None and not (
                isinstance(reference, str) and reference == "final_state"
            ):
                raise ValueError(
                    "the second order is only defined relative to the "
                    "unperturbed final state (reference='final_state')"
                )
            return 1.0 - self.infidelity(lam)
        if method != "exact":
            raise ValueError(
                f"method must be 'exact' or 'second_order', got {method!r}"
            )
        if reference is None:
            reference = "ground_space"
        return self.population(self.final_states(lam), reference)

    def to_dict(self) -> dict:
        """Arrays worth saving with np.savez (G complete, so the fidelity for
        any direction of lam can be rebuilt without simulating again)."""
        if self.G is None:
            self.compute()
        return dict(
            G=self.G,
            cov=self.cov,
            lambda_star=np.array([self.lambda_star]),
            labels=np.array(self.labels),
            space=np.array(["full"]),
            T=np.array([self.tf]),
            times_ctrl=self.times_ctrl,
            schedule_ctrl=self.schedule_ctrl,
        )
