import numpy as np
from ManyBodyQutip.qutip_class import SpinOperator, SpinHamiltonian
import numpy as np
from typing import Dict
import scipy
from tqdm import trange
from typing import Optional
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def computational_basis(n):
    """
    Returns matrix of shape (2**n, n).
    Row i is the binary representation of i as a length-n bit string.
    """
    N = 2**n
    basis = np.zeros((N, n), dtype=int)
    for i in range(N):
        basis[i] = np.array(list(np.binary_repr(i, width=n)), dtype=int)
    return basis


def get_longitudinal_hamiltonian(
    jij: np.ndarray, hz: Optional[np.ndarray] = [0.0] * 100
):
    """Initialize the scipy form of the longitudinal hamiltonian on the quantum annealing protocol

    Args:
        jij (np.ndarray): Adjacency matrix in the dense format n_qubit X n_qubit
        hz (Optional[np.ndarray], optional): Longitudinal field hz. Defaults to [0.]*100.

    Returns:
        _type_: _description_
    """

    n_qubits = jij.shape[0]
    # lets start with the Z_A Z_B of the constrain
    hamiltonian_zz = 0.0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            hamiltonian_zz += SpinOperator(
                [("z", i, "z", j)], coupling=[jij[i, j]], size=n_qubits, verbose=1
            ).qutip_op

    # then the linear terms
    hamiltonian_z = 0.0
    for i in range(n_qubits):
        # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
        hamiltonian_z += SpinOperator(
            [("z", i)], coupling=[hz[i]], size=n_qubits, verbose=1
        ).qutip_op

    return (hamiltonian_zz + hamiltonian_z).data.as_scipy()


def get_driver_hamiltonian(nqubits):

    # then the linear terms
    hamiltonian_x = 0.0
    for i in range(nqubits):
        # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
        hamiltonian_x += SpinOperator(
            [("x", i)], coupling=[-1], size=nqubits, verbose=1
        ).qutip_op

    return hamiltonian_x.data.as_scipy()


def get_unbiased_catalyst_term(nqubits):

    # then the linear terms
    hamiltonian_xx = 0.0
    for i in range(nqubits):
        for j in range(i + 1, nqubits):
            # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
            hamiltonian_xx += SpinOperator(
                [("x", i, "x", j)], coupling=[1 / nqubits], size=nqubits, verbose=1
            ).qutip_op

    return hamiltonian_xx.data.as_scipy()


def get_counteradiabatic_term(driver_hamiltonian, target_hamiltonian):
    """
    A^(1) = i * alpha_0 * [H_D, H_T]
           = i * alpha_0 * (H_D @ H_T - H_T @ H_D)
    alpha_0 is applied separately via optimal_alpha(J, lam)
    """
    return 1j * (
        driver_hamiltonian @ target_hamiltonian
        - target_hamiltonian @ driver_hamiltonian
    )


def compute_exact_AGP(driver_hamiltonian, target_hamiltonian, lam, n_eigs=None):
    """
    Exact AGP from instantaneous eigenstates.

    A(lambda) = i * sum_{n!=m} <m|dH/dlambda|n> / (E_n - E_m) * |m><n|

    Parameters
    ----------
    driver_hamiltonian : sparse matrix
    target_hamiltonian : sparse matrix
    lam                : float in [0,1]
    n_eigs             : int, number of eigenstates to use.
                         None = full diagonalization (exact but slow for large n)

    Returns
    -------
    A : (dim, dim) complex array — exact AGP at this lambda
    """
    dim = driver_hamiltonian.shape[0]

    H_lam = (1 - lam) * driver_hamiltonian + lam * target_hamiltonian
    dH = target_hamiltonian - driver_hamiltonian  # partial_lambda H

    # ── diagonalize ───────────────────────────────────────────────────────────
    if n_eigs is None or n_eigs >= dim:
        # full diagonalization — exact

        evals, evecs = eigh(H_lam.toarray())
    else:
        # partial — approximate, misses transitions to high-energy states

        evals, evecs = eigsh(H_lam.astype(complex), which="SA", k=n_eigs)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    # ── compute matrix elements <m|dH|n> ─────────────────────────────────────
    # dH_matrix[m, n] = <m|dH|n>  shape (n_eigs, n_eigs)
    if scipy.sparse.issparse(dH):
        dH_dense = dH.toarray()
    else:
        dH_dense = dH

    dH_matrix = evecs.conj().T @ dH_dense @ evecs  # (n_eigs, n_eigs)

    # ── build AGP ─────────────────────────────────────────────────────────────
    # A_mn = i * <m|dH|n> / (E_n - E_m)  for n != m
    n_eigs_actual = len(evals)
    E_diff = evals[None, :] - evals[:, None]  # E_n - E_m, shape (n_eigs, n_eigs)

    # avoid division by zero on diagonal
    mask = np.abs(E_diff) > 1e-10
    A_matrix = np.zeros((n_eigs_actual, n_eigs_actual), dtype=complex)
    A_matrix[mask] = 1j * dH_matrix[mask] / E_diff[mask]

    # ← convert to sparse so expm_multiply works correctly
    A_dense = evecs @ A_matrix @ evecs.conj().T
    return scipy.sparse.csr_matrix(A_dense)
