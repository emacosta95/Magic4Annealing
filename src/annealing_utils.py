from __future__ import annotations
import scipy.sparse
from itertools import product
import numpy as np
from ManyBodyQutip.qutip_class import SpinOperator, SpinHamiltonian
import numpy as np
from typing import Dict
import scipy
from tqdm import trange
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from itertools import product
from typing import Optional
from tqdm import tqdm


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


# src/agp_pauli_decomp.py
"""
Pauli decomposition of the AGP matrix and operator size distribution.

Given A (dim x dim, dense or sparse, already computed via compute_exact_AGP),
decomposes it as  A = sum_r c_r P_r  and builds the OSD.

    c_r(lambda)  = Tr[P_r A] / dim
    q_r(lambda)  = |c_r|^2 / sum_s |c_s|^2      <- treat as probability
    P_k(lambda)  = sum_{r : weight(r)=k} q_r     <- operator size distribution
    mu(lambda)   = sum_k k * P_k                 <- mean operator size
"""


_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_I, _X, _Y, _Z]
_LABELS = ["I", "X", "Y", "Z"]


def _pauli_string(combo: tuple) -> np.ndarray:
    P = _PAULIS[combo[0]]
    for idx in combo[1:]:
        P = np.kron(P, _PAULIS[idx])
    return P


def pauli_decompose(A, n: int) -> tuple[np.ndarray, list[str]]:
    """
    Decompose A into the n-qubit Pauli basis.

    Parameters
    ----------
    A : (dim, dim) array or sparse matrix — the AGP in the many-body basis
    n : number of qubits

    Returns
    -------
    coeffs : complex (4^n,) — c_r = Tr[P_r A] / dim
    labels : list of 4^n strings, e.g. "IXZY"
    """
    if scipy.sparse.issparse(A):
        A = A.toarray()

    dim = 2**n
    assert A.shape == (dim, dim), f"Expected ({dim},{dim}), got {A.shape}"

    coeffs = np.empty(4**n, dtype=complex)
    labels = []

    for i, combo in enumerate(product(range(4), repeat=n)):
        P = _pauli_string(combo)
        # Tr[P_r A] / dim  — Pauli orthonormality: Tr[P_r P_s] = dim * delta_{rs}
        coeffs[i] = np.trace(P.conj().T @ A) / dim
        labels.append("".join(_LABELS[c] for c in combo))

    return coeffs, labels


def osd_from_coeffs(coeffs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Operator size distribution from Pauli coefficients.

    Parameters
    ----------
    coeffs : complex (4^n,)
    n      : number of qubits

    Returns
    -------
    probs : (4^n,)  — q_r = |c_r|^2 / ||A||^2_F  (per-string probabilities)
    P_k   : (n+1,)  — probability mass at Pauli weight k
    mu    : float   — mean operator size
    """
    c1 = np.abs(coeffs)
    total = c1.sum()

    if total < 1e-14:
        return np.zeros_like(c1), np.zeros(n + 1), 0.0

    probs = c1 / total
    P_k = np.zeros(n + 1)

    for i, combo in enumerate(product(range(4), repeat=n)):
        weight = sum(1 for c in combo if c != 0)
        P_k[weight] += probs[i]

    mu = float(np.dot(np.arange(n + 1), P_k))
    return probs, P_k, mu


def agp_osd(A, n: int, top_k: int = 10) -> dict:
    """
    Full pipeline: AGP matrix -> Pauli coefficients -> OSD.

    Parameters
    ----------
    A     : (dim, dim) array or sparse — AGP already computed
    n     : number of qubits
    top_k : how many dominant Pauli strings to return

    Returns
    -------
    dict with keys:
        "coeffs"     : complex (4^n,)
        "labels"     : list of 4^n strings
        "probs"      : (4^n,) normalised probabilities q_r
        "P_k"        : (n+1,) OSD
        "mean_size"  : float
        "top_strings": list of (label, q_r, c_r) sorted by q_r descending
    """
    coeffs, labels = pauli_decompose(A, n)
    probs, P_k, mu = osd_from_coeffs(coeffs, n)

    idx = np.argsort(-probs)[:top_k]
    top_strings = [(labels[i], float(probs[i]), complex(coeffs[i])) for i in idx]

    return {
        "coeffs": coeffs,
        "labels": labels,
        "probs": probs,
        "P_k": P_k,
        "mean_size": mu,
        "top_strings": top_strings,
    }


def build_agp_unitary(
    driver_hamiltonian: scipy.sparse.spmatrix,
    target_hamiltonian: scipy.sparse.spmatrix,
    lam_values: np.ndarray,
    n_eigs: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build U_AGP(lambda) = prod_{k} exp(i * A(lambda_k) * dlambda)
    by Trotter integration over lam_values.

    Returns
    -------
    U : (dim, dim) complex array — the AGP unitary at lam_values[-1]
    U_history : (n_lam, dim, dim) — unitary at each lambda step
    """
    dim = driver_hamiltonian.shape[0]
    U = np.eye(dim, dtype=complex)
    U_history = np.zeros((len(lam_values), dim, dim), dtype=complex)
    U_history[0] = U

    iterator = enumerate(zip(lam_values[:-1], lam_values[1:]))
    if verbose:
        iterator = tqdm(list(iterator), desc="Building AGP unitary")

    for k, (lam, lam_next) in iterator:
        dlam = lam_next - lam

        # AGP at current lambda — sparse, convert to dense for expm
        A = compute_exact_AGP(driver_hamiltonian, target_hamiltonian, lam, n_eigs)
        A_dense = A.toarray() if scipy.sparse.issparse(A) else A

        # Trotter step: exp(i * A * dlam)
        step = scipy.linalg.expm(1j * A_dense * dlam)
        U = step @ U
        U_history[k + 1] = U

    return U, U_history


def build_local_operators(n: int) -> dict[str, np.ndarray]:
    """
    Local excitations of the driver H_D = -sum_i X_i.
    Returns single-site X, Y, Z on each qubit as (dim x dim) matrices.

    Keys: "X_0", "Y_0", "Z_0", "X_1", ...
    Also returns collective operators Sx, Sy, Sz.
    """
    dim = 2**n
    ops = {}

    for i in range(n):
        for name, P in zip(["X", "Y", "Z"], [_X, _Y, _Z]):
            # single-site operator: I x ... x P_i x ... x I
            combo = [0] * n  # all identity
            combo[i] = ["X", "Y", "Z"].index(name) + 1
            ops[f"{name}_{i}"] = _pauli_string(tuple(combo))

    # collective spin operators (natural for the driver)
    ops["Sx"] = sum(ops[f"X_{i}"] for i in range(n)) / 2
    ops["Sy"] = sum(ops[f"Y_{i}"] for i in range(n)) / 2
    ops["Sz"] = sum(ops[f"Z_{i}"] for i in range(n)) / 2

    return ops


def pauli_decompose_operator(O: np.ndarray, n: int) -> tuple[np.ndarray, list[str]]:
    """c_r = Tr[P_r O] / dim, L1-normalized probabilities."""
    dim = 2**n
    coeffs = np.empty(4**n, dtype=complex)
    labels = []
    for i, combo in enumerate(product(range(4), repeat=n)):
        P = _pauli_string(combo)
        coeffs[i] = np.trace(P.conj().T @ O) / dim
        labels.append("".join(_LABELS[c] for c in combo))
    return coeffs, labels


def operator_spreading_agp(
    driver_hamiltonian: scipy.sparse.spmatrix,
    target_hamiltonian: scipy.sparse.spmatrix,
    lam_values: np.ndarray,
    initial_operators: Optional[dict] = None,
    n_eigs: Optional[int] = None,
) -> dict:
    """
    Track how local operators spread under U_AGP(lambda).

    For each initial operator O and each lambda:
        O(lambda) = U_AGP†(lambda) O U_AGP(lambda)
    then compute OSD of O(lambda).

    Parameters
    ----------
    initial_operators : dict of {name: (dim x dim) array}
                        defaults to single-site X, Y, Z on all qubits

    Returns
    -------
    dict keyed by operator name, each containing:
        "mu"    : (n_lam,) mean operator size
        "P_k"   : (n_lam, n+1) OSD
        "probs" : (n_lam, 4^n) full Pauli distribution
    """
    n = int(np.log2(driver_hamiltonian.shape[0]))

    if initial_operators is None:
        initial_operators = build_local_operators(n)

    # build the full unitary history
    U, U_history = build_agp_unitary(
        driver_hamiltonian, target_hamiltonian, lam_values, n_eigs
    )

    results = {name: {"mu": [], "P_k": [], "probs": []} for name in initial_operators}

    for k, lam in enumerate(tqdm(lam_values, desc="Computing OSD")):
        Uk = U_history[k]
        Ukdag = Uk.conj().T

        for name, O in initial_operators.items():
            # Heisenberg evolution
            O_lam = Ukdag @ O @ Uk

            # Pauli decomposition + OSD
            coeffs, labels = pauli_decompose_operator(O_lam, n)
            probs, P_k, mu = osd_l1(coeffs, n)

            results[name]["mu"].append(mu)
            results[name]["P_k"].append(P_k)
            results[name]["probs"].append(probs)

    # convert to arrays
    for name in results:
        results[name]["mu"] = np.array(results[name]["mu"])
        results[name]["P_k"] = np.array(results[name]["P_k"])
        results[name]["probs"] = np.array(results[name]["probs"])

    results["lam_values"] = lam_values
    results["labels"] = labels  # same for all operators

    return results
