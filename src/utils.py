import numpy as np
import scipy.sparse as sp

import numpy as np
import scipy.sparse as sp

import numpy as np

class EntanglementEntropy:
    """
    Bipartite entanglement entropy along a quantum annealing trajectory.

    Computes von Neumann and Rényi entropies by SVD of the Schmidt decomposition.
    The bipartition is A = first n_A qubits, B = remaining n_B qubits.

    Parameters
    ----------
    nqubits : int   — total number of qubits
    n_A     : int   — size of subsystem A (default: nqubits//2)
    """

    def __init__(self, nqubits: int, n_A: int = None):
        self.nqubits = nqubits
        self.n_A     = n_A if n_A is not None else nqubits // 2
        self.n_B     = nqubits - self.n_A
        self.dim_A   = 2 ** self.n_A
        self.dim_B   = 2 ** self.n_B
        print(f'Bipartition: A={self.n_A} qubits ({self.dim_A}d), '
              f'B={self.n_B} qubits ({self.dim_B}d)')

    # ─────────────────────────────────────────────────────────────────────────
    def schmidt_values(self, psi: np.ndarray) -> np.ndarray:
        """
        Schmidt values (singular values of the reshaped state matrix).
        Returns (min(dim_A, dim_B),) array of singular values squared = lambda_i^2.
        """
        psi  = np.asarray(psi, dtype=complex)
        psi  = psi / np.linalg.norm(psi)
        M    = psi.reshape(self.dim_A, self.dim_B)   # key reshape
        sv   = np.linalg.svd(M, compute_uv=False)    # singular values
        return sv ** 2   # Schmidt coefficients lambda_i^2

    # ─────────────────────────────────────────────────────────────────────────
    def von_neumann(self, psi: np.ndarray) -> float:
        """
        S_A = -sum_i lambda_i^2 * log(lambda_i^2)
        """
        lam2 = self.schmidt_values(psi)
        lam2 = lam2[lam2 > 1e-15]   # avoid log(0)
        return float(-np.sum(lam2 * np.log(lam2)))

    # ─────────────────────────────────────────────────────────────────────────
    def renyi(self, psi: np.ndarray, n: int = 2) -> float:
        """
        S_A^(n) = 1/(1-n) * log(sum_i lambda_i^{2n})
        n=1 limit → von Neumann (use von_neumann() instead)
        n=2 → log(Tr[rho_A^2])  — purity-based, fastest
        """
        if n == 1:
            return self.von_neumann(psi)
        lam2 = self.schmidt_values(psi)
        return float(np.log(np.sum(lam2 ** n)) / (1 - n))

    # ─────────────────────────────────────────────────────────────────────────
    def entanglement_spectrum(self, psi: np.ndarray) -> np.ndarray:
        """
        Full entanglement spectrum: xi_i = -2*log(lambda_i)
        Analogous to energy levels of the entanglement Hamiltonian H_E = -2*log(rho_A).
        Returns sorted array of xi_i values.
        """
        lam2 = self.schmidt_values(psi)
        lam2 = lam2[lam2 > 1e-15]
        return np.sort(-np.log(lam2))   # xi_i = -log(lambda_i^2)

    # ─────────────────────────────────────────────────────────────────────────
    def along_path(self, psi_history: np.ndarray,
                   renyi_orders: list = [1, 2],
                   verbose: bool = True) -> dict:
        """
        Compute entanglement measures along a full annealing trajectory.

        Parameters
        ----------
        psi_history  : (nsteps, 2^n) complex array
        renyi_orders : list of Rényi orders to compute (1 = von Neumann)

        Returns
        -------
        dict with keys:
            'von_neumann'     : (nsteps,) float
            'renyi_{n}'       : (nsteps,) float  for each n in renyi_orders
            'schmidt_gap'     : (nsteps,) float  = lambda_0^2 - lambda_1^2
            'participation'   : (nsteps,) float  = effective number of Schmidt terms
        """
        nsteps  = psi_history.shape[0]
        results = {f'renyi_{n}': np.zeros(nsteps) for n in renyi_orders}
        results['von_neumann']   = np.zeros(nsteps)
        results['schmidt_gap']   = np.zeros(nsteps)
        results['participation'] = np.zeros(nsteps)

        for i in range(nsteps):
            lam2 = self.schmidt_values(psi_history[i])

            # von Neumann
            lam2_nz = lam2[lam2 > 1e-15]
            results['von_neumann'][i] = -np.sum(lam2_nz * np.log(lam2_nz))

            # Rényi
            for n in renyi_orders:
                if n == 1:
                    results['renyi_1'][i] = results['von_neumann'][i]
                else:
                    results[f'renyi_{n}'][i] = np.log(np.sum(lam2 ** n)) / (1 - n)

            # Schmidt gap: lambda_0^2 - lambda_1^2
            # small gap → highly entangled, large gap → nearly product state
            results['schmidt_gap'][i] = lam2[0] - lam2[1] if len(lam2) > 1 else lam2[0]

            # participation number: 1 / sum_i lambda_i^4
            # = effective number of active Schmidt modes
            results['participation'][i] = 1.0 / np.sum(lam2 ** 2)

            if verbose and i % 20 == 0:
                print(f'  step {i}/{nsteps}: S={results["von_neumann"][i]:.4f}, '
                      f'gap={results["schmidt_gap"][i]:.4f}')

        return results


class Sector:
    """
    Restricts the Hilbert space to indices [0, ..., 2^(n-1) - 1].
    For every Z2 degenerate pair (|x>, |x̄>) where x̄ = 2^n - 1 - x,
    exactly one satisfies x < 2^(n-1) — so all degeneracies are broken.
    Sector dimension is always exactly 2^(n-1).
    """

    def __init__(self, nqubits: int):
        self.nqubits   = nqubits
        self.dim       = 2 ** nqubits
        self.dim_sector = self.dim // 2
        self.idx       = np.arange(self.dim_sector, dtype=np.int32)
        print(f'Sector: {self.dim_sector} states out of {self.dim}')

    def project(self, obj, renormalize=True):
        """
        sparse matrix (2^n, 2^n)  → (2^(n-1), 2^(n-1)) sparse
        1D array      (2^n,)       → (2^(n-1),) complex, renormalized
        2D array      (nsteps, 2^n)→ (nsteps, 2^(n-1)) complex, renormalized
        """
        if sp.issparse(obj):
            return obj[self.idx, :][:, self.idx]

        obj = np.asarray(obj, dtype=complex)

        if obj.ndim == 1:
            psi = obj[self.idx]
            if renormalize:
                norm = np.linalg.norm(psi)
                if norm > 1e-10:
                    psi = psi / norm
            return psi

        if obj.ndim == 2:
            psi = obj[:, self.idx]
            if renormalize:
                norms = np.linalg.norm(psi, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                psi   = psi / norms
            return psi

        raise ValueError(f'Unsupported shape: {obj.shape}')

    def lift(self, psi_sector):
        """Embed sector wavefunction back into the full Hilbert space."""
        psi_sector = np.asarray(psi_sector, dtype=complex)
        if psi_sector.ndim == 1:
            psi_full = np.zeros(self.dim, dtype=complex)
            psi_full[self.idx] = psi_sector
            return psi_full
        if psi_sector.ndim == 2:
            psi_full = np.zeros((psi_sector.shape[0], self.dim), dtype=complex)
            psi_full[:, self.idx] = psi_sector
            return psi_full
        raise ValueError(f'Unsupported shape: {psi_sector.shape}')