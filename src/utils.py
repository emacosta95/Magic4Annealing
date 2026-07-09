import numpy as np
import scipy.sparse as sp


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
        self.n_A = n_A if n_A is not None else nqubits // 2
        self.n_B = nqubits - self.n_A
        self.dim_A = 2**self.n_A
        self.dim_B = 2**self.n_B
        print(
            f"Bipartition: A={self.n_A} qubits ({self.dim_A}d), "
            f"B={self.n_B} qubits ({self.dim_B}d)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def schmidt_values(self, psi: np.ndarray) -> np.ndarray:
        """
        Schmidt values (singular values of the reshaped state matrix).
        Returns (min(dim_A, dim_B),) array of singular values squared = lambda_i^2.
        """
        psi = np.asarray(psi, dtype=complex)
        psi = psi / np.linalg.norm(psi)
        M = psi.reshape(self.dim_A, self.dim_B)  # key reshape
        sv = np.linalg.svd(M, compute_uv=False)  # singular values
        return sv**2  # Schmidt coefficients lambda_i^2

    # ─────────────────────────────────────────────────────────────────────────
    def von_neumann(self, psi: np.ndarray) -> float:
        """
        S_A = -sum_i lambda_i^2 * log(lambda_i^2)
        """
        lam2 = self.schmidt_values(psi)
        lam2 = lam2[lam2 > 1e-15]  # avoid log(0)
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
        return float(np.log(np.sum(lam2**n)) / (1 - n))

    # ─────────────────────────────────────────────────────────────────────────
    def entanglement_spectrum(self, psi: np.ndarray) -> np.ndarray:
        """
        Full entanglement spectrum: xi_i = -2*log(lambda_i)
        Analogous to energy levels of the entanglement Hamiltonian H_E = -2*log(rho_A).
        Returns sorted array of xi_i values.
        """
        lam2 = self.schmidt_values(psi)
        lam2 = lam2[lam2 > 1e-15]
        return np.sort(-np.log(lam2))  # xi_i = -log(lambda_i^2)

    # ─────────────────────────────────────────────────────────────────────────
    def along_path(
        self, psi_history: np.ndarray, renyi_orders: list = [1, 2], verbose: bool = True
    ) -> dict:
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
        nsteps = psi_history.shape[0]
        results = {f"renyi_{n}": np.zeros(nsteps) for n in renyi_orders}
        results["von_neumann"] = np.zeros(nsteps)
        results["schmidt_gap"] = np.zeros(nsteps)
        results["participation"] = np.zeros(nsteps)

        for i in range(nsteps):
            lam2 = self.schmidt_values(psi_history[i])

            # von Neumann
            lam2_nz = lam2[lam2 > 1e-15]
            results["von_neumann"][i] = -np.sum(lam2_nz * np.log(lam2_nz))

            # Rényi
            for n in renyi_orders:
                if n == 1:
                    results["renyi_1"][i] = results["von_neumann"][i]
                else:
                    results[f"renyi_{n}"][i] = np.log(np.sum(lam2**n)) / (1 - n)

            # Schmidt gap: lambda_0^2 - lambda_1^2
            # small gap → highly entangled, large gap → nearly product state
            results["schmidt_gap"][i] = lam2[0] - lam2[1] if len(lam2) > 1 else lam2[0]

            # participation number: 1 / sum_i lambda_i^4
            # = effective number of active Schmidt modes
            results["participation"][i] = 1.0 / np.sum(lam2**2)

            if verbose and i % 20 == 0:
                print(
                    f'  step {i}/{nsteps}: S={results["von_neumann"][i]:.4f}, '
                    f'gap={results["schmidt_gap"][i]:.4f}'
                )

        return results


class Sector:
    """
    Restricts the Hilbert space to indices [0, ..., 2^(n-1) - 1].
    For every Z2 degenerate pair (|x>, |x̄>) where x̄ = 2^n - 1 - x,
    exactly one satisfies x < 2^(n-1) — so all degeneracies are broken.
    Sector dimension is always exactly 2^(n-1).
    """

    def __init__(self, nqubits: int):
        self.nqubits = nqubits
        self.dim = 2**nqubits
        self.dim_sector = self.dim // 2
        self.idx = np.arange(self.dim_sector, dtype=np.int32)
        print(f"Sector: {self.dim_sector} states out of {self.dim}")

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
                psi = psi / norms
            return psi

        raise ValueError(f"Unsupported shape: {obj.shape}")

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
        raise ValueError(f"Unsupported shape: {psi_sector.shape}")


"""
src/z2_sector.py

Correct Z2 symmetric-sector projection for spin models with a global
spin-flip symmetry Π = prod_i X_i, i.e. H(s) commuting with the bitwise
complement x -> x_bar = 2^N - 1 - x for all computational basis indices x.

WHY THIS EXISTS (vs. src.utils.Sector)
---------------------------------------
src.utils.Sector projects by naive index truncation:
    H_sector = H[idx, :][:, idx]     where idx = 0 .. 2^(n-1)-1

This is NOT a valid symmetric-sector projection in general. It only
reproduces correct eigenvalues if H already happens to be block-diagonal
under that raw index split - which is not guaranteed just because every
Z2-degenerate pair (x, x_bar) has exactly one representative with
x < 2^(n-1). Verified numerically on the frustrated Ising ring (N=9):
the truncated "sector" spectrum does not match any subset of the true
eigenvalues (max diff ~0.4-1.0, not ~1e-14).

The mathematically correct projection forms the symmetric/antisymmetric
combinations explicitly:

    |sym_x>  = (|x> + |x_bar>) / sqrt(2)      -- +1 eigenspace of Pi
    |asym_x> = (|x> - |x_bar>) / sqrt(2)       -- -1 eigenspace of Pi

for x in [0, 2^(n-1)), and projects via the isometry U (shape
(2^(n-1), 2^n)):

    H_sector = U @ H @ U^dagger

This was verified to reproduce the full spectrum EXACTLY (to ~1e-14) when
combining the +1 and -1 sector spectra, for the frustrated ring Hamiltonian
at multiple values of s.

USAGE
-----
For quantum annealing where the initial state |+>^N (uniform superposition)
is itself a +1 eigenstate of Pi (true for any permutation-symmetric state,
in particular the uniform superposition), and H(s) commutes with Pi for all
s (true whenever the target Hamiltonian only has ZZ terms, no local Z field,
and the driver is -sum_i X_i), the dynamics stays confined to the +1 sector
for all time. So for GRAPE/JAX optimal control and linear-ramp evolution,
project everything (Hamiltonians AND the initial state) with sign=+1 and
work entirely in the reduced (2^(n-1))-dimensional space.

For computing SRE (stabilizer Renyi entropy) or entanglement entropy, which
are computed as functionals of the FULL 2^n-dimensional wavefunction, lift
the sector state back with .lift() before calling SREJax / EntanglementEntropy.
"""


class Z2SymmetricSector:
    """
    Correct Z2 symmetric-sector projection via explicit symmetric/
    antisymmetric basis construction (NOT naive index truncation).

    Parameters
    ----------
    nqubits : int
    sign    : +1 for the symmetric (+1 eigenspace of the global flip Pi),
              -1 for the antisymmetric (-1 eigenspace) sector.
              For quantum annealing starting from the uniform superposition
              |+>^N, you want sign=+1 (that's where the dynamics lives).
    """

    def __init__(self, nqubits: int, sign: int = +1):
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 or -1")
        self.nqubits = nqubits
        self.sign = sign
        self.dim = 2**nqubits
        self.dim_sector = self.dim // 2

        idx = np.arange(self.dim_sector)
        idx_bar = self.dim - 1 - idx  # bitwise complement: flip every qubit
        self.idx = idx
        self.idx_bar = idx_bar

        rows = np.concatenate([idx, idx])
        cols = np.concatenate([idx, idx_bar])
        vals = np.concatenate(
            [
                np.full(self.dim_sector, 1.0 / np.sqrt(2)),
                np.full(self.dim_sector, sign / np.sqrt(2)),
            ]
        )
        self.U = sp.coo_matrix(
            (vals, (rows, cols)), shape=(self.dim_sector, self.dim)
        ).tocsr()

        print(
            f"Z2SymmetricSector: {self.dim_sector} states out of {self.dim} "
            f"(sign={'+1' if sign==1 else '-1'})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def project(self, obj, renormalize: bool = True):
        """
        sparse/dense matrix (2^n, 2^n) -> (2^(n-1), 2^(n-1)) via U @ obj @ U^dagger
        1D array          (2^n,)       -> (2^(n-1),) complex, renormalized
        2D array          (nsteps,2^n) -> (nsteps, 2^(n-1)) complex, renormalized

        NOTE: projecting a state that has support OUTSIDE this sector will
        silently discard that component (renormalizing what remains). Check
        the returned norm before renormalizing if you're not sure the state
        is confined to this sector.
        """
        if sp.issparse(obj):
            return self.U @ obj @ self.U.conj().T

        obj = np.asarray(obj)
        if obj.ndim == 2 and obj.shape[0] == obj.shape[1] == self.dim:
            # dense operator
            return self.U @ obj @ self.U.conj().T

        obj = obj.astype(complex)
        if obj.ndim == 1:
            psi = self.U @ obj
            if renormalize:
                norm = np.linalg.norm(psi)
                if norm > 1e-10:
                    psi = psi / norm
            return psi

        if obj.ndim == 2:
            # (nsteps, 2^n) — batch of states, project each row
            psi = (self.U @ obj.T).T
            if renormalize:
                norms = np.linalg.norm(psi, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                psi = psi / norms
            return psi

        raise ValueError(f"Unsupported shape: {obj.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    def lift(self, psi_sector: np.ndarray) -> np.ndarray:
        """
        Embed a sector wavefunction back into the full 2^n-dimensional
        Hilbert space (needed before computing SRE / entanglement entropy,
        which are functionals of the full-space state).

        psi_full = U^dagger @ psi_sector
        """
        psi_sector = np.asarray(psi_sector, dtype=complex)
        if psi_sector.ndim == 1:
            return np.asarray(self.U.conj().T @ psi_sector).ravel()
        if psi_sector.ndim == 2:
            # (nsteps, dim_sector) -> (nsteps, dim)
            return (self.U.conj().T @ psi_sector.T).T
        raise ValueError(f"Unsupported shape: {psi_sector.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    def check_confined(self, psi_full: np.ndarray, atol: float = 1e-8) -> bool:
        """
        Sanity check: does psi_full lie (numerically) entirely within this
        sector? Returns True if projecting and lifting reproduces psi_full
        up to atol. Useful before assuming dynamics stays confined.
        """
        psi_full = np.asarray(psi_full, dtype=complex)
        psi_full = psi_full / np.linalg.norm(psi_full)
        psi_sector = self.project(psi_full, renormalize=False)
        psi_reconstructed = self.lift(psi_sector)
        return np.allclose(psi_full, psi_reconstructed, atol=atol)
