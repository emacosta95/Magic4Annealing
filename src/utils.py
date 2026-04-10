import numpy as np
import scipy.sparse as sp

import numpy as np
import scipy.sparse as sp

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