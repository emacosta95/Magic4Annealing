"""
Free-fermion (Nambu / BdG) representation of the 1d nearest-neighbour Ising
chain in a transverse field, for annealing schedules

    H(t) = h_driver(t) * H_driver + h_target(t) * H_target
    H_driver = - sum_i sigma^z_i
    H_target = - sum_i J_i sigma^x_i sigma^x_{i+1}

Spin convention (fixed by NambuIsing1D.build_bdg):

    H = - sum_i J_i sigma^x_i sigma^x_{i+1} - sum_i h_i sigma^z_i
      = Psi^dag  H_nambu  Psi  + const,        Psi = (c_1..c_l, c^dag_1..c^dag_l)

NORMALISATION (verified in _selftest, do not guess it):
    E_vac   = -sum_k e[l+k]
    E({n})  = E_vac + sum_k (2 e[l+k]) n_k
so the PHYSICAL quasiparticle energy is TWICE the eigh eigenvalue.

    H_nambu = [[ A ,  B  ],
               [-B*, -A* ]],     A = j + diag(h),   B = j_b

Bogoliubov slicing convention (same as src/utils_nambu_system.py):
    e, w = np.linalg.eigh(h_nambu)        # e ascending
    first l columns = negative branch, last l columns = quasiparticles g_k

PARITY (read this before using the ring):
    pbc=True flips the sign of the boundary bond -> antiperiodic fermions,
    which is exact ONLY in the even-parity sector P = prod_i sigma^z_i = +1
    (Mbeng, Russomanno, Santoro, arXiv:2009.09208, Eq. 32).  The state
    evolved from the driver ground state (all up) stays in that sector.
    The eigh "vacuum" (all g_k empty) is the minimum over BOTH parities: for a
    frustrated ring (odd l, prod_i J_i < 0) one quasiparticle energy crosses
    zero at prod_i h_i = prod_i |J_i|, and past that point the eigh vacuum is
    odd, i.e. unphysical (zero domain walls on a frustrated ring).  The
    physical levels are then those with an ODD number of quasiparticles.
    The class fixes the physical sector once (self.sector_parity) and picks
    the right occupations automatically via sign Pf(Gamma).

numpy + pfapack (Wimmer, ACM TOMS 38, 30 (2012)).

Ema / Magic4Annealing
"""

import heapq

import numpy as np


def _dexp_weights(ek, dt):
    """F_jk = (phi_j - phi_k)/(e_j - e_k), phi = exp(-2i dt e); diag -> -2i dt phi.
    Exact derivative of exp(-2i dt H) in the eigenbasis (Duncan et al. 2025). https://arxiv.org/abs/2501.16436
    """
    phi = np.exp(-2j * dt * ek)
    de = ek[:, None] - ek[None, :]
    # condition of degeneracy
    deg = np.abs(de) < 1e-10
    # computation of the weight
    f = np.where(deg, 0.0, (phi[:, None] - phi[None, :]) / np.where(deg, 1.0, de))
    diag_val = -2j * dt * 0.5 * (phi[:, None] + phi[None, :])
    return np.where(deg, diag_val, f), phi


# NOTE: `from pfapack import pfaffian` imports the MODULE (not callable).
# ctypes = compiled backend (~30x faster than pure python at 2l=200).
from pfapack.ctypes import pfaffian


class NambuIsing1D:
    """Nambu representation of the 1d nn transverse-field Ising chain.

    Args:
        l:     number of qubits
        j_vec: [l] couplings, j_vec[i] on bond (i, i+1); j_vec[-1] is the
               boundary bond (ignored if pbc=False)
        pbc:   periodic spin chain (antiperiodic fermions, even sector)
    """

    def __init__(self, l: int, j_vec: np.ndarray, pbc: bool):
        self.l = l
        self.pbc = pbc
        self.j_vec = np.asarray(j_vec, dtype=np.float64)

        # build_bdg is AFFINE in (h, j_vec) with zero constant term, hence
        # exactly  H_nambu(h_d * 1, h_t * j_vec) = h_d * M_driver + h_t * M_target
        # which is precisely the linear structure the schedulers assume.
        self.m_driver = self.build_bdg(np.ones(l), np.zeros(l), pbc)
        self.m_target = self.build_bdg(np.zeros(l), self.j_vec, pbc)

        # parity of the physical sector = parity of the driver ground state
        _, w0 = np.linalg.eigh(self.m_driver)
        self.sector_parity = self.vacuum_parity(w0)

    # -----------------------------------------------------------------------
    # 0.  Model constructors
    # -----------------------------------------------------------------------
    @classmethod
    def frustrated_ring(cls, N: int, J: float = 1.0, JL: float = 0.5, JR: float = 0.45):
        """Frustrated ring of Cote et al. / Werner et al. (Z -> sigma^x):

            H_p = -sum_j J_j sx_j sx_{j+1},  J_N = -J_R,  J_{(N-/+1)/2} = J_L,  else J
            0 < J_R < J_L < J

        Same model as frustrated_ring_jij_hz.  Verified N=7:
        E0 = -(N-3)J + J_R - 2J_L, lowest levels vs ED to 1e-14 for all s;
        vacuum parity flips at s* = 1/(1 + gmean|J_j|) ~ 0.577.
        """
        assert N % 2 == 1, "N must be odd"
        jv = np.full(N, float(J))
        jv[(N - 1) // 2 - 1] = JL
        jv[(N + 1) // 2 - 1] = JL
        jv[-1] = -JR
        return cls(N, jv, pbc=True)

    @staticmethod
    def build_bdg(h: np.ndarray, j_vec: np.ndarray, pbc: bool) -> np.ndarray:
        """Nambu/BdG matrix [2l,2l], following "The quantum Ising chain for
        beginners" (Mbeng, Russomanno, Santoro).

        pbc: if True the boundary bond is sign-flipped -> antiperiodic
             (even-parity) sector.  See the PARITY note at the top.
        """
        h = np.asarray(h, dtype=np.float64)
        l = h.shape[-1]

        # bond[i] = coupling on the bond (i, i+1 mod l); bond[l-1] is the boundary
        bond = np.array(j_vec, dtype=np.float64)  # copy
        bond[-1] = -1 * bond[-1] if pbc else 0.0

        # T[i, i+1] = bond[i]  -> symmetric hopping, antisymmetric pairing.
        # NOTE: the original j_l/j_r + roll construction is only correct for a
        # UNIFORM j_vec.  For site-resolved couplings it produces
        # j[i,i+1] = -J_i/2 but j[i+1,i] = -J_{i+1}/2, i.e. a NON-Hermitian A
        # block; eigh then silently symmetrises using the lower triangle and
        # returns wrong eigenvalues.  Verified: max|H - H^T| = 0.51 for random
        # j_vec, 0.0 for uniform.
        idx = np.arange(l)
        t_mat = np.zeros((l, l))
        t_mat[idx, (idx + 1) % l] = bond

        j = -0.5 * (t_mat + t_mat.T)  # hopping block
        b = -0.5 * (t_mat - t_mat.T)  # pairing block
        a = j + np.diag(h)  # transverse field

        h_nambu = np.zeros((2 * l, 2 * l))
        h_nambu[:l, :l] = a
        h_nambu[:l, l:] = b
        h_nambu[l:, :l] = -1 * np.conj(b)
        h_nambu[l:, l:] = -1 * np.conj(a)
        return h_nambu

    # -----------------------------------------------------------------------
    # 1.  Instantaneous Hamiltonian
    # -----------------------------------------------------------------------
    def hamiltonian(self, h_driver: float, h_target: float) -> np.ndarray:
        return float(h_driver) * self.m_driver + float(h_target) * self.m_target

    def diagonalize(self, h_driver: float, h_target: float):
        """e [2l] ascending, w [2l,2l].  Quasiparticle energies: 2*e[l:].

        No zero-mode handling: on a ring there are no free edge Majoranas, so an
        exactly zero quasiparticle energy only occurs if a time step lands on
        the parity crossing s* itself (verified N=7, 15: min eps ~ 3e-4 on a
        2001-point grid).  OBC at h -> 0 WOULD need it (degenerate e = 0 pair,
        eigh basis not particle-hole paired).
        """
        return np.linalg.eigh(self.hamiltonian(h_driver, h_target))

    # -----------------------------------------------------------------------
    # 2.  Time evolution
    # -----------------------------------------------------------------------
    def evolve(self, h_driver, h_target, dt: float, w0=None, store_every: int = 0):
        """Piecewise-constant BdG propagation:  i dW/dt = 2 H_nambu(t) W.

        Args:
            h_driver, h_target: [nsteps] schedules, exactly what
                                get_driving() returns (Schedule, SparseGRAPEModel).
            w0:                 [2l,2l] initial Bogoliubov matrix; default is the
                                ground state of H(t=0).
            store_every:        keep a snapshot every n steps (0 = none), to
                                save memory.
        Returns:
            w_final [2l,2l] complex, and the list of snapshots.
        """
        if w0 is None:
            _, w0 = self.diagonalize(h_driver[0], h_target[0])
        w = np.asarray(w0, dtype=np.complex128)

        snapshots = []
        for i in range(len(h_driver)):
            # H is Hermitian -> eigh is faster and more stable than a general expm
            ek, vk = self.diagonalize(h_driver[i], h_target[i])
            # factor 2: see NORMALISATION (Mbeng, Russomanno, Santoro)
            prop = (vk * np.exp(-2j * dt * ek)) @ vk.conj().T
            w = prop @ w
            if store_every and (i % store_every == 0):
                snapshots.append(w.copy())
        return w, snapshots

    def instantaneous_states(
        self, h_driver, h_target, store_every: int = 1, level: int = 0
    ):
        """Instantaneous PHYSICAL eigenstate `level` (0 = ground state) along a
        schedule, at the same steps as evolve(..., store_every): i % store_every == 0.

        Adiabatic reference for the evolved snapshots.  Each state is a
        Bogoliubov vacuum W1 [2l, l] built with excited_vacuum, so past the
        parity flip of the frustrated ring the physical (odd-occupation)
        ground state is returned, not the unphysical eigh vacuum.
        Degenerate levels: the state within the multiplet is arbitrary.

        Returns:
            states [n_snap] list of W1 [2l, l],  energies [n_snap],  steps [n_snap]
        """
        steps = np.arange(0, len(h_driver), max(int(store_every), 1))
        states, energies = [], []
        for i in steps:
            en, occs, (_, w_inst) = self.levels(h_driver[i], h_target[i], level + 1)
            states.append(self.excited_vacuum(w_inst, occs[level]))
            energies.append(en[level])
        return states, np.array(energies), steps

    # -----------------------------------------------------------------------
    # 3.  Parity
    # -----------------------------------------------------------------------
    def vacuum_parity(self, w: np.ndarray) -> int:
        """Fermion parity (+1/-1) of the Bogoliubov vacuum of w.

        P = prod_i sigma^z_i = prod_i (-i A_i B_i) is the product of ALL 2l
        Majoranas, so by Wick <P> = Pf(Gamma) = +-1 for a pure Gaussian state.
        """
        return int(np.sign(pfaffian(self.majorana_covariance(w))))

    def relative_parity(self, w_inst: np.ndarray) -> int:
        """Quasiparticle-number parity (0/1) of the instantaneous eigenstates
        that live in the physical sector.

        NOT always 0: on a frustrated ring the eigh vacuum flips parity once a
        quasiparticle energy crosses zero (see PARITY note).  Verified L=5 AFM
        ring: parity=0 gives sum P = 0 at s=0.8, this value gives 1.
        """
        return 0 if self.vacuum_parity(w_inst) == self.sector_parity else 1

    # -----------------------------------------------------------------------
    # 4.  Spectrum
    # -----------------------------------------------------------------------
    @staticmethod
    def lowest_levels(eps, n_levels, parity=None):
        """The lowest `n_levels` MANY-BODY excitations without enumerating 2^l states.

        Best-first (heap) search over occupation patterns: each pop yields the
        next smallest excitation energy, each pop pushes at most l children.
        Cost O(n_levels * l * log(n_levels * l)).

        Args:
            eps:    [l] quasiparticle energies, ALREADY including the factor 2
                    (i.e. 2*e[l:]), ascending.
            parity: None -> all patterns; 0 / 1 -> even / odd number of
                    quasiparticles only.
        Returns:
            excitation energies above E_vac [n_levels], occupation tuples.

        Reference: best-first search (uniform-cost search / Dijkstra on a DAG
        with nonnegative edge weights) over the subset lattice -- the standard
        "K smallest subset sums" pattern (cf. LeetCode 2386).
        """
        eps = np.asarray(eps, dtype=float)
        l = len(eps)
        heap = [(0.0, -1, ())]
        out, occs = [], []
        while heap and len(out) < n_levels:
            de, last, occ = heapq.heappop(heap)
            if parity is None or (len(occ) % 2) == parity:
                out.append(de)
                occs.append(occ)
            for k in range(last + 1, l):
                heapq.heappush(heap, (de + eps[k], k, occ + (k,)))
        return np.array(out), occs

    def levels(self, h_driver: float, h_target: float, n_levels: int = 10):
        """Lowest n_levels PHYSICAL many-body levels of H(h_driver, h_target).

        Returns energies [n_levels], occupation tuples, and (e, w) of eigh.
        """
        e, w = self.diagonalize(h_driver, h_target)
        eps = 2.0 * e[self.l :]
        e_vac = -0.5 * eps.sum()
        de, occs = self.lowest_levels(eps, n_levels, parity=self.relative_parity(w))
        return e_vac + de, occs, (e, w)

    def spectrum_along_schedule(self, h_driver, h_target, n_levels: int = 10):
        """Lowest n_levels physical levels at every schedule point, [nsteps, n_levels].

        Labels are sorted energies: they do NOT track a single adiabatic state
        through a crossing.
        """
        out = np.zeros((len(h_driver), n_levels))
        for i in range(len(h_driver)):
            out[i] = self.levels(h_driver[i], h_target[i], n_levels)[0]
        return out

    # -----------------------------------------------------------------------
    # 5.  Populations of the instantaneous levels
    # -----------------------------------------------------------------------
    def excited_vacuum(self, w_inst: np.ndarray, occ) -> np.ndarray:
        """Multi-quasiparticle state prod_{k in occ} g^dag_k |0_inst> written as
        a Bogoliubov vacuum.  Bally & Bender, EPJA 57, 69 (2021)
        [arXiv:2010.14169], Eq. (14): (U_lk, V_lk) -> (V_lk^*, U_lk^*) for every
        excited k; each swap flips the number parity.

        In Nambu form: the vacuum is spanned by W1 = Sx Wp^*, and exciting mode
        k replaces column k of W1 with Wp_k.  Returns W1' [2l, l].

        GAUGE: W1 is rebuilt from Wp on purpose.  eigh returns the negative
        branch in REVERSED order with independent phases (arbitrary rotations
        if degenerate), so w_inst[:, :l] cannot be used column by column.
        """
        l = self.l
        wp = np.asarray(w_inst, dtype=np.complex128)[:, l:]
        w1 = np.concatenate([wp[l:].conj(), wp[:l].conj()], axis=0)  # Sx Wp^*
        occ = list(occ)
        # switching of the columns for describing a new effective vacuum for \prod_{a in occ} \gamma_a^dag
        w1[:, occ] = wp[:, occ]
        return w1

    def level_probabilities(
        self, w_t, h_driver: float, h_target: float, n_levels: int = 10
    ):
        """P_n = |<n_inst | psi(t)>|^2 for the lowest n_levels physical levels.

        Every eigenstate |S> is itself a Bogoliubov vacuum (excited_vacuum), so
        Onishi gives directly

            P_S = |det( W1_S^dag  W1_t )| ,     W1_t = w_t[:, :l]

        (Nambu doubling: this |det| is already the SQUARED overlap).  O(l^3)
        per level, no inversion, valid also when P_0 -> 0.
        Verified vs exact evolution (L=5,6,8 OBC; L=5 AFM ring; N=7
        frustrated ring): max |dP| ~ 1e-14, sum P = 1.

        Returns energies [n_levels], probabilities [n_levels], occupations.
        """
        energies, occs, (_, w_inst) = self.levels(h_driver, h_target, n_levels)
        wt1 = np.asarray(w_t, dtype=np.complex128)[:, : self.l]
        probs = np.array(
            [
                abs(np.linalg.det(self.excited_vacuum(w_inst, s).conj().T @ wt1))
                for s in occs
            ]
        )
        return energies, probs, occs

    def residual_energy(self, w_t, h_driver: float, h_target: float) -> float:
        """<psi(t)| H |psi(t)> - E_gs  for the instantaneous H, exact.

        Occupations of the instantaneous modes: overlap C = w_inst^dag w_t,
        B = C[l:, :l],  n_k = diag(B B^dag)  (ROW norms -- the column
        convention is wrong: verified 0.887 vs 0.347).
        E(psi) = E_vac + sum_k eps_k n_k exactly, and E_gs is the lowest
        PHYSICAL level, which is E_vac + eps_0 past the parity flip (using
        E_vac there gives an offset of eps_0 = 0.9 for the N=7 frustrated ring).
        """
        l = self.l
        energies, _, (e, w_inst) = self.levels(h_driver, h_target, 1)
        eps = 2.0 * e[l:]
        c = w_inst.conj().T @ np.asarray(w_t, dtype=np.complex128)
        b = c[l:, :l]
        n_k = np.clip(np.real(np.sum(np.abs(b) ** 2, axis=1)), 0.0, 1.0)
        e_psi = -0.5 * eps.sum() + float(eps @ n_k)
        return e_psi - float(energies[0])

    # -----------------------------------------------------------------------
    # 6.  Majorana covariance matrix  ->  Pauli expectation values
    # -----------------------------------------------------------------------
    # Majoranas:  A_i = c_i + c_i^dag ,  B_i = -i (c_i - c_i^dag)
    # ordered as  w_(2i) = A_i , w_(2i+1) = B_i.
    #
    # Gamma_{mn} = i (delta_{mn} - <w_m w_n>)  is REAL antisymmetric for ANY state
    # (i[w_m,w_n] is Hermitian).  For a Gaussian state
    #
    #     < i^m  w_{i1} ... w_{i2m} >  =  Pf( Gamma[{i},{i}] )
    #
    # IMPORTANT for dynamics: the ground state of a REAL BdG matrix has vanishing
    # <AA> and <BB> Majorana blocks, which is what lets utils_nambu_system.py get
    # <SX SX> from a plain determinant of a single block.  Once W(t) is complex
    # those blocks are non-zero and the determinant formula is WRONG.
    # Use the Pfaffian of the full Gamma.
    def majorana_covariance(self, w: np.ndarray) -> np.ndarray:
        """Gamma [2l,2l], real antisymmetric, from the Bogoliubov matrix w."""
        l = self.l
        w1 = np.asarray(w, dtype=np.complex128)[:, :l]
        r = w1 @ w1.conj().T  # R_{mu,nu} = <Psi_mu Psi_nu^dag>

        # <Psi_mu Psi_nu> = R[mu, swap(nu)]
        swap = np.zeros((2 * l, 2 * l))
        swap[:l, l:] = np.eye(l)
        swap[l:, :l] = np.eye(l)
        pmat = r @ swap

        # Omega: w_maj = Omega Psi
        omega = np.zeros((2 * l, 2 * l), dtype=np.complex128)
        rows = np.arange(l)
        omega[2 * rows, rows] = 1.0
        omega[2 * rows, l + rows] = 1.0
        omega[2 * rows + 1, rows] = -1j
        omega[2 * rows + 1, l + rows] = 1j

        ww = omega @ pmat @ omega.T  # <w_m w_n>
        gamma = 1j * (np.eye(2 * l) - ww)
        gamma = 0.5 * (gamma - gamma.T)  # enforce antisymmetry
        return gamma.real

    # -----------------------------------------------------------------------
    # 6b. Entanglement entropy
    # -----------------------------------------------------------------------
    def entanglement_entropy(self, w: np.ndarray, block, base: float = 2.0) -> float:
        """Von Neumann entropy S_A of a CONTIGUOUS block of spins (evolving state).

        The reduced state of a Gaussian state is Gaussian, with covariance
        Gamma_A = Gamma restricted to the block's 2|A| Majoranas.  i Gamma_A is
        Hermitian with eigenvalues +-nu_j (0 <= nu_j <= 1), and

            S_A = - sum_{all 2|A| eigenvalues} p log p ,   p = (1 + nu)/2
                = sum_j H2((1 + nu_j)/2)

        Contiguous blocks only: the Jordan-Wigner string does not cut the block,
        so for a fixed-parity state spin and fermion reduced states coincide
        (Vidal, Latorre, Rico, Kitaev, PRL 90, 227902 (2003)).

        Args:
            w:     w_t from evolve() (any matrix whose first l columns span the
                   vacuum, also the [2l, l] output of excited_vacuum()).
            block: (start, stop) sites, or an int la meaning (0, la).
        """
        start, stop = (0, int(block)) if np.isscalar(block) else block
        maj = np.arange(2 * start, 2 * stop)
        gamma_a = self.majorana_covariance(w)[np.ix_(maj, maj)]
        p = np.clip(0.5 * (1.0 + np.linalg.eigvalsh(1j * gamma_a)), 1e-300, 1.0)
        return float(-np.sum(p * np.log(p)) / np.log(base))

    def eigenstate_entanglement_entropy(
        self, h_driver: float, h_target: float, block, level: int = 0, base: float = 2.0
    ) -> float:
        """S_A of the `level`-th PHYSICAL instantaneous eigenstate of H(h_driver, h_target)
        (level 0 = ground state), same ordering as levels() / level_probabilities().

        The eigenstate is built as a Bogoliubov vacuum with excited_vacuum
        (TAURUS Eq. 14), so the parity flip of the frustrated ring is handled.
        Degenerate levels: the result depends on which state of the multiplet
        is picked (any combination is an eigenstate).
        """
        _, occs, (_, w_inst) = self.levels(h_driver, h_target, level + 1)
        return self.entanglement_entropy(
            self.excited_vacuum(w_inst, occs[level]), block, base
        )

    # Pauli string -> Majorana support, via Jordan-Wigner:
    #   sigma^x_j = (prod_{m<j} -i a_m b_m) a_j
    #   sigma^y_j = (prod_{m<j} -i a_m b_m) b_j
    #   sigma^z_j = -i a_j b_j
    # JW is a Clifford circuit, so the SRE of the spin state equals the SRE
    # computed from these fermionic data -- no ambiguity.
    @staticmethod
    def pauli_to_majorana(string: str):
        """P = phase * gamma_x, x sorted.  JW: sz = -i a b,
        sx/sy = (prod_{m<j} -i a_m b_m) a_j / b_j.  State-independent."""
        support, phase = [], 1.0 + 0j
        for j, s in enumerate(string):
            if s == "I":
                continue
            if s == "Z":
                support += [2 * j, 2 * j + 1]
                phase *= -1j
            elif s in ("X", "Y"):
                for m in range(j):
                    support += [2 * m, 2 * m + 1]
                    phase *= -1j
                support += [2 * j] if s == "X" else [2 * j + 1]
            else:
                raise ValueError(f"bad Pauli '{s}'")
        for _ in range(len(support)):  # sort, -1 per swap
            for k in range(len(support) - 1):
                if support[k] > support[k + 1]:
                    support[k], support[k + 1] = support[k + 1], support[k]
                    phase = -phase
        idx = []
        for x in support:  # w^2 = 1
            if idx and idx[-1] == x:
                idx.pop()
            else:
                idx.append(x)
        return phase, np.array(idx, dtype=int)

    @staticmethod
    def __majorana_expectation(gamma: np.ndarray, idx) -> complex:
        """<gamma_x> = (-i)^(|x|/2) Pf(Gamma|_x);  0 for odd |x|."""
        if len(idx) % 2:
            return 0.0
        p = len(idx) // 2
        return (-1j) ** p * (pfaffian(gamma[np.ix_(idx, idx)]) if p else 1.0)

    def __pauli_expectation(self, gamma: np.ndarray, string: str) -> float:
        phase, idx = self.pauli_to_majorana(string)
        return float((phase * self.__majorana_expectation(gamma, idx)).real)

    def expectation(self, gamma: np.ndarray, index, coupling) -> float:
        """<O> for O = sum_t coupling[t] * P_t, with the same `index`/`coupling`
        convention as ManyBodyQutip.qutip_class.SpinOperator, e.g.
            index=[("x", 0, "x", 1), ("z", 3)], coupling=[1.0, 0.5]
        One Pauli per site per term.
        """
        total = 0.0
        for term, c in zip(index, coupling):
            ops, sites = term[0::2], term[1::2]
            if len(set(sites)) != len(sites):
                raise ValueError(f"repeated site in term {term}")
            string = ["I"] * self.l
            for o, j in zip(ops, sites):
                string[int(j)] = o.upper()
            total += c * self.__pauli_expectation(gamma, "".join(string))
        return float(total)

    # -----------------------------------------------------------------------
    # 7.  Non-stabilizerness: Majorana sampling (Algorithm 1)
    # -----------------------------------------------------------------------
    @staticmethod
    def _minor_det(gamma, idx, n_unit):
        """det of (D + Gamma)|_idx, D = 1 on the last n_unit entries of idx, 0 before."""
        if len(idx) == 0:
            return 1.0
        # np.ix_ creates a submatrix with the indices idx X idx, which is what we want for the principal minor.
        sub = gamma[np.ix_(idx, idx)].copy()
        if n_unit:
            k = np.arange(len(idx) - n_unit, len(idx))
            sub[k, k] += 1.0
        return float(np.linalg.det(sub))

    def majorana_sampling(self, gamma: np.ndarray, n_samples: int, seed: int = 0):
        """Algorithm 1 of Collura, De Nardis, Alba, Lami, arXiv:2412.05367
        ("The non-stabilizerness of fermionic Gaussian states").

        Perfect (non-Markov) sampling of Majorana monomials x in {0,1}^{2L} from

            pi(x) = det(Gamma|_x) / det(1 + Gamma)                     Eq. (6)

        (= <P>^2 / 2^L for the Pauli string P <-> gamma_x).  Chain rule, Eq. (9),
        with marginals, Eq. (11):

            pi(x_1..x_mu) = det[(1_[mu+1,2L] + Gamma)|_(x_1..x_mu, 1..1)] / det(1+Gamma)

        Setting x_mu = 1 keeps index mu with diagonal 0; x_mu = 0 drops it; the
        two determinants sum to the previous marginal (det is linear in a
        diagonal entry), so p(x_mu = 1 | x) = det_1 / (det_0 + det_1).
        The sign convention of Gamma is irrelevant (principal minors of an
        antisymmetric matrix are invariant under Gamma -> -Gamma).
        Cost O(L^4) per sample.

        Returns x [n_samples, 2L] (bool) and log pi(x) [n_samples] (natural log).
        """
        gamma = np.asarray(gamma, dtype=float)
        n = gamma.shape[0]
        rng = np.random.default_rng(seed)
        _, logdet_norm = np.linalg.slogdet(np.eye(n) + gamma)

        xs = np.zeros((n_samples, n), dtype=bool)
        logp = np.zeros(n_samples)
        for s in range(n_samples):
            chosen = []
            for mu in range(n):
                rest = list(range(mu + 1, n))
                d0 = self._minor_det(gamma, chosen + rest, n_unit=len(rest))
                d1 = self._minor_det(gamma, chosen + [mu] + rest, n_unit=len(rest))
                d0, d1 = max(d0, 0.0), max(d1, 0.0)  # both >= 0 up to rounding
                if rng.random() < d1 / (d0 + d1):
                    chosen.append(mu)
                    xs[s, mu] = True
            # pi(x) = det(Gamma|_x) / det(1+Gamma)
            ld = np.linalg.slogdet(gamma[np.ix_(chosen, chosen)])[1] if chosen else 0.0
            logp[s] = ld - logdet_norm
        return xs, logp

    def sre(
        self, w: np.ndarray, alpha: float = 2, n_samples: int = 2000, seed: int = 0
    ):
        """Stabilizer Renyi entropy of the Gaussian state w (Algorithm 1).

            sum_P pi^alpha = E_{x ~ pi}[ pi(x)^(alpha-1) ]

            M_alpha   = log2(sum pi^alpha) / (1-alpha) - log2 D,        D = 2^L
            M~_alpha  filtered version, Eqs. (7)-(8): I and the parity string
                      (pi = 1/D each for a pure Gaussian state) removed and
                      pi renormalised; subtracted exactly from the estimate.

        Returns dict(m_alpha, m_alpha_filtered, err), results in bits, err =
        standard error of M_alpha (delta method).
        Verified: N=7 frustrated ring after anneal, M2 exact 3.151 vs 3.148(32);
        filtered 3.344 vs 3.341.
        """
        gamma = self.majorana_covariance(w)
        l = self.l
        _, logp = self.majorana_sampling(gamma, n_samples, seed)
        wts = np.exp((alpha - 1) * logp)
        mean, sem = wts.mean(), wts.std(ddof=1) / np.sqrt(n_samples)
        d = 2.0**l
        m_alpha = np.log2(mean) / (1 - alpha) - l
        # filtered: remove the two trivial strings, renormalise pi~ = pi / (1 - 2/D)
        s_f = (mean - 2 * d ** (-alpha)) / (1 - 2 / d) ** alpha
        m_filt = np.log2(s_f) / (1 - alpha) - np.log2(d - 2)
        err = sem / (mean * abs(1 - alpha) * np.log(2))
        return dict(
            m_alpha=float(m_alpha), m_alpha_filtered=float(m_filt), err=float(err)
        )

    def grape_energy_and_grad(
        self, h_driver, h_target, dt, h_ref=(0.0, 1.0), w0=None, return_state=False
    ):
        """E = <psi_T| H_ref |psi_T> = tr(W1^dag H_ref W1) and EXACT dE/dh_driver_i,
        dE/dh_target_i for the piecewise-constant propagator of `evolve`.
        Forward: W1_{i+1} = U_i W1_i,  U_i = exp(-2i dt H_i).
        Backward co-state: X_N = H_ref W1_N,  X_i = U_i^dag X_{i+1}.
        dE/da_i = 2 Re tr(X_{i+1}^dag  dU_i/da  W1_i),  dU/da via Daleckii-Krein.
        Cost O(nsteps * l^3), memory nsteps * 2l * l.
        """
        l, nsteps = self.l, len(h_driver)
        if w0 is None:
            _, w0 = self.diagonalize(h_driver[0], h_target[0])
        w = np.asarray(w0, dtype=np.complex128)[:, :l]
        # in this formalism every hamiltonian is a linear combination of the two building block hamiltonians, H_D and H_T
        hr = self.hamiltonian(*h_ref)
        # initialize the state the optimize the usage
        ws = np.empty((nsteps + 1, 2 * l, l), dtype=np.complex128)
        ws[0] = w
        for i in range(nsteps):
            ek, vk = self.diagonalize(h_driver[i], h_target[i])
            # time evolution in the eigenbasis of the instantaneous hamiltonian
            w = (vk * np.exp(-2j * dt * ek)) @ (vk.conj().T @ w)
            ws[i + 1] = w
        energy = float(np.real(np.trace(w.conj().T @ hr @ w)))

        g_drv, g_tgt = np.zeros(nsteps), np.zeros(nsteps)
        x = hr @ w
        md, mt = self.m_driver, self.m_target
        # time reversal
        for i in reversed(range(nsteps)):
            ek, vk = self.diagonalize(h_driver[i], h_target[i])
            # this formula is from eq 147 paper "Taming quantum systems:
            # A tutorial for using shortcuts-to-adiabaticity, quantum optimal control, & reinforcement learning" by Duncan and P. Poggi
            f, phi = _dexp_weights(ek, dt)
            xv, yv = vk.conj().T @ x, vk.conj().T @ ws[i]  # eigenbasis
            k = xv.conj() @ yv.T  # (x^* y^T)_jk
            g_drv[i] = 2.0 * np.real(np.sum((vk.conj().T @ md @ vk) * f * k))
            g_tgt[i] = 2.0 * np.real(np.sum((vk.conj().T @ mt @ vk) * f * k))
            x = vk @ (np.conj(phi)[:, None] * xv)  # X_i = U_i^dag X_{i+1}
        if return_state:
            return energy, g_drv, g_tgt, ws[-1]
        return energy, g_drv, g_tgt


# ---------------------------------------------------------------------------
# Self-test against exact diagonalisation (run: python free_fermions_utils.py)
# ---------------------------------------------------------------------------
def _exact_ising(l, h, j_vec, pbc):
    """Dense H = -sum J_i sx_i sx_{i+1} - sum h_i sz_i."""
    sx = np.array([[0, 1], [1, 0]], dtype=float)
    sz = np.array([[1, 0], [0, -1]], dtype=float)

    def op(mat, site):
        out = np.array([[1.0]])
        for k in range(l):
            out = np.kron(out, mat if k == site else np.eye(2))
        return out

    ham = np.zeros((2**l, 2**l))
    for i in range(l):
        ham -= h[i] * op(sz, i)
    for i in range(l if pbc else l - 1):
        ham -= j_vec[i] * op(sx, i) @ op(sx, (i + 1) % l)
    return ham


def _even_sector(l):
    z = np.array([1, -1])
    par = np.array([1])
    for _ in range(l):
        par = np.kron(par, z)
    return par > 0


def _exact_entropy(psi, l, block, base=2.0):
    """S of sites [start, stop) from a dense state vector (Schmidt values)."""
    start, stop = block
    keep = list(range(start, stop))
    rest = [i for i in range(l) if i not in keep]
    m = np.transpose(psi.reshape([2] * l), keep + rest).reshape(2 ** len(keep), -1)
    p = np.linalg.svd(m, compute_uv=False) ** 2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)) / np.log(base))


def _selftest_statics(l=6, seed=1):
    import itertools

    print("--- statics (ring ground state, even sector) ---")
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.3, 1.7, size=l)
    j = rng.uniform(0.3, 1.7, size=l)
    model = NambuIsing1D(l, j, pbc=True)
    # a generic site-resolved field is not h_d*1, so diagonalise build_bdg directly
    e, w = np.linalg.eigh(NambuIsing1D.build_bdg(h, j, pbc=True))
    even = _even_sector(l)
    ev, evec = np.linalg.eigh(_exact_ising(l, h, j, pbc=True)[np.ix_(even, even)])
    psi = np.zeros(2**l)
    psi[even] = evec[:, 0]

    eps = 2.0 * e[l:]
    lows = (
        -0.5 * eps.sum()
        + model.lowest_levels(eps, 8, parity=model.relative_parity(w))[0]
    )
    print(f"lowest 8 levels, max dev vs exact: {np.abs(lows - ev[:8]).max():.2e}")

    gamma = model.majorana_covariance(w)
    pauli = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    worst = 0.0
    for s in [
        "Z" + "I" * (l - 1),
        "XX" + "I" * (l - 2),
        "IZZ" + "I" * (l - 3),
        "XIIX" + "I" * (l - 4),
        "YY" + "I" * (l - 2),
    ]:
        mat = np.array([[1.0 + 0j]])
        for ch in s:
            mat = np.kron(mat, pauli[ch])
        worst = max(
            worst,
            abs(np.real(psi.conj() @ mat @ psi) - model.pauli_expectation(gamma, s)),
        )
    print(f"worst Pauli discrepancy: {worst:.2e}")

    tot = sum(
        model.pauli_expectation(gamma, "".join(c)) ** 4
        for c in itertools.product("IXYZ", repeat=l)
    )
    res = model.sre(w, alpha=2, n_samples=4000)
    print(
        f"M2  brute force={-np.log2(tot / 2**l):.5f}   "
        f"Algorithm 1={res['m_alpha']:.5f} +- {res['err']:.5f}"
    )


def _selftest_dynamics():
    """level_probabilities / residual_energy vs exact many-body evolution."""
    from scipy.linalg import expm

    print("\n--- dynamics vs exact diagonalisation ---")
    rng = np.random.default_rng(0)
    j_fm = rng.uniform(0.5, 1.5, 6)
    cases = [
        ("FM ring L=6", NambuIsing1D(6, j_fm, pbc=True), 3.0),
        ("AFM ring L=5", NambuIsing1D(5, -np.ones(5), pbc=True), 3.0),
        ("frustrated N=7", NambuIsing1D.frustrated_ring(7), 10.0),
    ]
    for name, model, tf in cases:
        l = model.l
        hd_s = _exact_ising(l, np.ones(l), np.zeros(l), model.pbc)
        ht_s = _exact_ising(l, np.zeros(l), model.j_vec, model.pbc)
        nsteps = 600
        dt = tf / nsteps
        s = np.linspace(0, 1, nsteps)
        hd, ht = 1 - s, s

        psi = np.linalg.eigh(hd_s)[1][:, 0].astype(complex)
        for i in range(nsteps):
            psi = expm(-1j * dt * (hd[i] * hd_s + ht[i] * ht_s)) @ psi
        w, _ = model.evolve(hd, ht, dt)

        even = _even_sector(l)
        for sf in (0.3, 0.8, 1.0):
            h_ins = (1 - sf) * hd_s + sf * ht_s
            ev, vec = np.linalg.eigh(h_ins[np.ix_(even, even)])
            amp = vec.conj().T @ psi[even]
            n_lv = min(2 ** (l - 1), 64)
            E, P, _ = model.level_probabilities(w, 1 - sf, sf, n_levels=n_lv)
            dE = np.abs(np.sort(E)[:6] - ev[:6]).max()
            dP = max(
                abs(
                    P[np.isclose(E, x, atol=1e-6)].sum()
                    - (np.abs(amp[np.isclose(ev, x, atol=1e-6)]) ** 2).sum()
                )
                for x in E[:6]
            )
            # entanglement: evolving state and instantaneous GS (if non-degenerate)
            dS = max(
                abs(model.entanglement_entropy(w, blk) - _exact_entropy(psi, l, blk))
                for blk in [(0, l // 2), (1, l - 1)]
            )
            if ev[1] - ev[0] > 1e-8:
                gs = np.zeros(2**l)
                gs[even] = vec[:, 0]
                dS = max(
                    dS,
                    max(
                        abs(
                            model.eigenstate_entanglement_entropy(1 - sf, sf, blk)
                            - _exact_entropy(gs, l, blk)
                        )
                        for blk in [(0, l // 2), (1, l - 1)]
                    ),
                )
            e_res_ex = np.real(psi.conj() @ h_ins @ psi) - ev[0]
            e_res = model.residual_energy(w, 1 - sf, sf)
            print(
                f"{name:>15} s={sf:.1f} | max dE={dE:.1e}  max dP={dP:.1e}  "
                f"sumP={P.sum():.6f}  dE_res={abs(e_res - e_res_ex):.1e}  dS={dS:.1e}  "
                f"flip={model.relative_parity(model.diagonalize(1 - sf, sf)[1])}"
            )


if __name__ == "__main__":
    _selftest_statics()
    _selftest_dynamics()
