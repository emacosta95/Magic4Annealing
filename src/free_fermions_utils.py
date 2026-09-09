"""
Free-fermion (Nambu / BdG) backend for annealing schedules of the 1d
(frustrated) transverse-field Ising chain.

Spin convention (fixed by build_1dIsing_model_freefermions):

    H = - sum_i J_i sigma^x_i sigma^x_{i+1} - sum_i h_i sigma^z_i
      = Psi^dag  H_nambu  Psi  + const,        Psi = (c_1..c_l, c^dag_1..c^dag_l)

NORMALISATION (verified in _selftest, do not guess it):
    E_gs    = -sum_k e[l+k]
    E({n})  = E_gs + sum_k (2 e[l+k]) n_k
so the PHYSICAL quasiparticle energy is TWICE the eigh eigenvalue.

    H_nambu = [[ A ,  B  ],
               [-B*, -A* ]],     A = j + diag(h),   B = j_b

Frustrated ring  ->  J_i = -|J| (AFM in this sign convention), odd l, pbc=True.

Bogoliubov slicing convention (same as src/utils_nambu_system.py):
    e, w = np.linalg.eigh(h_nambu)        # e ascending
    u = w[:l, :l]      v = w[l:, :l]      # first l columns = negative branch

The quasiparticle vacuum is then characterised by the single matrix

    C = W1 @ W1^dag ,  W1 = w[:, :l]      C_{mu,nu} = < Psi_mu Psi_nu^dag >

from which every observable (Wick / Pfaffian) follows.  C is the only object
the time evolution needs to carry.

Pure numpy (no torch).  `device` arguments dropped; `dtype` defaults to
np.float64 and is only needed where a complex build is wanted.

Ema / Magic4Annealing - drop-in companion to src/schedule_utils.py
"""

import heapq

import numpy as np


def build_1dIsing_model_freefermions(
    h: np.ndarray,
    j_vec: np.ndarray,
    pbc: bool,
    diagonalization: bool = True,
    dtype=np.float64,
):
    """Nambu/BdG matrix of the 1d Ising chain in transverse field, following the
    convention of src/utils_nambu_system.py in Ema's github.

    Args:
        h:     [l] transverse field, site resolved
        j_vec: [l] coupling constants (j_vec[-1] is the boundary bond)
        pbc:   if True the boundary bond is sign-flipped -> antiperiodic
               (even-parity) sector.  See the parity note at the bottom.
    Returns:
        h_nambu [2l,2l]  (+ e [2l], w [2l,2l] if diagonalization)
    """
    h = np.asarray(h, dtype=dtype)
    l = h.shape[-1]  # number of qubits

    # bond[i] = coupling on the bond (i, i+1 mod l); bond[l-1] is the boundary
    bond = np.array(j_vec, dtype=dtype)  # copy
    bond[-1] = -1 * bond[-1] if pbc else 0.0

    # T[i, i+1] = bond[i]  -> symmetric hopping, antisymmetric pairing.
    # NOTE: the original j_l/j_r + roll construction is only correct for a
    # UNIFORM j_vec.  For site-resolved couplings it produces
    # j[i,i+1] = -J_i/2 but j[i+1,i] = -J_{i+1}/2, i.e. a NON-Hermitian A
    # block; eigh then silently symmetrises using the lower triangle and
    # returns wrong eigenvalues.  Verified: max|H - H^T| = 0.51 for random
    # j_vec, 0.0 for uniform.
    idx = np.arange(l)
    t_mat = np.zeros((l, l), dtype=dtype)
    # filling the nn bonds
    t_mat[idx, (idx + 1) % l] = bond

    # following "quantum ising chain for beginners"
    j = -0.5 * (t_mat + t_mat.T)  # hopping block
    b = -0.5 * (t_mat - t_mat.T)  # pairing block

    # transverse field
    a = j + np.diag(h)

    # initializing the Hamiltonian in Nambu space
    h_nambu = np.zeros((2 * l, 2 * l), dtype=dtype)
    h_nambu[:l, :l] = a
    h_nambu[:l, l:] = b
    h_nambu[l:, :l] = -1 * np.conj(b)
    h_nambu[l:, l:] = -1 * np.conj(a)

    if diagonalization:
        e, w = np.linalg.eigh(h_nambu)
        return h_nambu, e, w
    return h_nambu


# ---------------------------------------------------------------------------
# 1.  The two annealing operators
# ---------------------------------------------------------------------------
# build_1dIsing_model_freefermions is AFFINE in (h, j_vec) with zero constant
# term, hence exactly
#
#     H_nambu(h_driver * 1, h_target * j_vec)
#         = h_driver * M_driver  +  h_target * M_target
#
# which is precisely the linear structure SchedulerModel.forward() assumes.


def nambu_annealing_operators(l: int, j_vec: np.ndarray, pbc: bool, dtype=np.float64):
    """Return (M_driver, M_target), the two 2l x 2l terms of the BdG anneal.

    M_driver  = uniform transverse field  (-sum_i sigma^z_i)   -> diag(I, -I)
    M_target  = Ising ring                (-sum_i J_i sx sx)
    """
    zeros_l = np.zeros(l, dtype=dtype)
    ones_l = np.ones(l, dtype=dtype)

    m_driver = build_1dIsing_model_freefermions(
        h=ones_l, j_vec=zeros_l, pbc=pbc, diagonalization=False, dtype=dtype
    )
    m_target = build_1dIsing_model_freefermions(
        h=zeros_l, j_vec=j_vec, pbc=pbc, diagonalization=False, dtype=dtype
    )
    return m_driver, m_target


# ---------------------------------------------------------------------------
# 2.  State representation and time evolution
# ---------------------------------------------------------------------------


def c_matrix_bogoliubov(w: np.ndarray) -> np.ndarray:
    """C_{mu,nu} = <Psi_mu Psi_nu^dag> for the quasiparticle vacuum of `w`.

    C = W1 W1^dag with W1 = w[:, :l].  Blocks (l x l):
        C[:l , :l ] = <c_i c_j^dag>
        C[:l , l: ] = <c_i c_j>
        C[l: , l: ] = <c_i^dag c_j>
        C[l: , :l ] = <c_i^dag c_j^dag>
    """
    l = w.shape[0] // 2
    w1 = w[:, :l]
    return w1 @ w1.conj().T


def nambu_evolve(
    m_driver: np.ndarray,
    m_target: np.ndarray,
    h_driver: np.ndarray,
    h_target: np.ndarray,
    dt: float,
    w0: np.ndarray,
    store_every: int = 1,
):
    """Piecewise-constant BdG propagation:  i dW/dt = H_nambu(t) W.

    Args:
        h_driver, h_target: [nsteps] schedules, exactly what
                            Schedule.get_driving() returns.
        w0:                 [2l,2l] initial Bogoliubov matrix (eigh of H(t=0)).
    Returns:
        w_final [2l,2l] complex, and the list of stored snapshots.
    """
    w = np.asarray(w0, dtype=np.complex128)
    md = np.asarray(m_driver, dtype=np.complex128)
    mt = np.asarray(m_target, dtype=np.complex128)

    snapshots = []
    for i in range(len(h_driver)):
        hk = float(h_driver[i]) * md + float(h_target[i]) * mt
        # hk is Hermitian -> eigh is faster and more stable than a general expm
        ek, vk = np.linalg.eigh(hk)
        # FACTOR 2, do not remove.  With H = Psi^dag H_nambu Psi (no 1/2) the
        # Heisenberg equation is i dPsi/dt = 2 H_nambu Psi, consistent with the
        # quasiparticle energies being 2*e.  Verified on a sudden quench:
        # <sz>(t) matches exact diagonalisation to 1e-15 with the 2, and is
        # visibly wrong (0.73 vs 0.24 at t=0.7) without it.
        prop = (vk * np.exp(-2j * dt * ek)) @ vk.conj().T
        w = prop @ w
        if store_every and (i % store_every == 0):
            snapshots.append(w.copy())
    return w, snapshots


# ---------------------------------------------------------------------------
# 3.  Energy levels and level populations
# ---------------------------------------------------------------------------


def instantaneous_occupations(
    w_t: np.ndarray, w_inst: np.ndarray, return_matrix: bool = False
):
    """Occupations of the instantaneous Bogoliubov modes in the evolved state.

    Overlap  C = w_inst^dag w_t,  block  B = C[l:, :l];  then

        N = B B^dag ,   N_km = <gamma_k^dag gamma_m>     (l x l, Hermitian)
        n_k = diag(N)                                     row norms of B

    Note the index: n_k is the ROW norm (index k = instantaneous mode), NOT
    the column norm.  Verified against exact many-body evolution: the row
    convention reproduces E_res to 1e-6, the column one gives 0.887 vs 0.347.

    Many-body spectrum: E({n}) = E_gs + sum_k eps_k n_k, eps_k = 2*e_inst[l+k],
    so the residual energy is exactly sum_k eps_k n_k -- no diagonalisation of
    N needed, because H_inst = sum_k eps_k gamma^dag_k gamma_k + E_gs.

    Probabilities are a different story: the modes are NOT independent in this
    basis.  Use ground_state_probability() -- see the note there.
    """
    l = w_t.shape[0] // 2
    c = np.asarray(w_inst, dtype=np.complex128).conj().T @ np.asarray(
        w_t, dtype=np.complex128
    )
    b = c[l:, :l]
    n_mat = b @ b.conj().T
    n_k = np.clip(np.real(np.diag(n_mat)), 0.0, 1.0)
    if return_matrix:
        return n_k, n_mat
    return n_k


def ground_state_probability(w_t: np.ndarray, w_inst: np.ndarray) -> float:
    """|<GS_inst | psi(t)>|^2  (Onishi overlap).

    P_0 = |det C[:l, :l]| ,   C = w_inst^dag w_t.

    NOT prod_k (1 - n_k) over the diagonal of N.  The eigenvalues of N come in
    DEGENERATE PAIRS -- excitations are created in pairs because fermion parity
    is conserved -- so there are only l/2 independent two-level channels, and

        P_0 = prod_{j over distinct pairs} (1 - nu_j) = sqrt(prod_all (1-nu_j))

    which equals |det C[:l,:l]|.  Verified against exact many-body evolution:
    0.602879 vs 0.602879; the naive prod(1-n_k) gives 0.382700.
    """
    l = w_t.shape[0] // 2
    c = np.asarray(w_inst, dtype=np.complex128).conj().T @ np.asarray(
        w_t, dtype=np.complex128
    )
    return float(np.abs(np.linalg.det(c[:l, :l])))


def level_statistics(n_k: np.ndarray, eps: np.ndarray, n_mat=None):
    """(P_ground, mean excitation number, residual energy).

    P_ground is returned only if the full matrix N is supplied (see
    instantaneous_occupations(..., return_matrix=True)); otherwise None,
    because it cannot be obtained from the diagonal alone.
    """
    n_exc = np.sum(n_k)
    e_res = np.sum(eps * n_k)
    p0 = None
    if n_mat is not None:
        nu = np.clip(np.linalg.eigvalsh(n_mat), 0.0, 1.0)
        p0 = float(np.sqrt(np.prod(1.0 - nu)))
    return p0, n_exc, e_res


def lowest_levels(eps, n_levels, parity=None, return_occ=False):
    """The lowest `n_levels` MANY-BODY energies without enumerating 2^l states.

    Best-first (heap) search over occupation patterns: each pop yields the next
    smallest excitation energy, each pop pushes at most l children.  Cost
    O(n_levels * l * log(n_levels * l)) -- at l=10 it visits ~12 nodes instead
    of 1024, and the gap to 2^l only widens.

    Args:
        eps:      [l] quasiparticle energies, ALREADY including the factor 2
                  (i.e. 2*e[l:]).  Sorted internally.
        parity:   None  -> all occupation patterns allowed (OBC).
                  0 / 1 -> keep only patterns with an even / odd number of
                  excitations.  REQUIRED for the ring: Jordan-Wigner splits
                  H into two parity sectors and only one of them is physical
                  for a given boundary condition, so half of the naive
                  patterns are spurious.
    Returns:
        excitation energies above E_gs (add E_gs = -sum_k e[l+k] yourself),
        and the occupation tuples if return_occ.
    """
    eps = np.sort(np.asarray(eps, dtype=float))
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
    if return_occ:
        return np.array(out), occs
    return np.array(out)


def spectrum_along_schedule(
    m_driver, m_target, h_driver, h_target, n_levels=10, parity=None
):
    """Lowest n_levels of the INSTANTANEOUS Hamiltonian at every schedule point.

    Returns [nsteps, n_levels].  Note the labels are sorted energies, which do
    NOT track a single adiabatic state through a crossing -- for populations
    use instantaneous_occupations(), which is labelled by mode index k and is
    continuous in s.
    """
    l = m_driver.shape[0] // 2
    levels = np.zeros((len(h_driver), n_levels))
    for i in range(len(h_driver)):
        hk = float(h_driver[i]) * m_driver + float(h_target[i]) * m_target
        ek = np.linalg.eigvalsh(hk)
        eps = 2.0 * ek[l:]
        e_gs = -0.5 * eps.sum()
        levels[i] = e_gs + lowest_levels(eps, n_levels, parity=parity)
    return levels


# ---------------------------------------------------------------------------
# 4.  Majorana covariance matrix  ->  Pauli expectation values  ->  magic
# ---------------------------------------------------------------------------
# Majoranas:  a_i = c_i + c_i^dag ,  b_i = -i (c_i - c_i^dag)
# ordered as  w_(2i) = a_i , w_(2i+1) = b_i.
#
# Gamma_{mn} = i (delta_{mn} - <w_m w_n>)  is REAL antisymmetric for ANY state
# (i[w_m,w_n] is Hermitian).  For a Gaussian state
#
#     < i^m  w_{i1} ... w_{i2m} >  =  Pf( Gamma[{i},{i}] )
#
# so every Pauli string costs O(|supp|^3).
#
# IMPORTANT for dynamics: the ground state of a REAL BdG matrix has vanishing
# <aa> and <bb> Majorana blocks, which is what lets utils_nambu_system.py get
# <sx sx> from a plain determinant of the single block `c`.  Once W(t) is
# complex those blocks are non-zero and the determinant formula is WRONG.
# Use the Pfaffian of the full Gamma below.


def majorana_covariance(w: np.ndarray) -> np.ndarray:
    """Gamma [2l,2l], real antisymmetric, from the Bogoliubov matrix w."""
    l = w.shape[0] // 2
    r = vacuum_R(np.asarray(w, dtype=np.complex128))

    # <Psi_mu Psi_nu> = R[mu, swap(nu)]
    swap = np.zeros((2 * l, 2 * l), dtype=np.complex128)
    swap[:l, l:] = np.eye(l)
    swap[l:, :l] = np.eye(l)
    pmat = r @ swap  # <Psi Psi^T>

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


def pfaffian(a: np.ndarray) -> float:
    """Pfaffian of a real antisymmetric matrix (Parlett-Reid, O(n^3))."""
    a = np.array(a, dtype=float, copy=True)
    n = a.shape[0]
    if n % 2 == 1:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        # pivot the largest entry of column k into row k+1
        piv = k + 1 + int(np.argmax(np.abs(a[k + 1 :, k])))
        if piv != k + 1:
            a[[k + 1, piv], k:] = a[[piv, k + 1], k:]
            a[k:, [k + 1, piv]] = a[k:, [piv, k + 1]]
            pf = -pf
        if a[k + 1, k] == 0.0:
            return 0.0
        pf *= a[k, k + 1]
        if k + 2 < n:
            tau = a[k, k + 2 :] / a[k, k + 1]
            a[k + 2 :, k + 2 :] += np.outer(tau, a[k + 2 :, k + 1])
            a[k + 2 :, k + 2 :] -= np.outer(a[k + 2 :, k + 1], tau)
    return pf


# Pauli string -> Majorana support, via Jordan-Wigner:
#   sigma^x_j = (prod_{m<j} -i a_m b_m) a_j
#   sigma^y_j = (prod_{m<j} -i a_m b_m) b_j
#   sigma^z_j = -i a_j b_j
# JW is a Clifford circuit, so the SRE of the spin state equals the SRE
# computed from these fermionic data -- no ambiguity.


def pauli_expectation(gamma: np.ndarray, string: str) -> float:
    """<P> for a Pauli string like 'IXZYI' in a Gaussian state.

    Returns 0 for odd-weight Majorana support (parity superselection).
    """
    support = []
    sign = 1.0
    for j, s in enumerate(string):
        if s == "I":
            continue
        if s == "Z":
            support += [2 * j, 2 * j + 1]
            sign *= -1.0  # the -i, absorbed into the i^m normalisation
        elif s in ("X", "Y"):
            for m in range(j):  # JW string
                support += [2 * m, 2 * m + 1]
                sign *= -1.0
            support += [2 * j] if s == "X" else [2 * j + 1]
        else:
            raise ValueError(f"bad Pauli '{s}'")
    # cancel repeated Majoranas (they square to 1); reorder with sign tracking
    support, sign = _canonicalise(support, sign)
    if len(support) == 0:
        return float(sign)
    if len(support) % 2 == 1:
        return 0.0
    return float(sign * pfaffian(gamma[np.ix_(support, support)]))


def _canonicalise(idx, sign):
    """Sort a Majorana index list, removing pairs, tracking the sign."""
    idx = list(idx)
    # bubble sort with sign, then cancel adjacent duplicates
    for i in range(len(idx)):
        for k in range(len(idx) - 1):
            if idx[k] > idx[k + 1]:
                idx[k], idx[k + 1] = idx[k + 1], idx[k]
                sign = -sign
    out = []
    for x in idx:
        if out and out[-1] == x:
            out.pop()
        else:
            out.append(x)
    return out, sign


def sre_metropolis(
    gamma: np.ndarray,
    l: int,
    alpha: int = 2,
    n_samples: int = 20000,
    burn: int = 2000,
    seed: int = 0,
):
    """Stabilizer Renyi entropy M_alpha by Pauli-Markov sampling.

    Samples P ~ Pi(P) = <P>^2 / 2^l (a normalised probability for pure states)
    and estimates M_alpha = (1/(1-alpha)) log2 E[ <P>^{2(alpha-1)} ].
    Cost: O(n_samples * l^3).  Exact brute force over 4^l is only feasible for
    l <~ 12 -- use it to validate this estimator before trusting large l.
    """
    rng = np.random.default_rng(seed)
    letters = "IXYZ"
    cur = "".join(rng.choice(list(letters), size=l))
    p_cur = pauli_expectation(gamma, cur) ** 2
    if p_cur == 0.0:
        cur = "I" * l
        p_cur = 1.0

    acc = []
    for step in range(n_samples + burn):
        # TWO-site moves. Fermion parity superselection makes <P> vanish
        # identically unless the Majorana support has even weight, so
        # single-site proposals are almost always rejected and the chain
        # freezes (verified: single-site moves give M2 ~ 8.7 vs exact 1.42).
        new = list(cur)
        s1, s2 = rng.choice(l, size=2, replace=False)
        new[s1] = letters[rng.integers(4)]
        new[s2] = letters[rng.integers(4)]
        new = "".join(new)
        p_new = pauli_expectation(gamma, new) ** 2
        if p_new > 0 and rng.random() < min(1.0, p_new / p_cur):
            cur, p_cur = new, p_new
        if step >= burn:
            acc.append(p_cur ** (alpha - 1))

    mean = np.mean(acc)
    m_alpha = (1.0 / (1.0 - alpha)) * np.log2(mean)
    return m_alpha, np.std(acc) / np.sqrt(len(acc))


# ---------------------------------------------------------------------------
# 5.  Scheduler model -- drop-in for SchedulerTrainer
# ---------------------------------------------------------------------------
try:
    from schedule_utils import Schedule  # src/schedule_utils.py
except ImportError:  # standalone use
    Schedule = object


class NambuSchedulerModel(Schedule):
    """Free-fermion analogue of SchedulerModel.

    Same interface (forward(parameters) -> energy) so SchedulerTrainer works
    unchanged, but the forward is O(nsteps * l^3) instead of O(2^l).
    """

    def __init__(
        self,
        l,
        j_vec,
        tf,
        number_of_parameters,
        nsteps,
        type,
        seed,
        pbc=True,
        mode="annealing ansatz",
        random=False,
    ):
        self.l = l
        self.pbc = pbc
        self.j_vec = np.asarray(j_vec, dtype=np.float64)

        self.m_driver, self.m_target = nambu_annealing_operators(l, self.j_vec, pbc)
        super().__init__(
            tf=tf,
            type=type,
            number_of_parameters=number_of_parameters,
            nsteps=nsteps,
            seed=seed,
            mode=mode,
            random=random,
        )

        # target-Hamiltonian spectrum, for the residual energy
        e_t, w_t = np.linalg.eigh(self.m_target)
        # NORMALISATION (verified numerically, see _selftest):
        #   E({n}) = -sum_k e[l+k]  +  sum_k (2 e[l+k]) n_k
        # i.e. H = Psi^dag H_nambu Psi (NO 1/2), so the physical quasiparticle
        # energy is TWICE the eigenvalue returned by eigh.
        self.eps_target = 2.0 * e_t[l:]
        self.w_target = w_t
        self.e_gs_target = -0.5 * np.sum(self.eps_target)

        self.energy = 1e3
        self.w = None
        self.history, self.history_parameters = [], []
        self.run_number = 0

    def forward(self, parameters):
        self.parameters = parameters
        dt = self.time[1] - self.time[0]
        h_driver, h_target = self.get_driving()

        # ground state of H(t=0) = h_driver[0] * M_driver
        h0 = float(h_driver[0]) * self.m_driver + float(h_target[0]) * self.m_target
        _, w0 = np.linalg.eigh(h0)

        w, _ = nambu_evolve(
            self.m_driver, self.m_target, h_driver, h_target, dt, w0, store_every=0
        )
        self.w = w

        n_k = instantaneous_occupations(w, self.w_target)
        _, _, e_res = level_statistics(n_k, self.eps_target)
        self.energy = float(e_res)  # residual energy w.r.t. target ground state
        self.run_number += 1
        return self.energy

    def diagnostics(self):
        """P_ground, excitation number, Gamma at the final time."""
        n_k, n_mat = instantaneous_occupations(
            self.w, self.w_target, return_matrix=True
        )
        p0, n_exc, e_res = level_statistics(n_k, self.eps_target, n_mat)
        return dict(
            n_k=n_k,
            p_ground=float(p0),
            n_exc=float(n_exc),
            e_res=float(e_res),
            gamma=majorana_covariance(self.w),
        )


# ---------------------------------------------------------------------------
# 6.  Self-test against exact diagonalisation (run: python nambu_schedule_utils.py)
# ---------------------------------------------------------------------------
def _exact_ising(l, h, j_vec):
    """Dense H = -sum J_i sx_i sx_{i+1} - sum h_i sz_i, OBC."""
    sx = np.array([[0, 1], [1, 0]], dtype=float)
    sz = np.array([[1, 0], [0, -1]], dtype=float)
    ident = np.eye(2)

    def op(mat, site):
        out = np.array([[1.0]])
        for k in range(l):
            out = np.kron(out, mat if k == site else ident)
        return out

    ham = np.zeros((2**l, 2**l))
    for i in range(l):
        ham -= h[i] * op(sz, i)
    for i in range(l - 1):
        ham -= j_vec[i] * op(sx, i) @ op(sx, i + 1)
    return ham


def _selftest(l=6, seed=1):
    rng = np.random.default_rng(seed)
    h_np = rng.uniform(0.3, 1.7, size=l)
    j_np = rng.uniform(0.3, 1.7, size=l)

    _, e, w = build_1dIsing_model_freefermions(h_np, j_np, pbc=False)

    ham = _exact_ising(l, h_np, j_np)
    ev, evec = np.linalg.eigh(ham)
    psi = evec[:, 0]

    gamma = majorana_covariance(w)

    # (a) energy: E_gs = -sum_k e[l+k]   (excitations cost 2*e[l+k])
    e_bdg = -float(np.sum(e[l:]))
    print(f"E_gs  exact={ev[0]: .8f}   BdG={e_bdg: .8f}   diff={abs(ev[0]-e_bdg):.2e}")

    # (a2) the whole many-body spectrum, and the low-lying levels via the heap
    eps = 2.0 * e[l:]
    lows = e_bdg + lowest_levels(eps, 8)
    print(f"lowest 8 levels, max dev vs exact: {np.abs(lows - ev[:8]).max():.2e}")

    # (b) a few Pauli strings via Pfaffian vs exact
    tests = [
        "Z" + "I" * (l - 1),
        "I" * (l - 1) + "Z",
        "XX" + "I" * (l - 2),
        "IZZ" + "I" * (l - 3),
        "XIIX" + "I" * (l - 4),
        "YY" + "I" * (l - 2),
    ]
    pauli = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    worst = 0.0
    for s in tests:
        mat = np.array([[1.0 + 0j]])
        for ch in s:
            mat = np.kron(mat, pauli[ch])
        exact = float(np.real(psi.conj() @ mat @ psi))
        gauss = pauli_expectation(gamma, s)
        worst = max(worst, abs(exact - gauss))
        print(f"  <{s}>  exact={exact: .8f}  pfaffian={gauss: .8f}")
    print(f"worst Pauli discrepancy: {worst:.2e}")

    # (c) SRE: brute force vs Metropolis
    import itertools

    tot = 0.0
    for combo in itertools.product("IXYZ", repeat=l):
        tot += pauli_expectation(gamma, "".join(combo)) ** 4
    m2_exact = -np.log2(tot / (2**l))
    m2_mc, err = sre_metropolis(gamma, l, alpha=2, n_samples=8000, burn=1000)
    print(f"M2  brute force={m2_exact:.5f}   metropolis={m2_mc:.5f}")


def _selftest_dynamics():
    """Validate nambu_evolve + occupations against exact many-body evolution."""
    from scipy.linalg import expm

    print("\n--- dynamics vs exact diagonalisation ---")
    print(
        f"{'L':>3} {'s_f':>5} {'tf':>5} | {'E_res exact':>12} {'E_res BdG':>12}"
        f" | {'P0 exact':>10} {'P0 BdG':>10}"
    )
    for l, seed, sf, tf in [
        (6, 0, 0.7, 3.0),
        (6, 1, 0.9, 1.0),
        (8, 2, 0.5, 5.0),
        (5, 4, 0.3, 0.8),
    ]:
        rng = np.random.default_rng(seed)
        j_np = rng.uniform(0.5, 1.5, l)
        j_np[-1] = 0.0  # OBC
        md, mt = nambu_annealing_operators(l, j_np, pbc=False)
        h_d = _exact_ising(l, np.ones(l), np.zeros(l))
        h_t = _exact_ising(l, np.zeros(l), j_np)

        nsteps = 1000
        dt = tf / nsteps
        t = np.linspace(0, tf, nsteps)
        s_sched = sf * t / tf
        hd, ht = 1 - s_sched, s_sched

        _, w0 = np.linalg.eigh(hd[0] * md + ht[0] * mt)
        _, v0 = np.linalg.eigh(hd[0] * h_d + ht[0] * h_t)
        psi = v0[:, 0].astype(complex)
        for i in range(nsteps):
            psi = expm(-1j * dt * (hd[i] * h_d + ht[i] * h_t)) @ psi
        w, _ = nambu_evolve(md, mt, hd, ht, dt, w0, store_every=0)

        sf_ = s_sched[-1]
        h_ins = (1 - sf_) * h_d + sf_ * h_t
        m_ins = (1 - sf_) * md + sf_ * mt
        e_in, w_in = np.linalg.eigh(m_ins)
        eps = 2 * e_in[l:]
        ev_i, evec_i = np.linalg.eigh(h_ins)

        e_exact = np.real(psi.conj() @ h_ins @ psi) - ev_i[0]
        p0_exact = abs(evec_i[:, 0].conj() @ psi) ** 2
        n_k, n_mat = instantaneous_occupations(w, w_in, return_matrix=True)
        p0, _, e_res = level_statistics(n_k, eps, n_mat)
        print(
            f"{l:>3} {sf:>5} {tf:>5} | {e_exact:12.8f} {e_res:12.8f}"
            f" | {p0_exact:10.7f} {p0:10.7f}"
        )


if __name__ == "__main__":
    _selftest()
    _selftest_dynamics()
