import numpy as np


def frustrated_ring_jij_hz(N, J=1.0, JL=0.5, JR=0.45):
    """
    Build (jij, hz) for get_longitudinal_hamiltonian(jij, hz), which computes
        H = sum_{i<j} jij[i,j] Z_i Z_j + sum_i hz[i] Z_i     (NO extra sign)

    Target model (Cote et al. / Werner et al.):
        H_p = -sum_{j=1}^N J_j Z_j Z_{j+1}     (site N+1 == site 1, 1-indexed j)
        J_j = -J_R if j==N ; J_L if j==(N-1)/2 or (N+1)/2 ; J otherwise
        0 < J_R < J_L < J = 1

    The minus sign in H_p is folded into jij directly (get_longitudinal_hamiltonian
    applies no sign of its own), so jij[i,i+1] = -J_j.

    Ground state energy check: E0 = -(N-3)*J + J_R - 2*J_L  (verified numerically
    this session to match exactly for N=9,11,13).
    """
    assert N % 2 == 1, "N must be odd"
    jij = np.zeros((N, N))
    mid1 = (N - 1) // 2  # 1-indexed (N-1)/2
    mid2 = (N + 1) // 2  # 1-indexed (N+1)/2
    for j in range(1, N + 1):
        if j == N:
            Jj = -JR
        elif j == mid1 or j == mid2:
            Jj = JL
        else:
            Jj = J
        i0 = j - 1
        i1 = j % N
        jij[i0, i1] += -Jj
        jij[i1, i0] += -Jj
    hz = np.zeros(N)
    return jij, hz
