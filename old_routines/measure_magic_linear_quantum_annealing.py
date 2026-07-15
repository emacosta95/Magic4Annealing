import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.annealing_utils import get_driver_hamiltonian,get_longitudinal_hamiltonian,computational_basis
from src.schedule_utils import SchedulerModel,Schedule
from ManyBodyQutip.qutip_class import SpinOperator 
from src.utils import Sector
from scipy.sparse.linalg import eigsh, expm_multiply
from src.jax_utils import SREJax
from src.utils import EntanglementEntropy
from tqdm import trange


#### create istances on networkx
nqubits = 6
basis=computational_basis(nqubits)
jij = np.roll(np.eye(nqubits),axis=1,shift=1)+np.roll(np.eye(nqubits),axis=1,shift=-1)


#### Hamiltonians
nqubits=jij.shape[0]
PS=Sector(nqubits=nqubits)
target_hamiltonian=get_longitudinal_hamiltonian(jij)
target_hamiltonian_s=PS.project(target_hamiltonian)
driver_hamiltonian=get_driver_hamiltonian(nqubits=nqubits)
driver_hamiltonian_s=PS.project(driver_hamiltonian)

#### time evolution parameters
nlevels    = 10
tau        = 30

time_steps = int(20 * tau)
time       = np.linspace(0, tau, time_steps)
delta_t    = time[1] - time[0]

tau_qs=np.linspace(0.1*tau,tau,10)

# build once — reuse for all states
sre = SREJax(n_qubits=nqubits, batch_size=4096)
entanglement_entropy = EntanglementEntropy(nqubits=nqubits, n_A=nqubits//2)


# ── initial state ─────────────────────────────────────────────────────────────
dim      = 2 ** nqubits
psi_init = np.ones(dim, dtype=complex) / np.sqrt(dim)
psi_init=PS.project(psi_init)


spectrums = []
energies = []
probabilities_list = []
psi_histories = []
eigenstates_histories = []
magic_list= []
magic_gs_level_list = []
entanglement_entropy_history_quench_list = []
entanglement_entropy_gs_level_history_list = []

for tau_q in tau_qs:


    # ── initialization ────────────────────────────────────────────────────────────
    spectrum            = np.zeros((time_steps, nlevels))
    energy              = np.zeros(time_steps)
    probabilities       = np.zeros((time_steps, nlevels))
    psi_history          = np.zeros((time_steps, dim//2), dtype=complex)
    eigenstates_history = np.zeros((time_steps, dim//2, nlevels), dtype=complex)
    
    magic_quench = np.zeros(time_steps)
    magic_gs_level = np.zeros(time_steps)
    entanglement_entropy_history_quench = np.zeros(time_steps)
    entanglement_entropy_gs_level_history = np.zeros(time_steps)
    eigenstates_prev = None
    psi = psi_init.copy()

    # ── time evolution ────────────────────────────────────────────────────────────
    for i, t in enumerate(time):
        hamiltonian_t = np.max(((1 - t / tau_q),0)) * driver_hamiltonian_s +np.min(((t / tau_q),1)) * target_hamiltonian_s
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)

        # ── diagonalize ── #
        spectrum_t, eigenstates_t = eigsh(hamiltonian_t.astype(complex), which='SA', k=nlevels)
        order              = np.argsort(spectrum_t)
        spectrum[i]        = spectrum_t[order]
        eigenstates_raw    = eigenstates_t[:, order].astype(complex)  # raw — for probabilities
        eigenstates_history[i] = eigenstates_raw
        # overlap
        probabilities[i]=np.einsum('i,ia->a',psi.conj(),eigenstates_raw)*np.einsum('i,ia->a',psi.conj(),eigenstates_raw).conj()
        energy[i] = np.real(np.vdot(psi, hamiltonian_t @ psi))
        psi_history[i] = psi
        magic_quench[i] = sre(PS.lift(psi))
        magic_gs_level[i] = sre(PS.lift(eigenstates_raw[:, 0]))
        entanglement_entropy_history_quench[i] = entanglement_entropy.von_neumann(PS.lift(psi))
        entanglement_entropy_gs_level_history[i] = entanglement_entropy.von_neumann(PS.lift(eigenstates_raw[:, 0])) 

    spectrums.append(spectrum)
    energies.append(energy)
    probabilities_list.append(probabilities)
    psi_histories.append(psi_history)
    eigenstates_histories.append(eigenstates_history)
    magic_list.append(magic_quench)
    magic_gs_level_list.append(magic_gs_level)
    entanglement_entropy_history_quench_list.append(entanglement_entropy_history_quench)
    entanglement_entropy_gs_level_history_list.append(entanglement_entropy_gs_level_history)

    print(probabilities[-1,0])
    print(spectrum.shape)
    print(time[np.argmin(spectrum[:,2]-spectrum[:,0])],np.min(spectrum[:,2]-spectrum[:,0]))

np.savez(f'magic_annealing_results_qubit_{nqubits}_tau_{tau:.1f}.npz',time=time, tau_qs=tau_qs, spectrums=np.asarray(spectrums), energies=np.asarray(energies), probabilities_list=np.asarray(probabilities_list), psi_histories=np.asarray(psi_histories), eigenstates_histories=np.asarray(eigenstates_histories), magic_list=np.asarray(magic_list), magic_gs_level_list=np.asarray(magic_gs_level_list), entanglement_entropy_history_quench_list=np.asarray(entanglement_entropy_history_quench_list), entanglement_entropy_gs_level_history_list=np.asarray(entanglement_entropy_gs_level_history_list))