import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.annealing_utils import (
    get_driver_hamiltonian,
    get_longitudinal_hamiltonian,
    computational_basis,
)
from src.schedule_utils import SchedulerModel, Schedule
from ManyBodyQutip.qutip_class import SpinOperator
from src.utils import Sector
from scipy.sparse.linalg import eigsh, expm_multiply
from src.jax_utils import SREJax
from src.utils import EntanglementEntropy
from tqdm import trange
from src.jax_utils import JaxSchedulerModel, JaxTrainer
import pickle

#### create istances on networkx
nqubits = 6
basis = computational_basis(nqubits)
jij = np.roll(np.eye(nqubits), axis=1, shift=1) + np.roll(
    np.eye(nqubits), axis=1, shift=-1
)


#### Hamiltonians
nqubits = jij.shape[0]
PS = Sector(nqubits=nqubits)
target_hamiltonian = get_longitudinal_hamiltonian(jij)
target_hamiltonian_s = PS.project(target_hamiltonian)
driver_hamiltonian = get_driver_hamiltonian(nqubits=nqubits)
driver_hamiltonian_s = PS.project(driver_hamiltonian)

#### time evolution parameters
nlevels = 10
number_parameters = 3
type = "F-CRAB"
taus = np.linspace(0.1, 30, 10)


# build once — reuse for all states
sre = SREJax(n_qubits=nqubits, batch_size=4096)
entanglement_entropy = EntanglementEntropy(nqubits=nqubits, n_A=nqubits // 2)


# ── initial state ─────────────────────────────────────────────────────────────
dim = 2**nqubits
psi_init = np.ones(dim, dtype=complex) / np.sqrt(dim)
psi_init = PS.project(psi_init)


spectrums = []
energies = []
probabilities_list = []
psi_histories = []
eigenstates_histories = []
magic_list = []
entanglement_entropy_history_quench_list = []

histories = []
histories_drivings_driver = []
histories_drivings_target = []


for tau in taus:

    time_steps = int(100 * tau)
    time = np.linspace(0, tau, time_steps)
    delta_t = time[1] - time[0]

    magic_quench = np.zeros(time_steps)
    magic_gs_level = np.zeros(time_steps)
    entanglement_entropy_history_quench = np.zeros(time_steps)
    entanglement_entropy_gs_level_history = np.zeros(time_steps)
    eigenstates_prev = None
    psi = psi_init.copy()

    model = JaxSchedulerModel(
        initial_state=psi_init,
        target_hamiltonian=target_hamiltonian_s,
        initial_hamiltonian=driver_hamiltonian_s,
        reference_hamiltonian=target_hamiltonian_s,
        tf=tau,
        nsteps=time_steps,
        number_of_parameters=number_parameters,
        type=type,
        seed=42,
        mode="annealing ansatz",
        random=False,
    )

    trainer = JaxTrainer(
        model, maxiter=500, tol=1e-3, ftol=1e-5, gtol=1e-4, verbose=True
    )
    results = trainer.run()

    h_driver = results["h_driver"]
    h_target = results["h_target"]
    psi_final = results["psi"]
    energy = results["energy"]  # already in physical units

    # ── time evolution ────────────────────────────────────────────────────────────
    # optimal
    h_driver, h_target = model.get_driving()

    histories.append(model.history)
    histories_drivings_driver.append(
        [model.history_drivings[i][0] for i in range(len(model.history_drivings))]
    )
    histories_drivings_target.append(
        [model.history_drivings[i][1] for i in range(len(model.history_drivings))]
    )

    # initialization
    psi = psi_init.copy()
    psi_history_optimalcontrol = np.zeros(
        (time_steps, psi_init.shape[0]), dtype=complex
    )
    for i, t in enumerate(time):

        hamiltonian_t = (
            h_driver[i] * driver_hamiltonian_s + target_hamiltonian_s * h_target[i]
        )
        psi = expm_multiply(-1j * delta_t * hamiltonian_t, psi)
        psi_history_optimalcontrol[i] = psi
        magic_quench[i] = sre(PS.lift(psi))
        entanglement_entropy_history_quench[i] = entanglement_entropy.von_neumann(
            PS.lift(psi)
        )

    magic_list.append(magic_quench)
    entanglement_entropy_history_quench_list.append(entanglement_entropy_history_quench)

    with open(
        f"magic_annealing_results_qubit_{nqubits}_optimal_control.pkl", "wb"
    ) as f:
        pickle.dump(
            {
                "time": time,
                "tau_qs": taus,
                "spectrums": spectrums,
                "energies": energies,
                "probabilities_list": probabilities_list,
                "psi_histories": psi_histories,
                "eigenstates_histories": eigenstates_histories,
                "magic_list": magic_list,
                "entanglement_entropy_history_quench_list": entanglement_entropy_history_quench_list,
                "history": histories,
                "history_drivings_driver": histories_drivings_driver,
                "history_drivings_target": histories_drivings_target,
                "hamiltonian_driver": driver_hamiltonian_s,
                "hamiltonian_target": target_hamiltonian_s,
                "psi_init": psi_init,
            },
            f,
        )
