import numpy as np
from ManyBodyQutip.qutip_class import SpinOperator, SpinHamiltonian
import numpy as np
from typing import Dict
import scipy
from tqdm import trange
from typing import Optional



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

def get_longitudinal_hamiltonian(jij:np.ndarray,hz:Optional[np.ndarray]=[0.]*100):
    """Initialize the scipy form of the longitudinal hamiltonian on the quantum annealing protocol

    Args:
        jij (np.ndarray): Adjacency matrix in the dense format n_qubit X n_qubit
        hz (Optional[np.ndarray], optional): Longitudinal field hz. Defaults to [0.]*100.

    Returns:
        _type_: _description_
    """

    n_qubits=jij.shape[0]
    # lets start with the Z_A Z_B of the constrain
    hamiltonian_zz=0.
    for i in range(n_qubits):
        for j in range(i+1,n_qubits):
            hamiltonian_zz+=SpinOperator([('z',i,'z',j)],coupling=[jij[i,j]],size=n_qubits,verbose=1).qutip_op

    # then the linear terms
    hamiltonian_z=0.
    for i in range(n_qubits):
        # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
        hamiltonian_z+=SpinOperator([('z',i)],coupling=[hz[i]],size=n_qubits,verbose=1).qutip_op

    return (hamiltonian_zz+hamiltonian_z).data.as_scipy()
    

def get_driver_hamiltonian(nqubits):
    
        # then the linear terms
    hamiltonian_x=0.
    for i in range(nqubits):
        # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
        hamiltonian_x+=SpinOperator([('x',i)],coupling=[-1],size=nqubits,verbose=1).qutip_op
    
    return hamiltonian_x.data.as_scipy()


def get_unbiased_catalyst_term(nqubits):
    
    # then the linear terms
    hamiltonian_xx=0.
    for i in range(nqubits):
        for j in range(i+1,nqubits):
            # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
            hamiltonian_xx+=SpinOperator([('x',i,'x',j)],coupling=[1/nqubits],size=nqubits,verbose=1).qutip_op
    
    return hamiltonian_xx.data.as_scipy()


def get_counteradiabatic_term(driver_hamiltonian,target_hamiltonian):
    # we compute the counterdiabatic term as i [H_target-H_driver,H_driver]
    return 1j*(target_hamiltonian-driver_hamiltonian)@driver_hamiltonian-1j*driver_hamiltonian@(target_hamiltonian-driver_hamiltonian)