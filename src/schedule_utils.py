
import numpy as np
from typing import List, Dict, Callable,Optional
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import scipy.sparse as sp

def configuration(res,energy,grad_energy):
    
    print('Optimization Success=',res.success)
    print(f'energy={energy:.5f}')
    print(f'average gradient={np.average(grad_energy):.5f} \n')
    


class Schedule: 
    def __init__(self,tf:float,type:str,number_of_parameters:int,nsteps:int,number_target_hamiltonians:int,seed:Optional[int]=None,mode:Optional[str]='annealing ansatz',random:Optional[bool]=False):
        
        self.tf=tf
        self.nsteps=nsteps
        self.type=type
        self.mode=mode
        self.number_target_hamiltonian=number_target_hamiltonians
        self.time=np.linspace(0,self.tf,nsteps)
        self.parameters=np.zeros((1+number_target_hamiltonians)*number_of_parameters)
        if self.type=='F-CRAB' or self.type=='fourier':
            self.parameters=np.zeros(2*(1+number_target_hamiltonians)*number_of_parameters)
        self.number_parameters=number_of_parameters
        if random:
            self.parameters=np.random.uniform(-2,2,size=self.parameters.shape[0])
        
        self.seed=seed
        
        if self.type=='F-CRAB':
            dim=self.number_parameters
            self.omegas=2*np.pi*np.arange(1,dim+1)*(1+np.random.uniform(-0.5,0.5,dim))/self.tf
        if self.type=='fourier':
            self.omegas=2*np.pi*np.arange(1,dim+1)*(1+np.random.uniform(-0.5,0.5,dim))/self.tf
        
        
        
    def get_driving(self)-> np.ndarray:
        
        if self.type=='power law':
            matrix_driver=(self.parameters[:self.number_parameters,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters[:self.number_parameters].shape[0]+1)[:,None])
            matrix_target=(self.parameters[self.number_parameters:2*self.number_parameters,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters[:self.number_parameters].shape[0]+1)[:,None])
            if self.number_target_hamiltonian==2:
                matrix_target_2=(self.parameters[2*self.number_parameters:,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters[:self.number_parameters].shape[0]+1)[:,None])
                
            
        if self.type=='F-CRAB':
            #np.random.seed(self.seed)
            dim=self.number_parameters
            matrix_driver=(self.parameters[:dim,None]*np.sin(self.time[None,:]*self.omegas[:,None])+self.parameters[dim:2*dim,None]*np.cos(self.time[None,:]*self.omegas[:,None]))
            matrix_target=(self.parameters[2*dim:3*dim,None]*np.sin(self.time[None,:]*self.omegas[:,None])+self.parameters[3*dim:4*dim,None]*np.cos(self.time[None,:]*self.omegas[:,None]))
            if self.number_target_hamiltonian==2:
                matrix_target_2=(self.parameters[4*dim:5*dim,None]*np.sin(self.time[None,:]*self.omegas[:,None])+self.parameters[5*dim:,None]*np.cos(self.time[None,:]*self.omegas[:,None]))
            
            
        if self.type=='fourier':
            #np.random.seed(self.seed)
            dim=self.number_parameters
            

            matrix_driver=(self.parameters[:dim,None]*np.sin(self.time[None,:]*self.omegas[:,None])+self.parameters[dim:2*dim,None]*np.cos(self.time[None,:]*self.omegas[:,None]))
            matrix_target=(self.parameters[2*dim:3*dim,None]*np.sin(self.time[None,:]*self.omegas[:,None])+self.parameters[3*dim:4*dim,None]*np.cos(self.time[None,:]*self.omegas[:,None]))
            if self.number_target_hamiltonian==2:
                matrix_target_2=(self.parameters[4*dim:5*dim,None]*np.sin(self.time[None,:]*self.omegas[:,None])+self.parameters[5*dim:,None]*np.cos(self.time[None,:]*self.omegas[:,None]))
            
            
        
        if self.mode=='annealing ansatz':
            h_driver=(1-self.time/self.tf)*(1+np.average(matrix_driver,axis=0))
            h_target=(self.time/self.tf)*(1+np.average(matrix_target,axis=0))
            if self.number_target_hamiltonian==2:
                h_target_2=(self.time/self.tf)*(1+np.average(matrix_target_2,axis=0))

        
                return h_driver,h_target,h_target_2
            
            else:
                return h_driver,h_target
            
        else:
            h_driver=(self.time/self.tf)*(1+np.average(matrix_driver,axis=0))
            h_target=(self.time/self.tf)*(1+np.average(matrix_target,axis=0))
            if self.number_target_hamiltonian==2:
                h_target_2=(self.time/self.tf)*(1+np.average(matrix_target_2,axis=0))

        
                return h_driver,h_target,h_target_2


    def load(self,parameters:np.ndarray):
        
        if parameters.shape[0]==self.parameters.shape[0]:
            self.parameters=parameters
        else:
            print('Mbare ma ri unni cachi ca ie tuttu reshapeato! \n')
            exit()
            
        
class SchedulerModel(Schedule):    
    def __init__(self,initial_state:np.ndarray,target_hamiltonian: scipy.sparse.spmatrix,initial_hamiltonian: scipy.sparse.spmatrix,reference_hamiltonian:scipy.sparse.spmatrix,tf:float,number_of_parameters:int,nsteps:int,type:str,seed:int,mode:Optional[str]='annealing ansatz',random:Optional[bool]=False):
        
        
        
        self.target_hamiltonian=target_hamiltonian
        self.initial_hamiltonian=initial_hamiltonian
        self.reference_hamiltonian=reference_hamiltonian
        self.initial_state=initial_state

        
        super().__init__(tf=tf,type=type, number_of_parameters=number_of_parameters,nsteps=nsteps,seed=seed,mode=mode,number_target_hamiltonians=self.number_target_hamiltonian,random=random)
        
        
        self.energy=1000
        self.psi=None
        
        #### memory
        self.history=[]
        self.history_psi=[]
        self.history_drivings=[]
        self.history_parameters=[]
        self.history_run=[]
        
        self.run_number=0
        
    def forward(self,parameters):
        psi=self.initial_state
        dt=self.time[1]-self.time[0]
        self.parameters=parameters
        
        
        
        for i,t in enumerate(self.time):
            time_hamiltonian=0.
            
            hamiltonians=[self.initial_hamiltonian,self.target_hamiltonian]

            for r,driver in enumerate(self.get_driving()):
                time_hamiltonian+=driver[i]*hamiltonians[r]
            psi=expm_multiply(-1j*dt*time_hamiltonian,psi)

        self.energy=psi.conjugate().transpose().dot(self.reference_hamiltonian.dot(psi))
        self.psi=psi
        #every time we run a simulation
        self.run_number+=1
        
        return self.energy.real
    
    def callback(self,*args):
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters)
        self.history_drivings.append(self.get_driving())
        self.history_psi.append(self.psi)
        self.history_run.append(self.run_number)

        print(self.energy)
        
        
    def depolarization_option(self,noise_option:bool,noise_coupling:float):
        
        self.depolarization_coupling=noise_coupling
        self.noise_option=noise_option