import numpy as np

import tenpy
from tenpy.models.lattice import Site, Chain, Lattice
from tenpy.models.model import CouplingModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite, BosonSite, set_common_charges, group_sites

class CollectiveDisplacement(CouplingModel, MPOModel):
    
    def __init__(self, model_params):
        
        # read out/set default parameters
        model_params = asConfig(model_params, "CollectiveTrappedIon")
        
        Trap = model_params.get('Trap', None)
        gaussian_beam = model_params.get('gaussian_beam', False)
        omega0 = model_params.get('omega0', 5*1e-5)
        L = model_params.get('L', 2)
        Omega = model_params.get('Omega', 1.)
        g = model_params.get('g', 1.)
        X_eq = model_params.get('X_eq', np.array([i for i in range(L)]))
        Nmax = model_params.get('Nmax', 5)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        sort_charge = model_params.get('sort_charge', None)
        HR_couplings = model_params.get('HR_couplings', [i for i in range(L)])
        HR_const = model_params.get('HR_const', 1)
        
        # define the conserved charges and initialize the sites
        halfSpin = SpinHalfSite(conserve='Sz', sort_charge=sort_charge)
        boson = BosonSite(Nmax=Nmax, conserve='None')
        # common charges for both sites
        set_common_charges([halfSpin, boson], new_charges = 'independent')
        
        # add spon-phonon interaction operators
        '''
        Id = halfSpin.get_op('Id')
        Sigmaz = halfSpin.get_op('Sigmaz')
        halfSpin.add_op('IdPlusSigmaz', Id + Sigmaz)
        '''
        Bd = boson.get_op('Bd')
        B = boson.get_op('B')
        boson.add_op('BdPlusB', Bd + B)
        
        Id_B = boson.get_op('Id')
        N = boson.get_op('N')
        boson.add_op('NPlusIdHalf', N + Id_B * 1/2)
        
        # initialize a lattice with 2*L = L * (spin, boson) sites
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        unit_cell = [halfSpin, boson]
        lat = Lattice([L], unit_cell, bc=bc, bc_MPS='finite')
        
        # initialize CouplingModel
        CouplingModel.__init__(self, lat)
        
        # add terms of the Hamiltonian
        
        # on site phonons interactions
        self.add_onsite(Omega, 1, 'NPlusIdHalf')
        
        # spin-phonon interaction : (alpha + alpha_dagger) * sigmaz
        for i in range(0, 2*L, 2):
            for n in range(1, 2*L, 2):
                g_in = g[int(i/2), int((n-1)/2)]
                if i < n:
                    self.add_coupling_term(- g_in, i, n, 'Sigmaz', 'BdPlusB')
                elif n < i:
                    self.add_coupling_term(- g_in, n, i, 'BdPlusB', 'Sigmaz')
        
        # spin-spin coupling
        # differentiate the Microtraps and Paul Traps cases
        
        if Trap == 'Paul Traps':
            for i in range(0, 2*L, 2):
                for j in range(0, 2*L, 2):
                    # we have to differenciate the two following cases (i<j and j<i)
                    # because the function add_coupling_term only takes i,j as arguments if i<j
                    
                    if i != j and i < j:
                        
                        # make the difference between gaussian beam and normal J coefficients
                        gaussian_arg = - 2 * (X_eq[int(i/2)]**2 + X_eq[int(j/2)]**2) / (omega0**2)
                        if gaussian_beam:
                            _strength = np.exp(gaussian_arg) * (1/np.abs(X_eq[int(i/2)] - X_eq[int(j/2)])**3)
                        else:
                            _strength = (1/np.abs(X_eq[int(i/2)] - X_eq[int(j/2)])**3)
                        self.add_coupling_term(2 * _strength, i, j, 'Sp', 'Sm')
                        self.add_coupling_term(2 * _strength, i, j, 'Sm', 'Sp')
                    
                    elif i != j and j < i:
                        
                        gaussian_arg = - 2 * (X_eq[int(i/2)]**2 + X_eq[int(j/2)]**2) / (omega0**2)
                        if gaussian_beam:
                            _strength = np.exp(gaussian_arg) * (1/np.abs(X_eq[int(i/2)] - X_eq[int(j/2)])**3)
                        else:
                            _strength = (1/np.abs(X_eq[int(i/2)] - X_eq[int(j/2)])**3)
                        self.add_coupling_term(2 * _strength, j, i, 'Sm', 'Sp')
                        self.add_coupling_term(2 * _strength, j, i, 'Sp', 'Sm')
        
        elif Trap == 'Microtraps':
            for i in range(0, 2*L, 2):
                for j in range(0, 2*L, 2):
                    # we have to differenciate the two following cases (i<j and j<i)
                    # because the function add_coupling_term only takes i,j as arguments if i<j
                    
                    if i != j and i < j:
                        
                        # make the difference between gaussian beam and normal J coefficients
                        gaussian_arg = - (i**2 + j**2) / (2 * (omega0**2)) # we take i/2 and j/2
                        if gaussian_beam:
                            _strength = np.exp(gaussian_arg) * 8 * (1/np.abs(i-j)**3)
                        else:
                            _strength = 8 * (1/np.abs(i-j)**3)
                        self.add_coupling_term(2 * _strength, i, j, 'Sp', 'Sm')
                        self.add_coupling_term(2 * _strength, i, j, 'Sm', 'Sp')
                    
                    elif i != j and j < i:
                        
                        gaussian_arg = - (i**2 + j**2) / (2 * (omega0**2)) # we take i/2 and j/2
                        if gaussian_beam:
                            _strength = np.exp(gaussian_arg) * 8 * (1/np.abs(i-j)**3)
                        else:
                            _strength = 8 * (1/np.abs(i-j)**3)
                        self.add_coupling_term(2 * _strength, j, i, 'Sm', 'Sp')
                        self.add_coupling_term(2 * _strength, j, i, 'Sp', 'Sm')
        
        # add HR term
        self.add_onsite(- HR_couplings, 0, 'Sigmaz')
        self.constant = HR_const
        
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())