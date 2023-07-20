import numpy as np
from scipy.optimize import fsolve

class physicalParameters():
    
    r'''
    The eigenvalues and eigenvectors returned agree with
    the values from the litterature for the Paul Traps (James, 1997)
    '''
    
    def __init__(self, physical_params):
        
        Trap = physical_params.get('Trap')
        Picture = physical_params.get('Picture')
        standardPicture = physical_params.get('Standard Coefficients')
        gaussian_beam = physical_params.get('gaussian_beam')
        omega0 = physical_params.get('omega0')
        L = physical_params.get('L')
        omega_z = physical_params.get('omega_z')
        beta = physical_params.get('beta')
        Fz = physical_params.get('Fz')
        Z = physical_params.get('Z')
        J = physical_params.get('J')
        M = physical_params.get('M')
        hbar = 1.054571818 * 1e-34
        
        self.omega0 = omega0
        self.gaussian_beam = gaussian_beam
        
        # calculate the equilibrium positions in the case of the Paul Traps and store the values in X_eq
        
        def equations(U):
            _eq = []
            
            for i in range(len(U)):
                _m = np.sum([1/(U[j] - U[i])**2 for j in range(0, i)])
                _p = np.sum([1/(U[j] - U[i])**2 for j in range(i+1, len(U))])
                _eq.append(U[i] - _m + _p)

            return (_eq)
        
        X = np.linspace(-2, 2, L)
        X_eq = fsolve(equations, X)
        self.X_eq = np.array(X_eq)
        
        # matrix whose eigenvalues/eigenvectors will be calculated
        # distinguish the cases depending on the type of traps
        
        if Trap == 'Paul Traps':
            A = []
            for i in range(L):
                A.append([])
                for j in range(L):
                    if i == j:
                        a = np.sum([1/(np.abs(X_eq[i] - X_eq[n])**3) for n in range(0, i)])
                        b = np.sum([1/(np.abs(X_eq[i] - X_eq[n])**3) for n in range(i+1, L)])
                        A[i].append((omega_z**2) * (1 + 2 * (a + b)/(Z**2) ))
                    else:
                        A[i].append(-2 * (omega_z**2) / ((np.abs(X_eq[i] - X_eq[j])**3) * (Z**2)) )
            
            eigvals, eigvecs = np.linalg.eig(A)
            eigvals, eigvecs = [list(x) for x in zip(*sorted(zip(eigvals, eigvecs.T)))]
            eigvals = np.array(eigvals)
            eigvecs = np.array(eigvecs)
        
        elif Trap == 'Microtraps':
            A = []
            for i in range(L):
                A.append([])
                for j in range(L):
                    if i == j:
                        a = np.sum([1/(np.abs(i-n)**3) for n in range(0, i)])
                        b = np.sum([1/(np.abs(i-n)**3) for n in range(i+1, L)])
                        A[i].append((omega_z**2) * (1 + 2 * beta * (a + b)))
                    else:
                        A[i].append(-2 * beta * (omega_z**2) / ((np.abs(i-j)**3)) )
            
            eigvals, eigvecs = np.linalg.eig(A)
            eigvals, eigvecs = [list(x) for x in zip(*sorted(zip(eigvals, eigvecs.T)))]
            eigvals = np.array(eigvals)
            eigvecs = np.array(eigvecs)
        
        else:
            raise Exception('Wrong Trap')
        
        
        ##############
        ##############
        # PARAMETERS #
        ##############
        ##############
        
        '''
        Adding here the following parameters, 
        used for the displacement transformation
        in the collective picture:
            · HR_couplings
            · HR_const
            · M_in
        '''
        
        ## the eigenvalues of the elasticity matrix A are \mu * (omega_z**2) = omega_n**2
        
        omega_n = []
        for i in eigvals:
            omega_n.append(np.sqrt(i))
        omega_n = np.array(omega_n)
        
        
        ## COLLECTIVE PICTURE WITH DISPLACEMENT TRANSFORMATION
        
        M_in = eigvecs.T
        self.M_in = M_in
        
        # coupling added to the Hamiltonian after the displacement transformation
        
        gamma_n = []
        
        if standardPicture:
            for n in range(L):
                a = sum(M_in[:, n])
                # use normalized omega_n
                gamma_n.append((Fz**2) * a / ( (hbar*omega_n[n]/J)**2 ))
        else:
            for n in range(L):
                a = sum(M_in[:, n])
                gamma_n.append((Fz**2) * a / (M * J * (omega_n[n]**2) ))
        
        HR_couplings = []
        for i in range(L):
            b = [M_in[i, n] * gamma_n[n] for n in range(L)]
            HR_couplings.append(sum(b))
        
        self.HR_couplings = np.array(HR_couplings)
        
        # constant added to the Hamiltonian after the displacement transformation
        
        HR_const = 0
        
        if standardPicture:
            for n in range(L):
                c = sum(M_in[:, n])**2
                # use normalized omega_n
                HR_const -= (Fz**2) * c / (2 * (hbar*omega_n[n]/J)**2 )
        else:
            for n in range(L):
                c = sum(M_in[:, n])**2
                HR_const -= (Fz**2) * c / (2 * M * J * (omega_n[n]**2) )
        
        self.HR_const = HR_const
        
        
        ##########
        '''
        for the 'Standard Picture' where we enter manually unitless falues for Fz and beta,
        the following conventions are chosen:
        
        J = 1
        hbar * omegaz = J = 1
        omegaz = 1
        hbar = 1
        m = 1
        
        Then the coupling factors are calculated using the same expression
        for the effective Hamiltonian as before, only replacing the given values by 1
        '''
        ##########
        
        '''
        Adding here the following parameters,
        taking into account if it is Paul/Micro Trap
        as well as Standard Parameters or Real Parameters:
        
        COLLECTIVE PICTURE:
            · omega_n
            · g_col
        
        LOCAL PICTURE:
            · omegaz
            · g_loc
            · position_coupling
        '''
        
        # Collective Picture
        
        g = []
        for i in range(len(omega_n)):
            g.append(eigvecs[i, :] * np.sqrt(hbar / (2 * M * omega_n[i]) ))
        g = np.array(g).T/J
        
        g_standard = []
        for i in range(len(omega_n)):
            # use normalized omega_n
            g_standard.append(eigvecs[i, :] * np.sqrt(1 / (2 * (hbar*omega_n[i]/J) ) ))
        g_standard = np.array(g_standard).T
        self.g_standard = g_standard
        # normalize the omega_n  as unitless values dividing hbar*omega by J
        
        omega_n = hbar * omega_n / J
        
        # factors used for the collective picture
        
        self.omega_n = omega_n
        
        if standardPicture:
            self.g_col = Fz * g_standard
        else:
            self.g_col = Fz * g
        
        ##########
        
        # Local Picture
        
        self.omegaz = hbar * omega_z / J
        
        if standardPicture:
            
            # spin-phonon coupling
            self.g_loc = Fz * np.sqrt(1 / 2 )
            
            # bosons coupling
            if Trap == 'Paul Traps':
                self.position_coupling = 1 / (2 * (Z**2) )
            elif Trap == 'Microtraps':
                self.position_coupling = beta / 2
        
        else:
            
            # spin-phonon coupling
            self.g_loc = Fz * np.sqrt(hbar / (2 * M * omega_z) ) / J
            
            # bosons coupling
            if Trap == 'Paul Traps':
                self.position_coupling = hbar * omega_z / (2 * (Z**2) * J )
            elif Trap == 'Microtraps':
                self.position_coupling = hbar * omega_z * beta / (2 * J)
        
        ##########
        
        '''
        Returning here the right parameters depending on:
            Collective/Local
            Standard Parameters/Real Parameters
        The resulting parameters will be:
            · omega
            · g
        '''
        
        # return the right coefficients depending on the picture
        
        if Picture == 'Collective':
            self.omega = omega_n
            
            if standardPicture:
                self.g = Fz * g_standard
            else:
                self.g = Fz * g
        
        elif Picture == 'Local':
            self.omega = hbar * omega_z / J
            
            if standardPicture:
                self.g = Fz * np.sqrt(1 / 2 )
            else:
                self.g = Fz * np.sqrt(hbar / (2 * M * omega_z) ) / J
        
        else:
            raise Exception('Wrong Picture')