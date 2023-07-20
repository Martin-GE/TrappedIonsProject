import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from CollectiveTrappedIonClass import CollectiveTrappedIon
from CollectiveDisplacementClass import CollectiveDisplacement
from LocalTrappedIonClass import LocalTrappedIon
from PhysicalParameters import physicalParameters

import tenpy
from tenpy.networks.mps import MPS, InitialStateBuilder, build_initial_state
from tenpy.algorithms.dmrg import SingleSiteDMRGEngine, SubspaceExpansion

# DMRG parameters
dmrg_params = {
    'mixer': SubspaceExpansion, # adds perturbation to the state with the terms of the Hamiltonian
                    # which have contributions in both the “left” and “right” side of the system
    'mixer_params': {
        'amplitude': 1e-10,
        'decay': 1.0,
        'disable_after': 18,
    },
    
    'max_sweeps': 10,
    'min_sweeps': 10,
    
    'chi_list': {
        0:  50,
        2:  100,
        4:  200,
        8:  400,
        10: 600},
        
    'N_sweeps_check': 2, # number of sweeps to perform between checking convergence criteria and giving a status update
}

#############################################
#############################################
# input the parameters of the model by hand #
#############################################
#############################################

'''
if 'Standard Coefficients' is True,
then Fz and beta should be changed with the desired unitless values
but J, omegaz and M can still keep their real values
without affecting the output couplings
'''

L = 24
standard_coefficients = True
M = 24.3 * 1.66e-27
Z = 1
omega_z = 1e6

if standard_coefficients:
    Fz = 2
    beta = 10
else:    
    rabi = 1e7
    detuning = 2e7
    _lambda = 5 * 1e-7 # optical laser with a wavelength of ~ 500 nm
    d_0 = 1e-5
    
    hbar = 1.054571818 * 1e-34
    k = 2 * np.pi / _lambda
    e = 1.602176634e-19
    eps_0 = 8.8541878128e-12
    
    Fz = hbar * (rabi**2) * k / (2 * detuning)
    beta = 10 # e**2 / ( 4 * np.pi * eps_0 * M * (omega_z**2) * (d_0**3) )

physical_params = {
    'Trap': 'Paul Traps', # can only be 'Paul Traps' or 'Microtraps'
    'Picture': 'Collective', # can only be 'Local' or 'Collective'
    'Displacement': True, 
    'Standard Coefficients': standard_coefficients, # input 'standard' values for Fz and beta
    'gaussian_beam': False, # gaussian profile for the Jij coefficients
    'omega0': 5 * 1e-5, # necessary parameter for the gaussian profile
    'L': L, 
    'omega_z': omega_z, # excitation laser frequency
    'beta': beta, # beta value for Microtraps, not used for Paul Traps
    'Fz': Fz, 
    'Z': Z, # atomic number of the ions used for the experiment, not used for Microtraps
    'M': M, 
    'J': 1.054571818 * 1e-34 * 1e6, # hbar * omega_z
    'Nmax': 20, 
}

###########################
###########################
# end of parameters input #
###########################
###########################

# create outer directory where to save the files
if physical_params['Picture'] == 'Local':
    out_dir = 'Results Local/'
else:
    if physical_params['Displacement']:
        out_dir = 'Results Collective Displacement/'
    else:
        out_dir = 'Results Collective No Displacement/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# save the info regarding each sweep
tenpy.tools.misc.setup_logging(to_file="INFO", filename=out_dir+'INFO.txt')

couplings = physicalParameters(physical_params)

model_params = {
    'Trap': physical_params['Trap'], 
    'gaussian_beam': couplings.gaussian_beam, 
    'omega0': couplings.omega0, 
    'L': physical_params['L'], 
    'Fz': Fz, 
    'M': 24.3 * 1.66e-27, 
    'Omega': couplings.omega, 
    'g': couplings.g, 
    'Position Coupling': couplings.position_coupling, 
    'X_eq': couplings.X_eq, 
    'Nmax': physical_params['Nmax'], 
    'bc_MPS': 'finite',
    'sort_charge': True,
    'HR_couplings': couplings.HR_couplings, 
    'HR_const': couplings.HR_const, 
}

if physical_params['Picture'] == 'Collective':
    if physical_params['Displacement']:
        M = CollectiveDisplacement(model_params)
    else:
        M = CollectiveTrappedIon(model_params)
elif physical_params['Picture'] == 'Local':
    M = LocalTrappedIon(model_params)

# create a random spin_up/spin_down state with a given filling

spin_states = build_initial_state(L, ['up', 'down'], [0.5, 0.5], mode='random')
boson_states = [0 for i in range(L)]
product_state = []

for i in range(L):
    product_state.append([spin_states[i], boson_states[i]])

options = {
    #'check_filling': (1, 2), 
    'full_empty': ('up', 'down'), 
    'method': 'lat_product_state', 
    #'product_state': [['up', 0]] * int(L//2) + [['down', 0]] * int(L//2), 
    'product_state': product_state, 
    'randomized_from_method': 'lat_product_state'
}
psi = InitialStateBuilder(M.lat, options).run()

# run the model with the randomly generated MPS

eng = SingleSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()

# save DMRG data

chi_max = []
t = []
max_E_trunc = []
max_trunc_err = []

chi_max.append(max(eng.sweep_stats['max_chi'])) # maximum bond dimension used
t.append(max(eng.sweep_stats['time'])) # time in seconds
max_E_trunc.append(max(eng.sweep_stats['max_E_trunc'])) # maximum change on energy due to truncation
max_trunc_err.append(max(eng.sweep_stats['max_trunc_err'])) # maximum truncation error per sweep
'''
# psi.save_hdf5()

import h5py

f = h5py.File(out_dir + "MPS.hdf5", "w")

hdf5_saver = tenpy.tools.hdf5_io.Hdf5Saver(f)
psi.save_hdf5(hdf5_saver, f, 'MPS.hdf5/')

f.close()
'''
# create plot with energy as function of sweep
plt.figure(dpi=150)
ax = plt.gca()

if physical_params['Picture'] == 'Collective' and physical_params['Displacement']:
    ax.plot(eng.sweep_stats['sweep'], np.array(eng.sweep_stats['E']) + model_params['HR_const'])
else:
    eng.plot_sweep_stats(ax, 'sweep', 'E')

for s, chi in eng.options['chi_list'].items():
    ax.axvline(s, linestyle='--', color='k')

ax.set_title('Energy as a fuction of sweeps')
plt.savefig(out_dir + 'Energy(sweep).pdf')

# save the values to a file
if physical_params['Picture'] == 'Collective' and physical_params['Displacement']:
    stats = [eng.sweep_stats['sweep'], eng.sweep_stats['E'] + model_params['HR_const'], eng.sweep_stats['max_trunc_err'], eng.sweep_stats['max_E_trunc']]
else:
    stats = [eng.sweep_stats['sweep'], eng.sweep_stats['E'], eng.sweep_stats['max_trunc_err'], eng.sweep_stats['max_E_trunc']]
np.savetxt(out_dir + "stats.txt", stats)

# save the g matrix to a file
np.savetxt(out_dir + 'g.txt', couplings.g)

# compute the expectation value of (1 + sigmaz)/2
expectation_sigmaz = []
for i in range(0, 2*L, 2):
    expectation_sigmaz.append((1 + psi.expectation_value_term([('Sigmaz', i)])) / 2)
print('(1+sigmaz)/2: ', expectation_sigmaz)
np.savetxt(out_dir + "SpinDensity.txt", expectation_sigmaz)

# compute the phonon expectation value
phonon = []
for i in range(0, 2*L, 2):
    phonon.append(psi.expectation_value_term([('N', i+1)]))
print('Phonon: ', phonon)
np.savetxt(out_dir + "Phonon.txt", phonon)

# compute the spin-spin expectation value
spinSpin = []
for i in range(0, 2*L, 2):
    spinSpin_i = []
    for j in range(0, 2*L, 2):
        spinSpin_ij = float(psi.expectation_value_term([('Sigmaz', i), ('Sigmaz', j)]))
        spinSpin_i.append(spinSpin_ij)
    spinSpin.append(spinSpin_i)
print('Spin-spin: ', spinSpin)
np.savetxt(out_dir + "SpinSpin.txt", spinSpin)

# compute the CDW order parameter
cdw_list = []
for n in range(0, 2*L, 2):
    term = ((-1)**(n//2)) * (1 + psi.expectation_value_term([('Sigmaz', n)])) / (2*L)
    cdw_list.append(term)
O_CDW = np.sum(cdw_list)
print('O_CDW: ', O_CDW)
np.savetxt(out_dir + "O_CDW.txt", [O_CDW])

# compute the four-spin correlator

O_SC = []

O_SC_delta_1 = []
delta_1 = 1
for i in range(0, 2*L-2*(delta_1+1), 2):
    four_spin = [('Sp', i), ('Sp', i+2), ('Sm', i + 2*delta_1), ('Sm', i + 2*(1+delta_1))]
    O_SC_delta_1.append(float(psi.expectation_value_term(four_spin)))
O_SC.append(O_SC_delta_1)
_length = len(O_SC_delta_1)

for delta in range(2, L):
    O_SC_delta = []
    for i in range(0, 2*L-2*(delta+1), 2):
        four_spin = [('Sp', i), ('Sp', i+2), ('Sm', i + 2*delta), ('Sm', i + 2*(1+delta))]
        O_SC_delta.append(float(psi.expectation_value_term(four_spin)))
    
    if len(O_SC_delta) != 0:
        _mean = np.mean(O_SC_delta)
        while len(O_SC_delta) < _length:
            O_SC_delta.append(_mean)
        
        O_SC.append(O_SC_delta)

np.savetxt(out_dir + "O_SC.txt", O_SC)

# compute the BdPlusB expectation value
phonon = []
for i in range(0, 2*L, 2):
    phonon.append(psi.expectation_value_term([('BdPlusB', i+1)]))
print('Phonon: ', phonon)
np.savetxt(out_dir + "BdPlusB.txt", phonon)

# compute the spin phonon correlator
# /!\ important to be in the local picture, otherwise the following code doesn't give what we want
Pi = []
for i in range(0, 2*L, 2):
    Pi_i = []
    for j in range(0, 2*L, 2):
        Pi_ij = psi.expectation_value_term([('Sz', i), ('BdPlusB', j+1)]) - psi.expectation_value_term([('Sz', i)]) * psi.expectation_value_term([('BdPlusB', j+1)])
        Pi_i.append(Pi_ij)
    Pi.append(Pi_i)
print('Spin-phonon correlator Pi_ij: ', Pi)

if physical_params['Picture'] == 'Local':
    np.savetxt(out_dir + "SpinPhononCorrelator.txt", Pi)

# compute the spin phonon correlator for the Collective Picture
g = couplings.g
Si_Rj = []
for i in range(0, 2*L, 2):
    Sigma_i = []
    for j in range(0, 2*L, 2):
        Sigmai_Rj = []
        for n in range(0, 2*L, 2):
            term = g[int(j//2), int(n//2)] * psi.expectation_value_term([('Sz', i), ('BdPlusB', n+1)])
            Sigmai_Rj.append(term)
        Sigma_i.append(np.sum(Sigmai_Rj))
    Si_Rj.append(Sigma_i)
np.savetxt(out_dir + "SiRj (Collective).txt", Si_Rj)

Si_Rj_2 = []
for i in range(0, 2*L, 2):
    Sigma_i_2 = []
    for j in range(0, 2*L, 2):
        Sigmai_Rj_2 = []
        for n in range(0, 2*L, 2):
            term_2 = g[int(j//2), int(n//2)] * ( psi.expectation_value_term([('Sz', i)]) * psi.expectation_value_term([('BdPlusB', n+1)]) )
            Sigmai_Rj_2.append(term_2)
        Sigma_i_2.append(np.sum(Sigmai_Rj_2))
    Si_Rj_2.append(Sigma_i_2)
np.savetxt(out_dir + "Si_Rj (Collective).txt", Si_Rj_2)

if physical_params['Picture'] == 'Collective':
    Pi_col = np.array(Si_Rj) - np.array(Si_Rj_2)
    np.savetxt(out_dir + "SpinPhononCorrelator.txt", Pi_col)


df = pd.DataFrame({
    "Chi max": chi_max, # maximum bond dimension used
    "Time (seconds)": t, # time in seconds
    "Time (minutes)": np.array(t)/60, # time in minutes
    "max_E_trunc": max_E_trunc, # maximum change on energy due to truncation
    "max_trunc_err": max_trunc_err # maximum truncation error in all the sweeps
})

df.to_csv(out_dir + "Time.csv", index=False)
