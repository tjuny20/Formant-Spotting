"""
0.0 - TDE WORKBENCH

This script is used to try different setups for the weights and decaying constants of the TDEs, and visually assess the
TDE response for different time intervals between the input spikes.

It simulates 2 TDEs with the parameter setup defined in the parameters (w_fac, tau_fac, w_trig, tau_trig) and show its
responses to spikes received with a time difference of dist1 and dist2 (in ms).
"""

# Some imports
import numpy as np
import time
from tools.model import TDE
from tools.TDE_weights import getAllToAllTdeWights2
import nengo

start_time = time.time()

# Parameters for creating the TDE layer in Nengo.
N_channels = 4
dt = 0.001

# Time differences between the input spikes.
dist1 = 20
dist2 = 25

# TDEs parameters
w_fac = 50000
tau_fac = 0.008
w_trig = 50000
tau_trig = 0.002
max_dist = 3

# Parameters for plotting
plots_tde = True
n_fac1 = 0
n_trig1 = 1
n_fac2 = 2
n_trig2 = 3

'''
Main program
'''
# Setting the TDE layer
w_fac, w_trig, tde_on = getAllToAllTdeWights2(N_channels, w_fac_value=w_fac, w_trig_value=w_trig,
                                              tau_fac=tau_fac, tau_trig=tau_trig, max_dist=max_dist)
TDESolver = TDE(N_channels, w_fac, w_trig, tau_fac=tau_fac, tau_trig=tau_trig, soma_type=nengo.LIF())

# Variables to store spikes
inputs_tr = []
outputs_tr = []

# Create custom matrix with the spikes separated by dist1, dist2
m = np.zeros((30, 4))
m[0, 0] = 1
m[dist1, 1] = 1
m[0, 2] = 1
m[dist2, 3] = 1

# formant_utils.plot_matrix(m)

# TDEs response
m = TDESolver.run(m, n_fac1, n_trig1, n_fac2, n_trig2, plots_tde=plots_tde,
                  input_dt=dt, output_dt=dt)
m = m / 1000

# Delete TDEs off
count = 0
for i in range(N_channels ** 2):
    if i not in tde_on:
        m = np.delete(m, i - count, 1)
        count += 1

# formant_utils.plot_matrix(m)

end_time = time.time()

print('Time:', end_time - start_time)
