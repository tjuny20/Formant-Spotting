import numpy as np
import time
from tools.model import TDE
from tools.TDE_weights import getAllToAllTdeWights2
import nengo
start_time = time.time()

'''
PARAMETERS:
'''
# Test params
N_channels = 4
dt = 0.001
dist1 = 20
dist2 = 25

# TDEs params
w_fac = 50000
tau_fac = 0.008
w_trig = 50000
tau_trig = 0.002
scale_w = 0.
max_dist = 3

# Plots TDE
plots_tde = True
n_fac1 = 0
n_trig1 = 1
n_fac2 = 2
n_trig2 = 3

'''
main program
'''
# Setting the TDE layer
w_fac, w_trig, tde_on = getAllToAllTdeWights2(N_channels, w_fac_value=w_fac, w_trig_value=w_trig,
                                             tau_fac=tau_fac, tau_trig=tau_trig, scale_w=scale_w, max_dist=max_dist)
TDESolver = TDE(N_channels, w_fac, w_trig, tau_fac=tau_fac, tau_trig=tau_trig, soma_type=nengo.LIF())


'''
TRAINING:
'''
# Variables to store spikes
inputs_tr = []
outputs_tr = []

# Create custom matrix to play with the spikes
m = np.zeros((30,4))
m[0,0] = 1
m[dist1,1] = 1
m[0,2] = 1
m[dist2,3] = 1

#formant_utils.plot_matrix(m)

# TDEs response
m = TDESolver.run(m, n_fac1, n_trig1, n_fac2, n_trig2, plots_tde=plots_tde,
                  input_dt=dt, output_dt=dt)
m = m / 1000

# Delete tdes off
count = 0
for i in range(N_channels ** 2):
    if (i not in tde_on):
        m = np.delete(m, i - count, 1)
        count += 1

#formant_utils.plot_matrix(m)

end_time = time.time()

print('Time:', end_time - start_time)