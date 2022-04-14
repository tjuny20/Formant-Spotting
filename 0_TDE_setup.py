
import os
import numpy as np
from tools import formant_utils
import time
from tools.model import TDE
from tools.TDE_weights import getAllToAllTdeWights2
import nengo
start_time = time.time()

'''
PARAMETERS:
'''
# Formants extraction params
N_channels = 32
max_freq = 4000
n_formants = 3
thresh_frac = 0.001
dt = 0.001

# TDEs params
w_fac = 50000
tau_fac = 0.008
w_trig = 50000
tau_trig = 0.002
scale_w = 0.
max_dist = 4

# Plots TDE
plots_tde = False
n_fac1 = 1
n_trig1 = 2
n_fac2 = 7
n_trig2 = 8

# Params
word = 'one'
N_words = 5

'''
main program
'''
# Setting the TDE layer
w_fac, w_trig, tde_on = getAllToAllTdeWights2(N_channels, w_fac_value=w_fac, w_trig_value=w_trig,
                                             tau_fac=tau_fac, tau_trig=tau_trig, scale_w=scale_w, max_dist=max_dist)
TDESolver = TDE(N_channels, w_fac, w_trig, tau_fac=tau_fac, tau_trig=tau_trig, soma_type=nengo.LIF())

# Variables to store spikes
inputs_tr = []
outputs_tr = []

# load files
path = os.path.join('database/waves/800hz', word)
files = os.listdir(path)[0:N_words]

for file in files:
    # Load formant spikes
    m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), N_channels, max_freq, thresh_frac,
                                           n_formants=n_formants)

    # Save formant spikecounts
    counts = np.sum(m, axis=0)
    inputs_tr.append(counts)

    # Kind of clipping
    formant_utils.nasty_clipping(m)

    formant_utils.plot_matrix(m)

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

    formant_utils.plot_matrix(m)

end_time = time.time()

print('Time:', end_time - start_time)