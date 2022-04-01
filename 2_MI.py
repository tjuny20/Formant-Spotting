
import os
import numpy as np
from tools import formant_utils
import time
from tools.model import TDE
from tools.TDE_weights import getAllToAllTdeWights2
import nengo
from pyentropy import SortedDiscreteSystem
start_time = time.time()

'''
PARAMETERS:
'''
# Formants extraction params
N_channels = 32
max_freq = 8000
n_formants = 2
thresh_frac = 0.05
dt = 0.001

# TDEs params
w_fac = 20000
tau_fac = 0.003
w_trig = 15000
tau_trig = 0.002
scale_w = 0.
max_dist = 4

# Plots TDE
plots_tde = False
n_fac1 = 1
n_trig1 = 2
n_fac2 = 7
n_trig2 = 11

# Params training
keyword = 'one'
N_training = 100
ch_selected = 0.10

# Params testing
words_test = ['one', 'two', 'three', 'four']
N_test = 400

'''
main program
'''
# Setting the TDE layer
w_fac, w_trig, tde_on = getAllToAllTdeWights2(N_channels, w_fac_value=w_fac, w_trig_value=w_trig,
                                             tau_fac=tau_fac, tau_trig=tau_trig, scale_w=scale_w, max_dist=max_dist)
TDESolver = TDE(w_fac, w_trig, tau_fac=tau_fac, tau_trig=tau_trig, soma_type=nengo.LIF())

'''
TRAINING:
'''
# Variables to store spikes
inputs_tr = []
outputs_tr = []

# load files
path = os.path.join('waves/800hz', keyword)
files = os.listdir(path)[0:N_training]

for file in files:
    # Load formant spikes
    m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), N_channels, max_freq, thresh_frac,
                                           n_formants=n_formants)
    # Save formant spikecounts
    counts = np.sum(m, axis=0)
    inputs_tr.append(counts)

    # Kind of clipping (REWORK, I don't like it that much)
    formant_utils.nasty_clipping(m)

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

    # Save TDE spikecounts
    counts = np.sum(m, axis=0)
    outputs_tr.append(counts)

# Sum the spikes x channel (Formant and TDE response)
inputs_tr = np.vstack(inputs_tr)
inputs_tr = np.sum(inputs_tr, axis=0)
outputs_tr = np.vstack(outputs_tr)
outputs_tr = np.sum(outputs_tr, axis=0)

# Find the x% channels with more spikes
n_ch_formant = int(inputs_tr.shape[0]*ch_selected)
ind_maxspikes_formant = formant_utils.ind_most_spikes(inputs_tr, n_ch_formant)
n_ch_tde = int(outputs_tr.shape[0]*ch_selected)
ind_maxspikes_tde = formant_utils.ind_most_spikes(outputs_tr, n_ch_tde)

print('Training finished')

'''
TESTING:
'''

counts_for = []
counts_tde = []
identifier = []

for word in words_test:

    # load files
    path = os.path.join('waves/800hz', word)
    files = os.listdir(path)[N_training:N_test+N_training]

    for file in files:
        # Load formant spikes
        m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), N_channels, max_freq, thresh_frac,
                                               n_formants=n_formants)
        # Save formant spikecounts
        inputs_test = np.sum(m, axis=0)

        # Kind of clipping (REWORK, I don't like it that much)
        formant_utils.nasty_clipping(m)

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

        # Save TDE spikecounts
        outputs_test = np.sum(m, axis=0)

        # Count spikes in the selected channels in the training (Formant and TDEs)
        sum_formant = 0.
        for i in ind_maxspikes_formant:
            sum_formant += inputs_test[i]
        sum_tde = 0.
        for i in ind_maxspikes_tde:
            sum_tde += outputs_test[i]

        # Save spikecounts and tag them as 'not Keyword' (0) or 'keyword' (1)
        counts_for.append(int(sum_formant))
        counts_tde.append(int(sum_tde))
        if word == keyword:
            identifier.append(1)
        else:
            identifier.append(0)

'''
MI CALCULATION:
'''

# System formant-keyword
X_1 = counts_for
x_n_1 = 1
x_m_1 = max(counts_for) + 1
X_dims_1 = (x_n_1, x_m_1)

# System tde-keyword
X_2 = counts_tde
x_n_2 = 1
x_m_2 = max(counts_tde) + 1
X_dims_2 = (x_n_2, x_m_2)

Ym = 2
Ny = np.array([N_test, N_test*3])

# MI calculations
s1 = SortedDiscreteSystem(X_1, X_dims_1, Ym, Ny)
s1.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
inf1 = s1.I()
s1.calculate_entropies(method='pt', calc=['HX', 'HXY'])
inf2 = s1.I()
s1.calculate_entropies(method='qe', calc=['HX', 'HXY'])
inf3 = s1.I()
#s1.calculate_entropies(method='nsb', calc=['HX', 'HXY'])
#inf4 = s1.I()

s2 = SortedDiscreteSystem(X_2, X_dims_2, Ym, Ny)
s2.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
inf1_2 = s2.I()
s2.calculate_entropies(method='pt', calc=['HX', 'HXY'])
inf2_2 = s2.I()
s2.calculate_entropies(method='qe', calc=['HX', 'HXY'])
inf3_2 = s2.I()
#s2.calculate_entropies(method='nsb', calc=['HX', 'HXY'])
#inf4_2 = s2.I()

print('MI formant-KW (plugin): {}'.format(inf1))
print('MI formant-KW (pt): {}'.format(inf2))
print('MI formant-KW (qe): {}'.format(inf3))
# print('MI cochlea-stimulus (nsb): {}'.format(inf4))
print('MI TDE-KW (plugin): {}'.format(inf1_2))
print('MI TDE-KW (pt): {}'.format(inf2_2))
print('MI TDE-KW (qe): {}'.format(inf3_2))
# print('MI TDE-stimulus (nsb): {}'.format(inf4_2))


end_time = time.time()
print('Time:', end_time - start_time)