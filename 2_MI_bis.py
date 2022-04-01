
import numpy as np
from matplotlib import pyplot as plt
import time
from tools.MI_experiment import MI_experiment

start_time = time.time()

'''
PARAMETERS:
'''

# Formants extraction params
N_channels = 32
max_freq = 4000
n_formants = 4
threshold_formant = 0.05
dt = 0.001

# TDEs params
configs = [33]
w_fac = [50000]
tau_fac = [0.008]
w_trig = [50000]
tau_trig = [0.002]
scale_w = 0.
max_dist = 3

# Params training
keyword = 'four'
N_training = 10
ch_selected = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Params testing (Put the keyword first!)
words_test = ['four', 'three', 'two', 'one']
N_test = 5

# Params plot
errorbars = True

'''
MAIN:
'''

for i in range(len(configs)):
    # var to save results
    results = []
    savefig = 'results/2_MI/figs/tdeconfig_{}.png'.format(configs[i])
    save_file = 'results/2_MI/csv/tdeconfig_{}.csv'.format(configs[i])
    tdes = MI_experiment(N_channels, max_freq, n_formants, threshold_formant, dt, w_fac[i], tau_fac[i], w_trig[i],
                         tau_trig[i],
                         scale_w, max_dist)
    tdes.training(keyword, N_training)
    tdes.testing(words_test, N_test)

    for val in ch_selected:
        tdes.statistics(val)
        MI = tdes.MI()
        results.append(MI)

    results = np.vstack(results)
    np.savetxt(save_file, results, delimiter=",")

    percentages = []
    for value in ch_selected:
        percentages.append(value*100.)

    plt.figure()
    plt.plot(percentages, results[:, 0], label='Formant', color='tab:blue')
    plt.plot(percentages, results[:, 1], label='TDEs', color='tab:orange')
    if errorbars:
        plt.errorbar(percentages, results[:, 0], yerr=results[:, 2], capsize=3)
        plt.errorbar(percentages, results[:, 1], yerr=results[:, 3], capsize=3)
    plt.xlabel('% channels')
    plt.ylabel('MI (bits)')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(savefig)

end_time = time.time()
print('Time:', end_time - start_time)
