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
w_fac = 50000
tau_fac = 0.008
w_trig = 50000
tau_trig = 0.002
scale_w = 0.
max_dist = 3

# Params training
keyword = 'one'
N_training = 400
ch_selected = [0.01, 0.02, 0.04, 0.06, 0.1, 0.2]

# Params testing
words_test = ['one', 'seven', 'two', 'three', 'four', 'five', 'six', 'eight']
N_test = 100
factor_threshold = np.arange(-2.5, 3., 0.1)

'''
MAIN:
'''

tdes = MI_experiment(N_channels, max_freq, n_formants, threshold_formant, dt, w_fac, tau_fac, w_trig, tau_trig,
                         scale_w, max_dist)
tdes.training(keyword, N_training)
tdes.test_classifier(words_test, N_test)
TPR_for_stack = []
FPR_for_stack = []
TPR_tde_stack = []
FPR_tde_stack = []

for i in range(len(ch_selected)):
    # var to save results
    results_for = []
    results_tde = []
    TPR_for = []
    FPR_for = []
    TPR_tde = []
    FPR_tde = []
    ACC_for = []
    ACC_tde = []
    save_for = 'results/3_classifier/confusion_matrices/formant_{}.csv'.format(ch_selected[i])
    save_tde = 'results/3_classifier/confusion_matrices/tde_{}.csv'.format(ch_selected[i])
    savefig_1 = 'results/3_classifier/figs/roc_{}.png'.format(ch_selected[i])
    savefig_2 = 'results/3_classifier/figs/ch_comparison_tde.png'
    savefig_3 = 'results/3_classifier/figs/ch_comparison_for.png'

    for val in factor_threshold:
        tdes.statistics(ch_selected[i])
        result_for, result_tde = tdes.classifier(val)
        results_for.append(result_for)
        TPR_for.append(result_for[8])
        FPR_for.append(result_for[9])
        ACC_for.append(result_for[10])
        results_tde.append(result_tde)
        TPR_tde.append(result_tde[8])
        FPR_tde.append(result_tde[9])
        ACC_tde.append(result_tde[10])

    results_for = np.vstack(results_for)
    results_tde = np.vstack(results_tde)
    np.savetxt(save_for, results_for, delimiter=",")
    np.savetxt(save_tde, results_tde, delimiter=",")

    TPR_for_stack.append(TPR_for)
    TPR_tde_stack.append(TPR_tde)
    FPR_for_stack.append(FPR_for)
    FPR_tde_stack.append(FPR_tde)

    plt.figure()
    plt.plot(FPR_for, TPR_for, label='Formant', color='tab:blue')
    plt.plot(FPR_tde, TPR_tde, label='TDEs', color='tab:orange')
    plt.plot([0, 1], [0, 1], '--', label='Random', color='r')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(savefig_1)

TPR_for_stack = np.vstack(TPR_for_stack)
TPR_tde_stack = np.vstack(TPR_tde_stack)
FPR_for_stack = np.vstack(FPR_for_stack)
FPR_tde_stack = np.vstack(FPR_tde_stack)
percentages = []
for value in ch_selected:
    percentages.append(value*100.)

plt.figure()
for i in range(TPR_tde_stack.shape[0]):
    plt.plot(FPR_tde_stack[i], TPR_tde_stack[i], label='{}% channels'.format(percentages[i]))
plt.plot([0, 1], [0, 1], '--', label='Random', color='r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.savefig(savefig_2)

plt.figure()
for i in range(TPR_for_stack.shape[0]):
    plt.plot(FPR_for_stack[i], TPR_for_stack[i], label='{}% channels'.format(percentages[i]))
plt.plot([0, 1], [0, 1], '--', label='Random', color='r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.savefig(savefig_3)



end_time = time.time()
print('Time:', end_time - start_time)
