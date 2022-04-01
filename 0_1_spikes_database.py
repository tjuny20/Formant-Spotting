import os
import numpy as np
from tools import formant_utils
import time

start_time = time.time()

'''
PARAMETERS:
'''
# Formants extraction params
N_channels = 32
max_freq = 8000
n_formants = 4
thresh_frac = 0.05
dt = 0.001


# Params
words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
N_words = 1

'''
main program
'''
for word in words:

    # load files
    path = os.path.join('database/waves/800hz', word)
    files = os.listdir(path)[0:N_words]

    for file in files:
        # Load formant spikes
        m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), N_channels, max_freq, thresh_frac,
                                               n_formants=n_formants)
        np.savetxt("database/spikes/{}/{}".format(word, file), m, delimiter=",")


end_time = time.time()

print('Time:', end_time - start_time)