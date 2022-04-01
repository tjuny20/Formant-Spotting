import os
from tools import formant_utils
import time

#words = ['one', 'two', 'three', 'four']
words = ['one']
n_channels = 32
max_freq = 8000
n_formants = 2
thresh_frac = 0.05

start_time = time.time()


for word in words:
    #load files
    path = os.path.join('../database/waves/800hz', word)
    files = os.listdir(path)

    for file in files[0:1]:
        # vars to store the formant tracks
        spikes = formant_utils.formant_to_spikes_v2(os.path.join(path, file), n_channels, max_freq, thresh_frac, n_formants=n_formants)
        print(spikes)


end_time = time.time()

print('Time:', end_time-start_time)

'''totals = []
for file in files[0:1]:
    # vars to store the formant tracks
    formants = pd.read_csv(os.path.join(path,file), sep=',',header=None)
    formant = []
    n_formants = int(formants.values.shape[0]/2)

    for k in range(n_formants):
        formant.append(formants.values[[k, k+n_formants], :])
    print(np.amax(formant[0][1, :]))'''



'''x_1 = np.arange(formant_1.shape[1])
x_2 = np.arange(formant_2.shape[1])

plt.plot(x_1, formant_1[0])
plt.plot(x_2, formant_2[0])
plt.show()'''

#test to know which are the 2 principal formants
'''totals = []
for file in files[0:3]:
    # vars to store the formant tracks
    formants = pd.read_csv(os.path.join(path,file), sep=',',header=None)
    formant = []
    n_formants = int(formants.values.shape[0]/2)

    for k in range(n_formants):
        formant.append(formants.values[[k, k+n_formants], :])

    sums = []
    try:
        for k in range(4):
            sums.append(np.sum(formant[k], axis=1)[1])
        totals.append(sums)

    except IndexError:
        print(file, ' corrupted')

totals = np.vstack(totals)
totals = np.sum(totals, axis=0)
print(totals)'''
















