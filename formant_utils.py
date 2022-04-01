import numpy as np
import pandas as pd
import thorns as th
import pandas

'''
takes as inputs:
    formants (matrixes 2xN_values): 1st row formant positions(Hz), 2nd row amplitudes
    threshold: amplitude threshold to cut the formant
returns: 
    matrix(2xN_values) with the freqs of each formant. 1st row -> 1st formant, 2nd row -> 2nd formant
'''
def extract_formant(formant1, formant2, threshold):
    # create a list good_values with the indices of the values its amplitude is higher than the threshold
    good_values = []
    for i in range(formant1.shape[1]):
        if formant1[1, i] > threshold:
            good_values.append(i)

    # locate the longest chain of consecutive indices (indices of the formant)
    chain_first = []
    chain_last = []
    difs = []
    formant_pos = []
    for j in range(max(good_values) + 1):
        if (j - 1 not in good_values) and (j in good_values):
            chain_first.append(j)
        if (j in good_values) and (j + 1 not in good_values):
            chain_last.append(j)
        elif j == max(good_values):
            chain_last.append(j)

    if len(chain_first) != len(chain_last):
        print('Some error locating formant')
    else:
        for j in range(len(chain_first)):
            dif = chain_last[j] - chain_first[j]
            difs.append(dif)
        for j in range(len(difs)):
            if difs[j] == max(difs):
                max_pos = j
        for j in range(chain_first[max_pos], chain_last[max_pos] + 1):
            formant_pos.append(j)

    final_formant = []
    # cut the formant from the complete chain
    for j in range(formant1.shape[1]):
        if j in formant_pos:
            final_formant.append([formant1[0, j], formant2[0,j]])
    final_formant = np.vstack(final_formant)
    final_formant = np.transpose(final_formant)
    return final_formant

def formant_to_spikes(path, threshold, n_channels, max_freq):
    formants = pd.read_csv(path, sep=',',header=None)
    formant = []
    n_tot_formants = int(formants.values.shape[0]/2)

    try:
        for k in range(2):
            formant.append(formants.values[[k, k+n_tot_formants], :])

        try:
            formant = extract_formant(formant[0], formant[1], threshold)
            n_steps = formant.shape[1]

            spikes = np.zeros((n_steps, n_channels))

            for i in range(n_steps):
                for j in range(n_channels):
                    freq_min = (max_freq / n_channels) * j
                    freq_max = (max_freq / n_channels) * (j + 1)
                    if (freq_max > formant[0, i] >= freq_min) or \
                            (freq_max > formant[1, i] >= freq_min):
                        spikes[i, j] = 1

            return spikes
        except ValueError:
            print('ValueError:', path, "don't work and I don't know why")

    except IndexError:
        print('IndexError:', path, 'is weird')

def extract_formant_v2(formant, threshold):
    # create a list good_values with the indices of the values its amplitude is higher than the threshold
    good_values = []
    for i in range(formant[0].shape[1]):
        if formant[0][1, i] > threshold:
            good_values.append(i)

    # locate the longest chain of consecutive indices (indices of the formant)
    chain_first = []
    chain_last = []
    difs = []
    formant_pos = []

    for j in range(max(good_values) + 1):
        if (j - 1 not in good_values) and (j in good_values):
            chain_first.append(j)
        if (j in good_values) and (j + 1 not in good_values):
            chain_last.append(j)
        elif j == max(good_values):
            chain_last.append(j)

    if len(chain_first) != len(chain_last):
        print('Some error locating formant')
    else:
        for j in range(len(chain_first)):
            dif = chain_last[j] - chain_first[j]
            difs.append(dif)
        for j in range(len(difs)):
            if difs[j] == max(difs):
                max_pos = j
        for j in range(chain_first[max_pos], chain_last[max_pos] + 1):
            formant_pos.append(j)

    final_formant = []
    # cut the formant from the complete chain
    for j in range(formant[0].shape[1]):
        if j in formant_pos:
            list = []
            for i in range(len(formant)):
                list.append(formant[i][0,j])
            final_formant.append(list)
    final_formant = np.vstack(final_formant)
    final_formant = np.transpose(final_formant)
    return final_formant

def formant_to_spikes_v2(path, n_channels, max_freq, thresh_frac, n_formants=2):
    formants = pd.read_csv(path, sep=',',header=None)
    formant = []
    n_tot_formants = int(formants.values.shape[0] / 2)

    try:
        for k in range(n_formants):
            formant.append(formants.values[[k, k+n_tot_formants], :])
    except IndexError:
        print(path, 'Incorrect data format')
        return
    threshold = np.amax(formant[0][1, :])*thresh_frac

    try:
        formant = extract_formant_v2(formant, threshold)
    except ValueError:
        print(path, 'No values over threshold')
        return

    n_steps = formant.shape[1]

    spikes = np.zeros((n_steps, n_channels))
    for i in range(n_steps):
        for j in range(n_channels):
            freq_min = (max_freq / n_channels) * j
            freq_max = (max_freq / n_channels) * (j + 1)
            check = []
            for k in range(n_formants):
                check.append(freq_max > formant[k, i] >= freq_min)
            if any(check):
                spikes[i, j] = 1

    return spikes


def plot_matrix(trains):
    # convert the input from time X train
    spikeTimes = []
    duration = trains.shape[0] / 1000
    data = []
    N_spikes = trains.shape[1]
    count = 0
    for train in range(0, N_spikes):
        spikeTrain = []
        for time in range(0, trains.shape[0]):
            spike = trains[time, train]
            if spike > 0.0:
                spikeTrain.append(time / 1000)
                count += 1
        spikeTimes.append(spikeTrain)
    for row in spikeTimes:
        data.append([row, duration])
    df = pandas.DataFrame(data, columns=['spikes', 'duration'])
    th.plot_raster(df)
    th.show()

def ind_most_spikes(m, n_ind):
    indices_max = np.argpartition(m, -n_ind)[-n_ind:]
    return np.sort(indices_max)

def count_spikes(m, indices):
    count = 0
    for i in indices:
        count += m[i]
    return count

def nasty_clipping(m):
    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            try:
                if (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0) and (
                        m[(row + 10), col] != 0) and (m[(row + 11), col] != 0) and (
                        m[(row + 12), col] != 0) and (m[(row + 13), col] != 0) and (
                        m[(row + 14), col] != 0) and (m[(row + 15), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                    m[(row + 10), col] = 0
                    m[(row + 11), col] = 0
                    m[(row + 12), col] = 0
                    m[(row + 13), col] = 0
                    m[(row + 14), col] = 0
                    m[(row + 15), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0) and (
                        m[(row + 10), col] != 0) and (m[(row + 11), col] != 0) and (
                        m[(row + 12), col] != 0) and (m[(row + 13), col] != 0) and (
                        m[(row + 14), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                    m[(row + 10), col] = 0
                    m[(row + 11), col] = 0
                    m[(row + 12), col] = 0
                    m[(row + 13), col] = 0
                    m[(row + 14), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0) and (
                        m[(row + 10), col] != 0) and (m[(row + 11), col] != 0) and (
                        m[(row + 12), col] != 0) and (m[(row + 13), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                    m[(row + 10), col] = 0
                    m[(row + 11), col] = 0
                    m[(row + 12), col] = 0
                    m[(row + 13), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0) and (
                        m[(row + 10), col] != 0) and (m[(row + 11), col] != 0) and (
                        m[(row + 12), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                    m[(row + 10), col] = 0
                    m[(row + 11), col] = 0
                    m[(row + 12), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0) and (
                        m[(row + 10), col] != 0) and (m[(row + 11), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                    m[(row + 10), col] = 0
                    m[(row + 11), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0) and (
                        m[(row + 10), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                    m[(row + 10), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0) and (m[(row + 9), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                    m[(row + 9), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0) and (
                        m[(row + 8), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                    m[(row + 8), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0) and (m[(row + 7), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                    m[(row + 7), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0) and (
                        m[(row + 6), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                    m[(row + 6), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0) and (m[(row + 5), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                    m[(row + 5), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0) and (
                        m[(row + 4), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                    m[(row + 4), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0) and (m[(row + 3), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                    m[(row + 3), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0) and (
                        m[(row + 2), col] != 0):
                    m[(row + 1), col] = 0
                    m[(row + 2), col] = 0
                elif (m[row, col] != 0) and (m[(row + 1), col] != 0):
                    m[(row + 1), col] = 0
            except IndexError:
                print()
    return m

