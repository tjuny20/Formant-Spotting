import numpy as np
import sys


def getAllToAllTdeWights(inputs=[], w_fac_value=100, w_trig_value=100, n_inputs=None, tau_fac=1, tau_trig=1, scale_w=0):
    if n_inputs is None:
        n_inputs = inputs.shape[1]
    n_tde_neurons = n_inputs * n_inputs

    w_fac = np.zeros([n_tde_neurons, n_inputs])
    w_trig = np.zeros([n_tde_neurons, n_inputs])
    start = 0
    if isinstance(tau_fac, list):
        length = len(tau_fac)
    else:
        length = 0
    for column in range(0, n_inputs):
        if length == n_tde_neurons:
            for i in range(start, start + n_inputs):
                w_fac[i, column] = w_fac_value * tau_fac[i]
                #w_fac[i, column] = w_fac_value
        else:
            w_fac[start:start + n_inputs, column] = w_fac_value * tau_fac
            #w_fac[start:start + n_inputs, column] = w_fac_value
        start += n_inputs

    column = 0
    if isinstance(tau_trig, list):
        length = len(tau_trig)
    else:
        length = 0
    for row in range(0, n_tde_neurons):
        if length == n_tde_neurons:
            w_trig[row, column] = w_trig_value * tau_trig[row]
            #w_trig[row, column] = w_trig_value
        else:
            scale_factor = int(row/n_inputs)*scale_w + 1
            w_trig[row, column] = w_trig_value * tau_trig * scale_factor
            #w_trig[row, column] = w_trig_value
        column += 1
        if column == n_inputs:
            column = 0
    #AutoTDEs out
    for i in range(n_inputs):
        row = i*n_inputs + i
        column = i
        w_trig[row, column] = 0.

    return w_fac, w_trig

def getAllToAllTdeWights2(N_channels, w_fac_value=100, w_trig_value=100, n_inputs=None, tau_fac=1, tau_trig=1, scale_w=0, max_dist=2):
    '''N'''
    n_inputs = N_channels
    n_tde_neurons = n_inputs * n_inputs

    w_fac = np.zeros([n_tde_neurons, n_inputs])
    w_trig = np.zeros([n_tde_neurons, n_inputs])
    start = 0
    if isinstance(tau_fac, list):
        length = len(tau_fac)
    else:
        length = 0
    for column in range(0, n_inputs):
        if length == n_tde_neurons:
            for i in range(start, start + n_inputs):
                w_fac[i, column] = w_fac_value * tau_fac[i]
                #w_fac[i, column] = w_fac_value
        else:
            w_fac[start:start + n_inputs, column] = w_fac_value * tau_fac
            #w_fac[start:start + n_inputs, column] = w_fac_value
        start += n_inputs

    column = 0
    if isinstance(tau_trig, list):
        length = len(tau_trig)
    else:
        length = 0
    for row in range(0, n_tde_neurons):
        if length == n_tde_neurons:
            w_trig[row, column] = w_trig_value * tau_trig[row]
            #w_trig[row, column] = w_trig_value
        else:
            scale_factor = int(row/n_inputs)*scale_w + 1
            w_trig[row, column] = w_trig_value * tau_trig * scale_factor
            #w_trig[row, column] = w_trig_value
        column += 1
        if column == n_inputs:
            column = 0
    #AutoTDEs out
    for i in range(n_inputs):
        row = i*n_inputs + i
        column = i
        w_fac[row, column] = 0.
    #Max distance
    for column in range(0, n_inputs):
        for row in range(0, n_tde_neurons):
            if (row < column*(n_inputs+1)-max_dist) or (row > column*(n_inputs+1)+max_dist):
                w_fac[row, column] = 0.

    fac_list=[]
    count = 0
    for row in w_fac:
        for value in row:
            if value != 0:
                fac_list.append(count)
        count+=1

    trig_list=[]
    count = 0
    for row in w_trig:
        for value in row:
            if value != 0:
                trig_list.append(count)
        count+=1

    tde_on = []
    for i in range(n_tde_neurons):
        if (i in fac_list) and (i in trig_list):
            tde_on.append(i)


    return w_fac, w_trig, tde_on


def getLsrToMsrTdeWights(inputs=[], w_fac_value=100, w_trig_value=100, n_inputs=None):
    if n_inputs is None:
        n_inputs = inputs.shape[1]
    n_tde_neurons = int(n_inputs / 2)
    w_fac = np.zeros([n_tde_neurons, n_inputs])
    w_trig = np.zeros([n_tde_neurons, n_inputs])
    row = 0
    for column in range(0, n_inputs):
        if column % 2 == 0:
            w_trig[row, column] = w_trig_value
        else:
            w_fac[row, column] = w_fac_value
            row += 1
    return w_fac, w_trig


def getLsrToMsrTdeAndMsrToLsrWights(inputs=[], w_fac_value=100, w_trig_value=100, n_inputs=None):
    if n_inputs is None:
        n_inputs = inputs.shape[1]
    n_tde_neurons = int(n_inputs)
    w_fac = np.zeros([n_tde_neurons, n_inputs])
    w_trig = np.zeros([n_tde_neurons, n_inputs])
    column = 0
    for row in range(0, n_inputs):
        w_fac[row, column] = w_fac_value
        if column + 2 < n_inputs:
            w_trig[row, column + 2] = w_trig_value
        else:
            w_trig[row, column + 2 - n_inputs] = w_trig_value

        column += 1

    return w_fac, w_trig


def getSpecificTde(Index_fac, Index_trig, inputs=[], w_fac_value=100, w_trig_value=100, tau_fac=1, tau_trig=1):

    n_inputs = inputs.shape[1]
    n_tde_neurons = len(Index_fac)
    w_fac = np.zeros([n_tde_neurons, n_inputs])
    w_trig = np.zeros([n_tde_neurons, n_inputs])

    if not isinstance(tau_fac, list):
        for i in range(0, len(Index_fac)):
            w_fac[i, Index_fac[i]] = w_fac_value*tau_fac
    else:

        for i in range(0, len(Index_fac)):
            w_fac[i, Index_fac[i]] = w_fac_value*tau_fac[i]

    if not isinstance(tau_trig, list):
        for i in range(0, len(Index_trig)):
            w_trig[i, Index_trig[i]] = w_trig_value*tau_trig
    else:

        for i in range(0, len(Index_fac)):
            w_trig[i, Index_trig[i]] = w_trig_value*tau_trig

    return w_fac, w_trig
# w_fac, W_trig = getAllToAllTdeWights(n_inputs=3)
