import numpy as np
import os
import nengo
import thorns as th
import pandas
import matplotlib.pyplot as plt
from All_to_All_TDE_wights import getAllToAllTdeWights2
import formant_utils
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance)


class TDE(object):
    """
    Nengo model for the TDE (tau_fac and tau_trig not flexible).
    """

    def __init__(self, N_channels, w_fac, w_trig,
                 tau_fac=0.01,
                 tau_trig=0.005,
                 soma_type=nengo.LIF(),
                 ):
        self.N_channels = N_channels
        assert w_fac.shape == w_trig.shape
        N_neurons, N_input = w_fac.shape
        self.model = nengo.Network()
        with self.model:
            self.input = nengo.Node(None, size_in=N_input)
            self.dendrite = nengo.Node(lambda t, x: x[:N_neurons] * x[N_neurons:], size_in=N_neurons * 2)
            self.fac_sinapse = nengo.Connection(self.input, self.dendrite[:N_neurons], transform=w_fac, synapse=tau_fac)
            nengo.Connection(self.input, self.dendrite[N_neurons:], transform=w_trig, synapse=None)

            self.neurons = nengo.Ensemble(n_neurons=N_neurons, dimensions=1,
                                          neuron_type=soma_type,
                                          gain=np.ones(N_neurons), bias=np.zeros(N_neurons))

            nengo.Connection(self.dendrite, self.neurons.neurons, synapse=tau_trig)

            self.output = nengo.Node(None, size_in=N_neurons)
            nengo.Connection(self.neurons.neurons, self.output, synapse=None)

    def run(self, input, n_fac1=0, n_trig1=0, n_fac2=0,  n_trig2=0, plots_tde=False, input_dt=0.001, output_dt=0.001, simulator=nengo.Simulator, progress_bar=True):
        model = nengo.Network()
        model.networks.append(self.model)
        test = []
        with model:


            stim = nengo.Node(nengo.processes.PresentInput(input, presentation_time=input_dt))
            nengo.Connection(stim, self.input, synapse=None)

            out = nengo.Probe(self.output, sample_every=output_dt)

            fac1 = nengo.Probe(self.input[n_fac1], sample_every=output_dt)
            fac_v = nengo.Probe(self.fac_sinapse, attr='output', sample_every=output_dt)
            trig1 = nengo.Probe(self.input[n_trig1], sample_every=output_dt)

            p11 = nengo.Probe(self.neurons.neurons[self.N_channels*n_fac1+n_trig1], attr='input', sample_every=output_dt)
            p21 = nengo.Probe(self.neurons.neurons[self.N_channels*n_fac1+n_trig1], attr='voltage', sample_every=output_dt)
            p31 = nengo.Probe(self.neurons.neurons[self.N_channels*n_fac1+n_trig1], attr='output', sample_every=output_dt)

            fac2 = nengo.Probe(self.input[n_fac2], sample_every=output_dt)
            trig2 = nengo.Probe(self.input[n_trig2], sample_every=output_dt)

            p12 = nengo.Probe(self.neurons.neurons[self.N_channels * n_fac2 + n_trig2], attr='input', sample_every=output_dt)
            p22 = nengo.Probe(self.neurons.neurons[self.N_channels * n_fac2 + n_trig2], attr='voltage', sample_every=output_dt)
            p32 = nengo.Probe(self.neurons.neurons[self.N_channels * n_fac2 + n_trig2], attr='output', sample_every=output_dt)

        sim = simulator(model)
        with sim:
            sim.run(len(input) * input_dt)

        if plots_tde==True:
            t = sim.trange()

            plt.figure()
            plt.title('')

            plt.subplot(6, 1, 1)
            plot_spikes(t, sim.data[fac1])
            plt.xlim(0, t[-1])
            plt.xticks([])
            #plt.yticks([])
            plt.ylabel('Facilitatory')

            plt.subplot(6, 1, 2)
            plt.plot(t, sim.data[fac_v][:, self.N_channels * n_fac1])
            plt.xlim(0, t[-1])
            plt.xticks([])
            #plt.yticks([])
            plt.ylabel('Fac_V')

            plt.subplot(6, 1, 3)
            plot_spikes(t, sim.data[trig1])
            plt.xlim(0, t[-1])
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Trigger')

            plt.subplot(6, 1, 4)
            plt.plot(t, sim.data[p11])
            plt.xlim(0, t[-1])
            plt.xticks([])
            #plt.yticks([])
            plt.ylabel('Input')

            plt.subplot(6, 1, 5)
            plt.plot(t, sim.data[p21])
            plt.xlim(0, t[-1])
            plt.xticks([])
            #plt.yticks([])
            plt.ylabel('Voltage')

            plt.subplot(6, 1, 6)
            plot_spikes(t, sim.data[p31])
            plt.xlim(0, t[-1])
            plt.yticks([])
            plt.xlabel('Time(s)')
            plt.ylabel('Output')

            plt.show()

            plt.figure()
            plt.title('')

            plt.subplot(6, 1, 1)
            plot_spikes(t, sim.data[fac2])
            plt.xlim(0, t[-1])
            plt.xticks([])
            # plt.yticks([])
            plt.ylabel('Facilitatory')

            plt.subplot(6, 1, 2)
            plt.plot(t, sim.data[fac_v][:, self.N_channels * n_fac2])
            plt.xlim(0, t[-1])
            plt.xticks([])
            # plt.yticks([])
            plt.ylabel('Fac_V')

            plt.subplot(6, 1, 3)
            plot_spikes(t, sim.data[trig2])
            plt.xlim(0, t[-1])
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Trigger')

            plt.subplot(6, 1, 4)
            plt.plot(t, sim.data[p12])
            plt.xlim(0, t[-1])
            plt.xticks([])
            # plt.yticks([])
            plt.ylabel('Input')

            plt.subplot(6, 1, 5)
            plt.plot(t, sim.data[p22])
            plt.xlim(0, t[-1])
            plt.xticks([])
            # plt.yticks([])
            plt.ylabel('Voltage')

            plt.subplot(6, 1, 6)
            plot_spikes(t, sim.data[p32])
            plt.xlim(0, t[-1])
            plt.yticks([])
            plt.xlabel('Time(s)')
            plt.ylabel('Output')

            plt.show()

   #         print(sim.data[fac_v][:, 32 * n_fac1])

        # for data in test:
        #   testData=sim.data[data]
        #  if max(testData)>10000:
        #     plt.plot(sim.trange(),testData)
        # plt.show()

        return sim.data[out]


class model(object):

    def __init__(self, N_words, word, N_channels=32, w_fac=1000, w_trig=1000,
                 tau_trig=0.005, tau_fac=0.005, plots=False, plots_tde=False,
                 clipping=False, n_fac1=0, n_trig1=1, n_fac2=0, n_trig2=1, save=False, scale_w=0, max_dist=2, dt=0.001):

        self.N_words = N_words
        self.words = word
        self.N_channels = N_channels
        self.w_fac = w_fac
        self.w_trig = w_trig
        self.tau_trig = tau_trig
        self.tau_fac = tau_fac
        self.plots = plots
        self.plots_tde = plots_tde
        self.clipping = clipping
        self.n_fac1 = n_fac1
        self.n_trig1 = n_trig1
        self.n_fac2 = n_fac2
        self.n_trig2 = n_trig2
        self.save = save
        self.nospike = False
        self.scale_w = scale_w
        self.max_dist = max_dist
        self.dt = dt
        # self.param('word used put nothing here', words=1.0)

    def run(self):
        #    Variables to store the spiketrains
        input = []
        output = []

        #Load of the formants
        path = os.path.join('waves/800hz', word)
        files = os.listdir(path)

        m = matlab_load.load_spikes2(self.word, self.dt)

        # This deletes simultaneous spikes in consecutive channels
        '''for row in range(m.shape[0]):
            for col in range(m.shape[1]):
                if (m[row,col] != 0) and (m[row,col+1]!=0):
                    m[row,col] = 0'''


        input.append(m)

        if self.plots:
            self.plot_matrix(m)

        # clipping consecutive values
        if self.clipping:
            for row in range(m.shape[0]):
                for col in range(m.shape[1]):
                    try:
                        if (m[row*int(0.01/self.dt), col] != 0) and (m[(row+1)*int(0.01/self.dt), col] != 0) and (m[(row+2)*int(0.01/self.dt), col] != 0):
                            m[(row+1)*int(0.01/self.dt), col] = 0
                            m[(row+2)*int(0.01/self.dt), col] = 0
                        elif (m[row*int(0.01/self.dt), col] != 0) and (m[(row+1)*int(0.01/self.dt), col] != 0):
                            m[(row+1)*int(0.01/self.dt), col] = 0
                    except IndexError:
                        print()


        #Calculate the TDE layer response if it's not in the cache
        #    Sets de values for all the weights
        w_fac, w_trig, tde_on = getAllToAllTdeWights2(self.N_channels, w_fac_value=self.w_fac, w_trig_value=self.w_trig,
                                             tau_fac=self.tau_fac, tau_trig=self.tau_trig, scale_w=self.scale_w, max_dist=self.max_dist)

        TDESolver = TDE(w_fac, w_trig, tau_fac=self.tau_fac, tau_trig=self.tau_trig, soma_type=nengo.LIF())
        m = TDESolver.run(m, self.n_fac1, self.n_trig1, self.n_fac2, self.n_trig2, plots_tde=self.plots_tde, input_dt=0.001, output_dt=0.001)

        #Delete tdes off
        count=0
        for i in range(self.N_channels**2):
            if (i not in tde_on):
                m = np.delete(m, i-count, 1)
                count+=1

        output.append(m)

        if self.plots and self.nospike == False:
            self.plot_matrix(m)
        return input, output, tde_on

    def saveData(self, trains, file_name):
        cache_path = 'TDE_spikes/{}/{}'.format(self.word, file_name)

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

        if count == 0:
            print('No spikes')
            self.nospike = True
        else:
            df = pandas.DataFrame(data, columns=['spikes', 'duration'])
            if (os.path.exists(cache_path) == False and count != 0):
                saveFile = open(cache_path, 'w+')
                saveFile.write(df.to_json())
                saveFile.close()


    def plot_matrix(self, trains):
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