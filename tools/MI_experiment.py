import os
import numpy as np
from tools import formant_utils
from model import TDE
from tools.TDE_weights import getAllToAllTdeWights2
import nengo
from pyentropy import SortedDiscreteSystem

class MI_experiment(object):

    def __init__(self, N_channels, max_freq, n_formants, threshold_formant, dt,
                 w_fac, tau_fac, w_trig, tau_trig, scale_w, max_dist):

        # Formants extraction params
        self.N_channels = N_channels
        self.max_freq = max_freq
        self.n_formants = n_formants
        self.threshold_formant = threshold_formant
        self.dt = dt

        # TDEs params
        self.tau_fac = tau_fac
        self.tau_trig = tau_trig
        self.scale_w = scale_w
        self.max_dist = max_dist
        self.w_fac, self.w_trig, self.tde_on = getAllToAllTdeWights2(N_channels, w_fac_value=w_fac, w_trig_value=w_trig,
                                                      tau_fac=tau_fac, tau_trig=tau_trig, scale_w=scale_w,
                                                      max_dist=max_dist)
        self.TDESolver = TDE(self.N_channels, self.w_fac, self.w_trig, tau_fac=tau_fac, tau_trig=tau_trig, soma_type=nengo.LIF())

    def training(self, keyword, N_training):

        self.keyword = keyword
        self.N_training = N_training

        # Variables to store spikes
        inputs_tr = []
        outputs_tr = []

        # load files
        path = os.path.join('database/waves/800hz', keyword)
        files = os.listdir(path)[0:N_training]

        for file in files:
            # Load formant spikes
            m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), self.N_channels, self.max_freq,
                                                   self.threshold_formant, n_formants=self.n_formants)

            # Save formant spikecounts
            counts = np.sum(m, axis=0)
            inputs_tr.append(counts)

            # Kind of clipping (REWORK, I don't like it that much)
            formant_utils.nasty_clipping(m)

            # TDEs response
            m = self.TDESolver.run(m, input_dt=self.dt, output_dt=self.dt)
            m = m / 1000

            # Delete tdes off
            count = 0
            for i in range(self.N_channels ** 2):
                if (i not in self.tde_on):
                    m = np.delete(m, i - count, 1)
                    count += 1

            # Save TDE spikecounts
            counts = np.sum(m, axis=0)
            outputs_tr.append(counts)

        # Sum the spikes x channel (Formant and TDE response)
        self.inputs_tr = np.vstack(inputs_tr)
        self.inputs_tr_totalcounts = np.sum(inputs_tr, axis=0)
        self.outputs_tr = np.vstack(outputs_tr)
        self.outputs_tr_totalcounts = np.sum(outputs_tr, axis=0)

    def statistics(self, ch_selected):
        # Find the x% channels with more spikes
        n_ch_formant = int(self.inputs_tr_totalcounts.shape[0] * ch_selected)
        self.ind_maxspikes_formant = formant_utils.ind_most_spikes(self.inputs_tr_totalcounts, n_ch_formant)
        n_ch_tde = int(self.outputs_tr_totalcounts.shape[0] * ch_selected)
        self.ind_maxspikes_tde = formant_utils.ind_most_spikes(self.outputs_tr_totalcounts, n_ch_tde)

        # Calculate the mean value and STD in the spikecount of the selected channels
        counts = []
        for trial in range(self.inputs_tr.shape[0]):
            count = formant_utils.count_spikes(self.inputs_tr[trial], self.ind_maxspikes_formant)
            counts.append(count)
        self.meancounts_formant = np.sum(counts)/self.N_training
        self.std_formant = np.std(counts)

        counts = []
        for trial in range(self.outputs_tr.shape[0]):
            count = formant_utils.count_spikes(self.outputs_tr[trial], self.ind_maxspikes_tde)
            counts.append(count)
        self.meancounts_tde = np.sum(counts) / self.N_training
        self.std_tde = np.std(counts)



    def testing(self, words_test, N_test):

        self.N_test = N_test
        self.words_test = words_test
        inputs_stack = []
        outputs_stack = []

        for word in words_test:

            # load files
            path = os.path.join('database/waves/800hz', word)
            files = os.listdir(path)[self.N_training:N_test + self.N_training]

            for file in files:
                # Load formant spikes
                m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), self.N_channels, self.max_freq,
                                                       self.threshold_formant, n_formants=self.n_formants)
                # Save formant spikecounts
                input = np.sum(m, axis=0)
                inputs_stack.append(input)

                # Kind of clipping (REWORK, I don't like it that much)
                formant_utils.nasty_clipping(m)

                # TDEs response
                m = self.TDESolver.run(m, plots_tde=False, input_dt=self.dt, output_dt=self.dt)
                m = m / 1000

                # Delete tdes off
                count = 0
                for i in range(self.N_channels ** 2):
                    if (i not in self.tde_on):
                        m = np.delete(m, i - count, 1)
                        count += 1

                # Save TDE spikecounts
                output = np.sum(m, axis=0)
                outputs_stack.append(output)

        self.inputs_test = np.vstack(inputs_stack)
        self.outputs_test = np.vstack(outputs_stack)

    def MI(self):

        counts_for = []
        counts_tde = []

        for input in self.inputs_test:
            # Count spikes in the selected channels in the training (Formant and TDEs)
            sum_formant = formant_utils.count_spikes(input, self.ind_maxspikes_formant)
            # Save spikecounts(1)
            counts_for.append(int(sum_formant))

        for output in self.outputs_test:
            sum_tde = formant_utils.count_spikes(output, self.ind_maxspikes_tde)
            counts_tde.append(int(sum_tde))

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
        Ny = np.array([self.N_test, self.N_test * (len(self.words_test)-1)])

        # MI calculations
        s1 = SortedDiscreteSystem(X_1, X_dims_1, Ym, Ny)
        s1.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
        inf1 = s1.I()
        s1.calculate_entropies(method='pt', calc=['HX', 'HXY'])
        inf2 = s1.I()
        s1.calculate_entropies(method='qe', calc=['HX', 'HXY'])
        inf3 = s1.I()
        inf_formant = np.array([inf1, inf2, inf3])

        s2 = SortedDiscreteSystem(X_2, X_dims_2, Ym, Ny)
        s2.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
        inf1_2 = s2.I()
        s2.calculate_entropies(method='pt', calc=['HX', 'HXY'])
        inf2_2 = s2.I()
        s2.calculate_entropies(method='qe', calc=['HX', 'HXY'])
        inf3_2 = s2.I()
        inf_tde = np.array([inf1_2, inf2_2, inf3_2])

        mi_formant = np.sum(inf_formant) / 3.
        mi_tde = np.sum(inf_tde) / 3.
        std_formant = np.std(inf_formant)
        std_tde = np.std(inf_tde)

        return mi_formant, mi_tde, std_formant, std_tde

    def test_classifier(self, words_test, N_test):

        self.N_test = N_test
        inputs_stack = []
        outputs_stack = []

        for word in words_test:

            # load files
            path = os.path.join('database/waves/800hz', word)
            files = os.listdir(path)[self.N_training:N_test + self.N_training]

            for file in files:
                # Load formant spikes
                m = formant_utils.formant_to_spikes_v2(os.path.join(path, file), self.N_channels, self.max_freq,
                                                       self.threshold_formant, n_formants=self.n_formants)
                # Save formant spikecounts
                input = np.sum(m, axis=0)
                inputs_stack.append(input)

                # Kind of clipping
                formant_utils.nasty_clipping(m)

                # TDEs response
                m = self.TDESolver.run(m, plots_tde=False, input_dt=self.dt, output_dt=self.dt)
                m = m / 1000

                # Delete tdes off
                count = 0
                for i in range(self.N_channels ** 2):
                    if (i not in self.tde_on):
                        m = np.delete(m, i - count, 1)
                        count += 1

                # Save TDE spikecounts
                output = np.sum(m, axis=0)
                outputs_stack.append(output)

        self.inputs_test = np.vstack(inputs_stack)
        self.outputs_test = np.vstack(outputs_stack)

    def classifier(self, factor_threshold):
        counts_for = []
        counts_tde = []
        identifier = []
        prediction_for = []
        prediction_tde = []

        # Calculate threshold based on the spikecounts from the training phase
        # threshold value: mean + factor_threshold*STD (97.8% included if follows a normal distribution)

        threshold_formant = self.meancounts_formant + factor_threshold * self.std_formant
        threshold_tde = self.meancounts_tde + factor_threshold * self.std_tde

        for i in range(self.inputs_test.shape[0]):

            # Count spikes in the selected channels (Formant and TDEs)
            sum_formant = formant_utils.count_spikes(self.inputs_test[i], self.ind_maxspikes_formant)
            sum_tde = formant_utils.count_spikes(self.outputs_test[i], self.ind_maxspikes_tde)

            # Save spikecounts, tag them as 'not Keyword' (0) or 'keyword' (1) in the identifier,
            # and predicted class in prediction
            counts_for.append(int(sum_formant))
            counts_tde.append(int(sum_tde))

            if i < self.N_test:
                identifier.append(1)
            else:
                identifier.append(0)

            if sum_formant > threshold_formant:
                prediction_for.append(1)
            else:
                prediction_for.append(0)
            if sum_tde > threshold_tde:
                prediction_tde.append(1)
            else:
                prediction_tde.append(0)

        '''
        Confusion matrix:
        row: real value
        column: predicted value
        '''
        conf_matrix_formant = np.zeros((2,2))

        for i in range(len(identifier)):
            if identifier[i] == prediction_for[i] == 1:
                conf_matrix_formant[0,0] += 1.
            elif identifier[i] == prediction_for[i] == 0:
                conf_matrix_formant[1,1] += 1.
            elif identifier[i] == 1 and prediction_for[i] == 0:
                conf_matrix_formant[0,1] += 1.
            elif identifier[i] == 0 and prediction_for[i] == 1:
                conf_matrix_formant[1,0] += 1

        conf_matrix_tde = np.zeros((2, 2))

        for i in range(len(identifier)):
            if identifier[i] == prediction_tde[i] == 1:
                conf_matrix_tde[0, 0] += 1.
            elif identifier[i] == prediction_tde[i] == 0:
                conf_matrix_tde[1, 1] += 1.
            elif identifier[i] == 1 and prediction_tde[i] == 0:
                conf_matrix_tde[0, 1] += 1.
            elif identifier[i] == 0 and prediction_tde[i] == 1:
                conf_matrix_tde[1, 0] += 1

        P = np.sum(identifier)
        N = len(identifier) - P

        PP_for = np.sum(prediction_for)
        PN_for = len(prediction_for) - PP_for
        TP_for = conf_matrix_formant[0, 0]
        FP_for = conf_matrix_formant[1, 0]
        FN_for = conf_matrix_formant[0, 1]
        TN_for = conf_matrix_formant[1, 1]
        TPR_for = TP_for/P
        FPR_for = FP_for/N
        ACC_for = (TP_for + TN_for)/(P + N)

        PP_tde = np.sum(prediction_tde)
        PN_tde = len(prediction_tde) - PP_tde
        TP_tde = conf_matrix_tde[0, 0]
        FP_tde = conf_matrix_tde[1, 0]
        FN_tde = conf_matrix_tde[0, 1]
        TN_tde = conf_matrix_tde[1, 1]
        TPR_tde = TP_tde / P
        FPR_tde = FP_tde / N
        ACC_tde = (TP_tde + TN_tde) / (P + N)

        results_for = np.array([P, N, PP_for, PN_for, TP_for, FP_for, FN_for, TN_for, TPR_for, FPR_for, ACC_for])
        results_tde = np.array([P, N, PP_tde, PN_tde, TP_tde, FP_tde, FN_tde, TN_tde, TPR_tde, FPR_tde, ACC_tde])


        return  results_for, results_tde

