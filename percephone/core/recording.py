"""
01/11/2024
Ourania Semelidou
Théo Gauvrit

Recording object classes
"""
import json
import os
import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
import matplotlib.pyplot as plt
from percephone.utils.io import read_info
from percephone.analysis.response import resp_matrice, auc_matrice, delay_matrice, peak_matrices
from percephone.analysis.mlr import mlr
from percephone.analysis.mlr_models import classic_model
from percephone.utils.io import extract_analog_from_mesc

matplotlib.use("Qt5Agg")
plt.switch_backend("Qt5Agg")


class Recording:
    def __init__(self, input_path, foldername, rois, cache):
        self.filename, inhibitory_ids, self.sf, self.genotype, self.threshold = read_info(foldername, rois)
        self.input_path = input_path
        self.matrices = {"EXC": {"Responsivity": [], "AUC": [], "Delay_onset": []},
                         "INH": {"Responsivity": [], "AUC": [], "Delay_onset": []}}

        iscell = np.load(input_path + 'iscell.npy', allow_pickle=True)
        # Create a dimension in iscell to define excitatory and inhibitory cells
        exh_inh = np.ones((len(iscell), 1))  # array to define excitatory
        exh_inh[inhibitory_ids] = 0  # correct the array to define inhibitory as 0
        is_exh_inh = np.append(iscell, exh_inh, axis=1)  # add the array in the iscell to have a 3rd column with exh/inh
        cells_list = np.concatenate(np.argwhere(is_exh_inh[:, 0]))
        excitatory_ids = np.concatenate(np.argwhere(is_exh_inh[cells_list, 2]))  # list with excitatory cells
        if os.path.exists(input_path + 'df_f_exc.npy') and cache:
            self.df_f_exc  = np.load(input_path + 'df_f_exc.npy')
            self.df_f_inh = np.load(input_path + 'df_f_inh.npy')
        else:
            self.df_f_exc = self.compute_df_f(excitatory_ids, input_path + 'df_f_exc.npy')
            self.df_f_inh = self.compute_df_f(inhibitory_ids, input_path + 'df_f_inh.npy')
        if os.path.exists(input_path + 'spks.npy'):
            spks = np.load(self.input_path + "spks.npy")
            self.spks_exc = spks[excitatory_ids]
            self.spks_inh = spks[inhibitory_ids]
        if os.path.exists(input_path + 'matrice_resp_exc.npy') and os.path.exists(input_path + 'matrice_resp_inh.npy'):
            self.matrices["EXC"]["Responsivity"] = np.load(self.input_path + "matrice_resp_exc.npy")
            self.matrices["INH"]["Responsivity"] = np.load(self.input_path + "matrice_resp_inh.npy")


    def compute_df_f(self, cell_ids, save_path):
        """
        Compute DF/F and save it as npy matrix

        Note
        ---------
        Defining The Baseline F0:
        Portera-Cailliau baseline (He et al, Goel et al):
        baseline period is the 10-s period with the lowest variation (s.d.) in ΔF/F.
        Since to get the DF/F we already need to define a baseline,
        here the Baseline Period was defined as the 10s period with the lowest variation in F_neuropil_corrected

        Parameters
        ----------
        cell_ids: list
            ids of cells for wich the df_f will be computed
        save_path : str
            folder where the df_f matrix will be saved (same as the input path)

        Returns
        -------
        df_f_percen : numpy.ndarray
            DF/F of all the cells selected
        """
        print("Calculation Delta F / F.")
        f_ = np.load(self.input_path + 'F.npy', allow_pickle=True)
        f_neu = np.load(self.input_path + 'Fneu.npy', allow_pickle=True)
        f_ = f_[cell_ids, :]
        f_neu = f_neu[cell_ids, :]
        session_n_frames = f_.shape[1]
        all_rois = f_.shape[0]

        # Correct for Neuropil contamination
        factor = 0.7
        f_nn_corrected = f_ - f_neu * factor

        # Define The Baseline F0
        # Find the 10s period (300 frames) with the lowest stdev in f_nn_corrected
        # window binning every 300 frames
        f_binned_10s = np.lib.stride_tricks.sliding_window_view(f_nn_corrected, 300, axis=1)
        # stdev for each binned 300 frames for all the cells
        f_binned_stdev = np.ndarray.std(f_binned_10s, axis=2)
        # find the indices of the smallest stdev (of the 300 frames with less variability)
        min_stdev_window_index = np.argmin(f_binned_stdev, axis=1)
        # get the values of the window that corresponds to the index with the smallest stdev for ech cell
        f_binned_10s_for_baseline = f_binned_10s[:, min_stdev_window_index, :]
        # transform the 3D array to 2D, eliminated axis checked and is correct
        f_binned_10s_for_baseline_2d = f_binned_10s_for_baseline[:, 0, :]
        # get the mean of the 300 values of each window for each cell
        f_baseline = np.mean(f_binned_10s_for_baseline_2d, axis=1)

        # Calculate the df/f
        # create a 2D array with the baseline value for each frame of f_nn_corrected
        f_baseline_2d_array = np.repeat(f_baseline, session_n_frames).reshape(all_rois, session_n_frames)
        # df/f for all cells for each frame
        df_f = (f_nn_corrected - f_baseline_2d_array) / f_baseline_2d_array
        df_f_percen = df_f * 100
        np.save(save_path, df_f_percen)
        return df_f_percen

    def responsivity(self):
        print("Calcul of repsonsivity.")
        self.matrices["EXC"]["Responsivity"] = np.array(resp_matrice(self, self.zscore_exc))
        self.matrices["INH"]["Responsivity"] = np.array(resp_matrice(self, self.zscore_inh))
        np.save(self.input_path +"matrice_resp_exc.npy", self.matrices["EXC"]["Responsivity"])
        np.save(self.input_path +"matrice_resp_inh.npy", self.matrices["INH"]["Responsivity"])

    def delay_onset_map(self):
        self.matrices["EXC"]["Delay_onset"] = delay_matrice(self, self.df_f_exc, self.stim_time,
                                                            self.matrices["EXC"]["Responsivity"])
        self.matrices["INH"]["Delay_onset"] = delay_matrice(self, self.df_f_inh, self.stim_time,
                                                            self.matrices["INH"]["Responsivity"])

    def auc(self):
        self.matrices["EXC"]["AUC"] = auc_matrice(self, self.df_f_exc, self.matrices["EXC"]["Responsivity"])
        self.matrices["INH"]["AUC"] = auc_matrice(self, self.df_f_inh, self.matrices["INH"]["Responsivity"])

    def peak_delay_amp(self):
        self.matrices["EXC"]["Peak_delay"], self.matrices["EXC"]["Peak_amplitude"] = peak_matrices(self,
                                                                                                   self.zscore_exc,
                                                                                                   self.matrices["EXC"][
                                                                                                       "Responsivity"])
        self.matrices["INH"]["Peak_delay"], self.matrices["INH"]["Peak_amplitude"] = peak_matrices(self,
                                                                                                   self.zscore_inh,
                                                                                                   self.matrices["INH"][
                                                                                                       "Responsivity"])


class RecordingStimulusOnly(Recording):
    def __init__(self, input_path, starting_trial, inhibitory_ids, sf, correction=True):
        super().__init__(input_path, inhibitory_ids, sf)
        self.analog = pd.read_csv(input_path + 'analog.txt', sep="\t")
        if os.path.exists(input_path + 'stim_ampl_time.csv'):
            print('Analog information already computed. Reading stimulus time and amplitude.')
            self.stim_time = pd.read_csv(input_path + 'stim_ampl_time.csv', usecols=['stim_time']).values.flatten()
            self.stim_ampl = pd.read_csv(input_path + 'stim_ampl_time.csv', usecols=['stim_ampl']).to_numpy().flatten()
        else:
            self.synchronization_no_iti(correction)

    def synchronization_no_iti(self, correction_shift):
        """
        Get the stimulus time and amplitude from the analog file and
        save a csv file with the stimulus amplitude and time (stimulus starting time) in ms

        Note
        -------
        Define stimulus delivery based on the analog
        Stimulus = when analog[1]>=0.184
        First find the first peak for each stimulus (stimuli every 500ms)
        Then find the stimulus onset when amplitude is >0.184

        Parameters
        -------
        correction_shift : bool
            indicate if the correction of the shift of frame should be executed

        """
        print('Obtaining time and amplitude from analog.')
        analog_trace = self.analog.iloc[:, 1].to_numpy()
        stim_peak_indx, stim_properties = ss.find_peaks(analog_trace, prominence=0.15, distance=200)
        peaks_diff = np.diff(stim_peak_indx)
        indices = np.concatenate([[True], peaks_diff > 50000])
        stim_peak_indx = stim_peak_indx[indices]
        stim_ampl_analog = analog_trace[stim_peak_indx]
        stim_ampl_pre = np.around(stim_ampl_analog, decimals=1)
        stim_ampl_sort = np.sort(np.unique(stim_ampl_pre))
        stim_ampl = np.zeros(len(stim_ampl_pre))
        convert = {4: [4, 6, 8, 10], 5: [4, 6, 8, 10, 12], 6: [2, 4, 6, 8, 10, 12], 7: [0, 2, 4, 6, 8, 10, 12]}
        for i in range(len(stim_ampl_sort)):
            stim_ampl[stim_ampl_pre == stim_ampl_sort[i]] = convert[len(stim_ampl_sort)][i]

        def stim_onset_calc(peak_index):
            time_window = 400  # 40 ms before peak
            signal_window = analog_trace[peak_index - time_window:peak_index]
            return np.argwhere(signal_window > (3 * np.std(analog_trace[peak_index - 1000:peak_index - 400]) + np.mean(
                analog_trace[peak_index - 1000:peak_index - 400])))[0][0] + peak_index - time_window  # 0.184

        stim_onset_idx = list(map(stim_onset_calc, stim_peak_indx))
        stim_onset_time = self.analog.iloc[stim_onset_idx, 0]
        self.stim_time = stim_onset_time  # TEMP 13-09-2023
        # plot the analog to test the index of stimulus onset
        fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        axs[0].plot(analog_trace)
        axs[0].plot(stim_onset_idx, analog_trace[stim_onset_idx], 'x')
        plt.show()
        stim_onsets = np.array([int((stim / 1000) * self.sf) for stim in stim_onset_time])

        def correction(idx):
            """ Perform a frames correction. 8 is the number of frames that the last stimulus will be shifted before"""
            coeff = ((8 / len(self.df_f_exc[0])) / (stim_onsets[-1] - stim_onsets[0]))
            b = coeff * stim_onsets[0]
            return int(idx - (((coeff * idx) - b) * len(self.df_f_inh[0])))

        self.stim_ampl = stim_ampl
        if correction_shift:
            self.stim_time = np.array(list(map(correction, stim_onsets))).flatten()
        else:
            self.stim_time = stim_onsets
        print(self.stim_time.shape, self.stim_ampl.shape)
        pd.DataFrame({'stim_time': self.stim_time,
                      'stim_ampl': stim_ampl}).to_csv(self.input_path + 'stim_ampl_time.csv')


class RecordingAmplDet(Recording):
    def __init__(self, input_path, starting_trial, foldername, rois, tuple_mesc=(0, 0), correction=True,
                 cache=True, analog_sf=10000):
        super().__init__(input_path, foldername, rois, cache)
        self.xls = pd.read_excel(input_path + 'bpod.xls', header=None)
        self.stim_time = []
        self.stim_ampl = []
        self.stim_durations = []
        self.reward_time = []
        self.timeout_time = []
        self.lick_time = []
        self.detected_stim = []
        self.mlr_labels_exc = {}
        self.mlr_labels_inh = {}

        with open(input_path + 'params_trial.json', "r") as read_file:
            self.json = json.load(read_file)
        if os.path.exists(input_path + 'behavior_events.json') and cache:
            print('Behavioural information already incorporated in the analog.')
            with open(input_path + 'behavior_events.json', "r") as events_file:
                events = json.load(events_file)
            self.stim_time = np.array(events["stim_time"])
            self.stim_ampl = np.array(events["stim_ampl"])
            self.reward_time = np.array(events["reward_time"])
            self.timeout_time = np.array(events["timeout_time"])
            self.detected_stim = np.array(events["detected_stim"])
            self.stim_durations = np.array(events["stim_durations"])
            self.lick_time = np.array(events["lick_time"])
        else:
            if not os.path.exists(input_path + 'analog.txt'):
                mesc_file = [file for file in os.listdir(input_path) if file.endswith(".mesc")]
                if mesc_file:
                    extract_analog_from_mesc(input_path + mesc_file, tuple_mesc, self.sf, savepath=input_path)

                else:
                    print("No analog.txt either mesc file in the folder!")
                    return
            self.analog = pd.read_csv(input_path + 'analog.txt', sep="\t", header=None)
            self.analog[0] = (self.analog[0] * 10).astype(int)
            self.synchronization_with_iti(starting_trial, analog_sf)

        self.zscore_exc = self.zscore(self.df_f_exc)
        self.zscore_inh = self.zscore(self.df_f_inh)

    def zscore(self, dff):
        """

        Parameters
        ----------
        dff: np.ndarray
            delta f over f array from inh or exc

        Returns
            zscore: np.ndarray
                zscore for one set of neurons (inh or exc)
        -------

        """
        data = np.concatenate(np.stack(
            dff[:,
            np.linspace(self.stim_time - int(0.5 * self.sf), self.stim_time, num=int(0.5 * self.sf) + 1, dtype=int)],
            axis=2))
        mean_bsl = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        zsc = np.divide(np.subtract(dff, mean_bsl[:, np.newaxis]), std[:, np.newaxis])
        return zsc

    def synchronization_with_iti(self, starting_trial, analog_s):
        """
        Update the analog file with information on stimulus time, reward time and timeout time

        Note
        -------
        Get the ITI2, stimulus and reward time from the excel
        Synchronize the ITI2 from the excel with the ITI from the analog


        Parameters
        -------
        excel file, analog file, starting trial for the recording (relative to the behavioral trials)

        analog_s analog sampling frequency
        starting_trial
        Output
        ------
        Updated analog file (csv) with the stimulus, timeout, reward time
        """
        print('Synchronization method with ITI.')

        # get the ITI2, reward, timeout and stimulus from the xls
        data_iti = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'ITI2']
        data_reward = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'Reward']
        data_timeout = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'Timeout']
        data_stimulus = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'Stimulus']

        splitted = np.split(self.xls, self.xls[self.xls[4] == 'The trial ended'].index)
        licks = [np.round(((split[2][(split[0] == 'EVENT') & (split[4] == 94)].values - 2) * analog_s).astype(float)) for
                 split in splitted]

        # get the time for ITI2, reward, timeout and stimulus
        ITI_time_xls = data_iti[2].values.astype(float)
        reward_time = data_reward[2].values.astype(float)
        timeout_time = data_timeout[2].values.astype(float)
        stimulus_time = data_stimulus[2].values.astype(float)

        # calculate the time of reward and timeout relative to ITI2
        reward_time_to_ITI = np.around((reward_time - 2) * analog_s).tolist()  # in ms
        timeout_time_to_ITI = np.around((timeout_time - 2) * analog_s).tolist()  # in ms
        stimulus_time_to_ITI = np.around((stimulus_time - 2) * analog_s).tolist()  # in ms

        self.analog.columns = ['t', 'stimulus', 't_iti', 'iti']
        self.analog['reward'] = 0
        self.analog['timeout'] = 0
        self.analog['stimulus_xls'] = 888
        self.analog['licks'] = 0
        # Get the ITI2 from the analog file, as the first "1" value in the digital input of the analog file
        index_iti_final = []
        index_iti_analog = self.analog.index[self.analog['iti'] == 1].tolist()

        for index, elem in enumerate(index_iti_analog):
            if index == 0:
                index_iti_final.append(elem)
            elif elem - index_iti_analog[index - 1] > 1:
                index_iti_final.append(elem)

        # index_iti_final.append(index_iti_analog[-1])

        # get the info for the recorded trials and align it with the excel
        reward_to_analog = []
        timeout_to_analog = []
        stimulus_to_analog = []
        ampl_recording = []
        licks_to_analog = []
        end_protocol = len(index_iti_final) + starting_trial

        # Get lists for the reward, timeout and stimulus from the xls calculations for the trials that were recorded
        for icount in range(starting_trial, len(ITI_time_xls)):
            if icount < end_protocol:
                reward_to_analog.append(reward_time_to_ITI[icount])
                timeout_to_analog.append(timeout_time_to_ITI[icount])
                stimulus_to_analog.append(stimulus_time_to_ITI[icount])
                licks_to_analog.append(licks[icount])
                ampl_recording.append(self.json[icount]["amp"])
        ampl_recording_iter = iter(ampl_recording)
        for icount_time in range(len(index_iti_final)):
            ITI_time_analog = self.analog.at[index_iti_final[icount_time], 't']
            reward_time_analog = ITI_time_analog + reward_to_analog[icount_time]
            timeout_time_analog = ITI_time_analog + timeout_to_analog[icount_time]
            stimulus_time_analog = ITI_time_analog + stimulus_to_analog[icount_time]
            licks_time_analog = ITI_time_analog + licks_to_analog[icount_time]
            index_reward = self.analog.index[self.analog['t'] == reward_time_analog].to_list()
            index_timeout = self.analog.index[self.analog['t'] == timeout_time_analog].to_list()
            index_stimulus = self.analog.index[self.analog['t'] == stimulus_time_analog].to_list()
            for lick in licks_time_analog:
                index_licks = self.analog.index[self.analog['t'] == lick].to_list()
                if len(index_licks) != 0:
                    self.analog.at[index_licks[0], 'licks'] = 4
                    self.lick_time.append(int((index_licks[0] / analog_s) * self.sf))
            if len(index_reward) != 0:
                self.analog.at[index_reward[0], 'reward'] = 2
                self.reward_time.append(int((index_reward[0] / analog_s) * self.sf))
            if len(index_timeout) != 0:
                self.analog.at[index_timeout[0], 'timeout'] = 3
                self.timeout_time.append(int((index_timeout[0] / analog_s) * self.sf))
                if len(index_stimulus) == 0:
                    timeout_trial = next(ampl_recording_iter)
            if len(index_stimulus) != 0:
                amp = next(ampl_recording_iter)
                self.analog.at[index_stimulus[0], 'stimulus_xls'] = amp
                index_stim = int((index_stimulus[0] / analog_s) * self.sf)
                self.stim_time.append(index_stim - int(((1 / self.sf) * (index_stimulus[0] / analog_s)) * (1 / 3)))
                self.stim_ampl.append(amp)
                if len(index_reward) != 0:
                    self.detected_stim.append(True)
                else:
                    self.detected_stim.append(False)
        stim_ampl = np.around(self.stim_ampl, decimals=1)
        stim_ampl_sort = np.sort(np.unique(stim_ampl))
        convert = {4: [4, 6, 8, 10], 5: [4, 6, 8, 10, 12], 6: [2, 4, 6, 8, 10, 12], 7: [0, 2, 4, 6, 8, 10, 12]}
        for i in range(len(stim_ampl_sort)):
            stim_ampl[stim_ampl == stim_ampl_sort[i]] = convert[len(stim_ampl_sort)][i]
        self.stim_ampl = stim_ampl
        self.stim_time = np.array(self.stim_time)
        self.reward_time = np.array(self.reward_time)
        self.timeout_time = np.array(self.timeout_time)
        self.detected_stim = np.array(self.detected_stim)
        self.lick_time = np.array(self.lick_time)
        # stim duration extraction
        durations = np.zeros(len(self.stim_time))
        for i, stim_t in enumerate(self.stim_time):
            diff_ar = np.absolute(self.reward_time - stim_t)
            if diff_ar[diff_ar.argmin()] >= int(0.5 * self.sf) - 1:
                durations[i] = 15
            else:
                durations[i] = diff_ar[diff_ar.argmin()]
        self.stim_durations = durations

        to_save = {"stim_time": self.stim_time.tolist(),
                   "stim_ampl": stim_ampl.tolist(),
                   "stim_durations": self.stim_durations.tolist(),
                   "reward_time": self.reward_time.tolist(),
                   "timeout_time": self.timeout_time.tolist(),
                   "detected_stim": self.detected_stim.tolist(),
                   "lick_time": self.lick_time.tolist()
                   }
        with open(self.input_path + "behavior_events.json", "w") as jsn:
            json.dump(to_save, jsn)

    def mlr(self, mlr_model, name_model):

            if os.path.exists(self.input_path + name_model + '.json'):
                print('MLR model already computed')
                with open(self.input_path + name_model + '.json', "r") as events_file:
                    events = json.load(events_file)
                    self.mlr_labels_exc= events["exc"]
                    self.mlr_labels_inh = events["inh"]

            else:

                self.mlr_labels_exc["text_labels"], self.mlr_labels_exc["n_neurons_per_label"], self.mlr_labels_exc[
                    "neuron_labels"], self.mlr_labels_exc["indices_r2"] = mlr(self.zscore_exc, mlr_model, self.sf)
                self.mlr_labels_inh["text_labels"], self.mlr_labels_inh["n_neurons_per_label"], self.mlr_labels_inh[
                    "neuron_labels"], self.mlr_labels_inh["indices_r2"] = mlr(self.zscore_inh, mlr_model, self.sf)

                def convert_to_list(obj):
                    return {key: value.tolist() for key, value in obj.items()}

                exc_list = convert_to_list(self.mlr_labels_exc)
                inh_list = convert_to_list(self.mlr_labels_inh)
                to_save_list = {"exc": exc_list, "inh": inh_list}

                with open(self.input_path + name_model + ".json", "w") as jsn:
                    json.dump(to_save_list, jsn)


if __name__ == '__main__':
    import pandas as pd
    import math
    import percephone.plts.heatmap as hm

    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
    roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
    folder = '20240329_6601_00_synchro'
    path = directory + folder + "/"

    rec = RecordingAmplDet(path, 0, folder, roi_info)
    hm.intereactive_heatmap(rec, rec.df_f_exc)

