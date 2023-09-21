"""Ourania Semelidou, 27/03/2023
Core classes for recording, synchronization, synchronization w/o ITI2 analog"""

import json
import os
import random as rnd
import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
from responsivity import responsivity, resp_single_neuron
from Helper_Functions.Utils_core import read_info
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")


class Recording:
    def __init__(self, input_path, foldername, rois):
        inhibitory_ids, self.sf = read_info(foldername, rois)
        self.input_path = input_path
        f = np.load(input_path + 'F.npy', allow_pickle=True)
        fneu = np.load(input_path + 'Fneu.npy', allow_pickle=True)
        iscell = np.load(input_path + 'iscell.npy', allow_pickle=True)

        # Create a dimension in iscell to define excitatory and inhibitory cells
        exh_inh = np.ones((len(iscell), 1))  # array to define excitatory
        exh_inh[inhibitory_ids] = 0  # correct the array to define inhibitory as 0
        is_exh_inh = np.append(iscell, exh_inh, axis=1)  # add the array in the iscell to have a 3rd column with exh/inh

        cells_list = np.concatenate(np.argwhere(is_exh_inh[:, 0]))
        excitatory_ids = np.concatenate(np.argwhere(is_exh_inh[cells_list, 2]))  # list with excitatory cells

        self.df_f_exc = self.compute_df_f(f, fneu, excitatory_ids, input_path + 'df_f_exc.npy')
        self.df_f_inh = self.compute_df_f(f, fneu, inhibitory_ids, input_path + 'df_f_inh.npy')
        if os.path.exists(input_path + 'spks.npy'):
            spks = np.load(self.input_path + "spks.npy")
            self.spks_exc = spks[excitatory_ids]
            self.spks_inh = spks[inhibitory_ids]

    def compute_df_f(self, f_, f_neu, cell_ids, save_path):
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
        f_ : numpy.ndarray (cells, frames)
            Fluorescence of all ROIs from suite2p
        f_neu : numpy.ndarray  (cells, frames)
            Fluorescence of all neuropils from suite2p
        cell_ids: list
            ids of cells for wich the df_f will be computed
        save_path : str
            folder where the df_f matrix will be saved (same as the input path)

        Returns
        -------
        df_f_percen : numpy.ndarray
            DF/F of all the cells selected
        """
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

        # save the df_f as a npy
        np.save(save_path, df_f_percen)
        return df_f_percen

    def compute_responsivity(self, row_metadata):
        resp, resp_neur = responsivity(self, row_metadata)
        return resp, resp_neur

    def resp_matrice(self, df_data):
        """ Method with interquartile measure of the baseline

        Parameters
        ----------
        df_data :  numpy array
            delta f over f (neurons,time) can be exc or inh
        """
        pre_boundary = int(0.25 * self.sf)  # index
        post_boundary = int(0.5 * self.sf)
        exclude_windows = [list(range(t, t + post_boundary)) for t in self.stim_time]
        exclude_windows.append(list(range(0, pre_boundary)))  # to not have edge problem
        exclude_windows.append(
            list(range(len(df_data[0]) - post_boundary, len(df_data[0]))))  # to not have edge problem
        range_iti = set(range(len(df_data[0]))).difference(set(np.concatenate(np.array(exclude_windows))))
        random_timing = rnd.sample(range_iti, k=1999)

        from multiprocessing import Pool, cpu_count
        workers = cpu_count()
        pool = Pool(processes=workers)
        async_results = [pool.apply_async(resp_single_neuron, args=(i, random_timing, self.stim_time)) for i in df_data]
        resp_mat = [ar.get() for ar in async_results]
        return resp_mat

    def delay_matrice(self, df_data, resp_masks):
        from delay_onset import delay_
        return delay_(self, df_data, self.stim_time, resp_masks)


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
    def __init__(self, input_path, starting_trial, inhibitory_ids, sf, correction=True, no_cache = False):
        super().__init__(input_path, inhibitory_ids, sf)
        self.xls = pd.read_excel(input_path + 'bpod.xls', header=None)
        self.stim_time = []
        self.reward_time = []
        self.stim_ampl = []
        self.timeout_time = []
        self.detected_stim = []
        with open(input_path + 'params_trial.json', "r") as read_file:
            self.json = json.load(read_file)
        if os.path.exists(input_path + 'behavior_events.json') and not no_cache:
            print('Behavioural information already incorporated in the analog.')
            with open(input_path + 'behavior_events.json', "r") as events_file:
                events = json.load(events_file)
            self.stim_time = np.array(events["stim_time"])
            self.stim_ampl = np.array(events["stim_ampl"])
            self.reward_time = np.array(events["reward_time"])
            self.timeout_time = np.array(events["timeout_time"])
            self.detected_stim = np.array(events["detected_stim"])
        else:
            self.analog = pd.read_csv(input_path + 'analog.txt', sep="\t", header=None)
            self.analog[0] = (self.analog[0] * 10).astype(int)
            self.synchronization_with_iti(starting_trial)

    def synchronization_with_iti(self, starting_trial):
        """
        Update the analog file with information on stimulus time, reward time and timeout time

        Note
        -------
        Get the ITI2, stimulus and reward time from the excel
        Synchronize the ITI2 from the excel with the ITI from the analog


        Parameters
        -------
        excel file, analog file, starting trial for the recording (relative to the behavioral trials)

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
        licks = [np.round(((split[2][(split[0] == 'EVENT') & (split[4] == 94)].values - 2) * 10000).astype(float)) for
                 split in splitted]

        # get the time for ITI2, reward, timeout and stimulus
        ITI_time_xls = data_iti[2].values.astype(float)
        reward_time = data_reward[2].values.astype(float)
        timeout_time = data_timeout[2].values.astype(float)
        stimulus_time = data_stimulus[2].values.astype(float)

        # calculate the time of reward and timeout relative to ITI2
        reward_time_to_ITI = np.around((reward_time - 2) * 10000).tolist()  # in ms
        timeout_time_to_ITI = np.around((timeout_time - 2) * 10000).tolist()  # in ms
        stimulus_time_to_ITI = np.around((stimulus_time - 2) * 10000).tolist()  # in ms

        self.analog.columns = ['t', 'stimulus', 't_iti', 'iti']
        self.analog['reward'] = 0
        self.analog['timeout'] = 0
        self.analog['stimulus_xls'] = 888
        self.analog['licks'] = 0
        self.detected_stim =[]
        # Get the ITI2 from the analog file, as the first "1" value in the digital input of the analog file
        index_iti_final = []
        index_iti_analog = self.analog.index[self.analog['iti'] == 1].tolist()

        for index, elem in enumerate(index_iti_analog):
            if index == 0:
                index_iti_final.append(elem)
            elif elem - index_iti_analog[index - 1] > 1:
                index_iti_final.append(elem)

        index_iti_final.append(index_iti_analog[-1])

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
            if len(index_reward) != 0:
                self.analog.at[index_reward[0], 'reward'] = 2
                self.reward_time.append(int((index_reward[0] / 10000) * self.sf))
            if len(index_timeout) != 0:
                self.analog.at[index_timeout[0], 'timeout'] = 3
                self.timeout_time.append(int((index_timeout[0] / 10000) * self.sf))
                if len(index_stimulus) == 0:
                    timeout_trial = next(ampl_recording_iter)
            if len(index_stimulus) != 0:
                amp = next(ampl_recording_iter)
                self.analog.at[index_stimulus[0], 'stimulus_xls'] = amp
                self.stim_time.append(int((index_stimulus[0] / 10000) * self.sf))
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
        self.analog.to_csv(self.input_path + 'analog_synchronized.csv', index=False)
        to_save = {"stim_time": self.stim_time.tolist(),
                   "stim_ampl": stim_ampl.tolist(),
                   "reward_time": self.reward_time.tolist(),
                   "timeout_time": self.timeout_time.tolist(),
                   "detected_stim": self.detected_stim.tolist()}
        with open(self.input_path + "behavior_events.json", "w") as jsn:
            json.dump(to_save, jsn)


if __name__ == '__main__':
    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
    roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
    folder = "20221008_4746_00_synchro_sigma2_acuFR_tau02"
    path = directory + folder + '/'
    rec = RecordingAmplDet(path, 0, folder, roi_info, no_cache=True)
