"""Ourania Semelidou, 27/03/2023
Core classes for recording, synchronization, synchronization w/o ITI2 analog, """

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")
import scipy.signal as ss


class Recording:
    def __init__(self, input_path, inhibitory_ids):
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
        f_ : npy matrix (cells, frames)
            Fluorescence of all ROIs from suite2p
        f_neu : npy matrix (cells, frames)
            Fluorescence of all neuropils from suite2p
        save_path : folder where the df_f matrix will be saved (same as the input path)

        Output
        -------
        npy matrix with DF/F of all excitatory or inhibitory cells
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


class RecordingStimulusOnly(Recording):
    def __init__(self, input_path, inhibitory_ids):
        super().__init__(input_path, inhibitory_ids)
        self.analog = np.loadtxt(input_path + 'analog.txt')
        if os.path.exists(input_path + 'stim_ampl_time.csv'):
            print('Analog information already computed. Reading stimulus time and amplitude.')
            self.stim_time = pd.read_csv('stim_ampl_time.csv', usecols=['stim_time'])
            self.stim_ampl = pd.read_csv('stim_ampl_time.csv', usecols=['stim_ampl'])
        else:
            self.synchronization_no_iti()

    def synchronization_no_iti(self):
        """
        Get the stimulus time and amplitude from the analog file

        Note
        -------
        Define stimulus delivery based on the analog
        Stimulus = when analog[1]>=0.184
        First find the first peak for each stimulus (stimuli every 500ms)
        Then find the stimulus onset when amplitude is >0.184

        Parameters
        -------
        input file

        Output
        ------
        csv file with the stimulus amplitude and time (stimulus starting time) in ms
        """

        print('Obtaining time and amplitude from analog.')
        analog_trace = self.analog[:, 1]
        stim_peak_indx, stim_properties = ss.find_peaks(analog_trace, prominence=0.15, distance=200)
        peaks_diff = np.diff(stim_peak_indx)
        indices = np.concatenate([[True], peaks_diff > 50000])
        stim_peak_indx = stim_peak_indx[indices]
        stim_ampl_analog = analog_trace[stim_peak_indx]
        stim_ampl = np.around(stim_ampl_analog, decimals=1)
        stim_ampl[stim_ampl == 1.5] = 12
        stim_ampl[stim_ampl == 1.3] = 10
        stim_ampl[stim_ampl == 1.1] = 8
        stim_ampl[stim_ampl == 0.8] = 6
        stim_ampl[stim_ampl == 0.6] = 4
        stim_ampl[stim_ampl == 0.4] = 2

        def stim_onset_calc(peak_index):
            time_window = 400  # 40 ms before peak
            signal_window = analog_trace[peak_index - time_window:peak_index]
            return np.argwhere(signal_window > 0.184)[0][0] + peak_index - time_window

        stim_onset_idx = list(map(stim_onset_calc, stim_peak_indx))
        stim_onset_time = self.analog[stim_onset_idx, 0]

        # plot the analog to test the index of stimulus onset
        fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        axs[0].plot(analog_trace)
        axs[0].plot(stim_onset_idx, analog_trace[stim_onset_idx], 'x')
        plt.show()

        self.stim_time = stim_onset_time
        self.stim_ampl = stim_ampl
        # save the stimulus time and amplitude as csv
        pd.DataFrame({'stim_time': stim_onset_time,'stim_ampl': stim_ampl}).to_csv(self.input_path + 'stim_ampl_time.csv')


class RecordingAmplDet(Recording):
    def __init__(self, input_path, starting_trial, inhibitory_ids):
        super().__init__(input_path, inhibitory_ids)
        self.analog = pd.read_csv(input_path + 'analog.txt', sep="\t", header=None)
        self.analog[0] = (self.analog[0] * 10).astype(int)
        self.xls = pd.read_excel(input_path + 'bpod.xls', header=None)
        with open(input_path + 'params_trial.json', "r") as read_file:
            self.json = json.load(read_file)
        if os.path.exists(input_path + 'analog_synchronized.csv'):
            print('Behavioural information already incorporated in the analog.')
        else:
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

        Output
        ------
        Updated analog file (csv) with the stimulus, timeout, reward time
        """

        # get the ITI2, reward, timeout and stimulus from the xls
        data_iti = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'ITI2']
        data_reward = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'Reward']
        data_timeout = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'Timeout']
        data_stimulus = self.xls[self.xls[0] == 'STATE'][self.xls[4] == 'Stimulus']

        # get the time for ITI2, reward, timeout and stimulus
        ITI_time_xls = data_iti[2].values.astype(float)
        reward_time = data_reward[2].values.astype(float)
        timeout_time = data_timeout[2].values.astype(float)
        stimulus_time = data_stimulus[2].values.astype(float)

        # calculate the time of reward and timeout relative to ITI2
        reward_time_to_ITI = np.around((reward_time - ITI_time_xls) * 10000).tolist()  # in ms
        timeout_time_to_ITI = np.around((timeout_time - ITI_time_xls) * 10000).tolist()  # in ms
        stimulus_time_to_ITI = np.around((stimulus_time - ITI_time_xls) * 10000).tolist()  # in ms

        self.analog.columns = ['t', 'stimulus', 't_iti', 'iti']
        self.analog['reward'] = 0
        self.analog['timeout'] = 0
        self.analog['stimulus_xls'] = 888

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
        end_protocol = len(index_iti_final) + starting_trial

        # Get lists for the reward, timeout and stimulus from the xls calculations for the trials that were recorded
        for icount in range(starting_trial, len(ITI_time_xls)):
            if icount < end_protocol:
                reward_to_analog.append(reward_time_to_ITI[icount])  #index (ms*10)
                timeout_to_analog.append(timeout_time_to_ITI[icount])
                stimulus_to_analog.append(stimulus_time_to_ITI[icount])
                ampl_recording.append(self.json[icount]["amp"])
        ampl_recording_iter = iter(ampl_recording)

        for icount_time in range(len(index_iti_final)):
            ITI_time_analog = self.analog.at[index_iti_final[icount_time], 't'] #round?
            reward_time_analog = ITI_time_analog + reward_to_analog[icount_time]
            timeout_time_analog = ITI_time_analog + timeout_to_analog[icount_time]
            stimulus_time_analog = ITI_time_analog + stimulus_to_analog[icount_time]
            index_reward = self.analog.index[self.analog['t'] == reward_time_analog].to_list()
            index_timeout = self.analog.index[self.analog['t'] == timeout_time_analog].to_list()
            index_stimulus = self.analog.index[self.analog['t'] == stimulus_time_analog].to_list()
            if len(index_reward) != 0:
                self.analog.at[index_reward[0], 'reward'] = 2
            if len(index_timeout) != 0:
                self.analog.at[index_timeout[0], 'timeout'] = 3
                if len(index_stimulus) == 0:
                    timeout_trial =next(ampl_recording_iter)
            if len(index_stimulus) != 0:
                self.analog.at[index_stimulus[0], 'stimulus_xls'] = next(ampl_recording_iter)

        self.analog.to_csv(self.input_path + 'analog_synchronized.csv', index=False)


if __name__ == '__main__':
    #test_recording = RecordingStimulusOnly("Z:\\Current_members\\Ourania_Semelidou\\2p\\Ca_imaging_analysis_PreSynchro\\Fmko\\StimulusOnly\\4445\\20220728_4445_00_synchro\\")
    test_detection_rec = RecordingAmplDet(input_path="Z:\\Current_members\\Ourania_Semelidou\\2p\\Ca_imaging_analysis_PreSynchro\\Fmko\\Amplitude_Detection\\4445\\20220710_4445_00_synchro\\",
                                          starting_trial=0, inhibitory_ids=[7, 24, 34, 73, 89, 103, 683])

    #analog_synced = pd.read_csv("Z:\\Current_members\\Ourania_Semelidou\\2p\\Ca_imaging_analysis_PreSynchro\\Fmko\\Amplitude_Detection\\4445\\20220710_4445_00_synchro\\analog_synchronized.csv", sep=',', header=0)
    #test_det_ampl = test_detection_rec.analog[test_detection_rec.analog['stimulus_xls']!=888 ]


