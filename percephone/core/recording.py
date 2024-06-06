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
from percephone.utils.io import read_info, correction_drift_fluo
from percephone.analysis.response import resp_matrice, auc_matrice, delay_matrice, peak_matrices
from percephone.analysis.mlr import mlr
from percephone.analysis.mlr_models import classic_model
from percephone.utils.io import extract_analog_from_mesc
matplotlib.use("Qt5Agg")
plt.switch_backend("Qt5Agg")


class Recording:
    """
    The Recording class represents a recording session for one mouse and provides methods to analyze the data.

    Attributes
    ----------
    filename : int
        The filename of the recording session.
    sf : float
        The sampling frequency of the recording session.
    genotype : str
        The genotype of the mouse recorded.
    threshold : float
        The threshold stimulation amplitude.
    input_path : str
        The path to the folder containing the recording data (ending with /).
    matrices : dict[str, dict[str, numpy.ndarray]]
        Stores the different matrices computed from the recording as 2D numpy.ndarray.
        The first dictionary has 2 keys : "EXC" and "INH" neurons, each having a dictionary as value.
        Both dictionaries are then keyed by the computed parameters ("Responsivity", "AUC", etc...) and have a
        2D np.ndarray (nb neurons * nb stimulations) as value.
    df_f_exc : numpy.ndarray
        Stores the computed ΔF/F values for excitatory neurons at each timestep as a 2D numpy.ndarray
        (nb neurons * nb frames)
    df_f_inh : numpy.ndarray
        Stores the computed ΔF/F values for inhibitory neurons at each timestep as a 2D numpy.ndarray
        (nb neurons * nb frames)
    spks_exc : numpy.ndarray #TODO: check where it is computed
        Stores the spikes trains for excitatory neurons at each timestep as a 2D numpy.ndarray (nb neurons * nb frames)
    spks_inh : numpy.ndarray
        Stores the spikes trains for inhibitory neurons at each timestep as a 2D numpy.ndarray (nb neurons * nb frames)
    """

    def __init__(self, input_path, rois_path, mean_f_bsl, cache=True):
        """
        Initializes the Recording object with the given input path, the rois file path, and cache parameters.

        Parameters
        ----------
        input_path: str
            Path to the folder that contains the input recording files.
        rois_path: str
            Path to the Excel ROI file.
        cache: bool, optional (default: True)
            Boolean value indicating whether to use cached files if available (for ΔF/F).
        """

        # Initialization of the instance attributes by reading the ROIs file
        folder_name = os.path.basename(os.path.normpath(input_path)) + "/"
        rois = pd.read_excel(rois_path)
        self.filename, inhibitory_ids, self.sf, self.genotype, self.threshold = read_info(folder_name, rois)
        self.input_path = input_path
        self.matrices = {"EXC": {}, "INH": {}}

        iscell = np.load(input_path + 'iscell.npy', allow_pickle=True)
        # Create a dimension in iscell to define excitatory and inhibitory cells
        exh_inh = np.ones((len(iscell), 1))  # array to define excitatory
        exh_inh[inhibitory_ids] = 0  # correct the array to define inhibitory as 0
        is_exh_inh = np.append(iscell, exh_inh, axis=1)  # add the array in the iscell to have a 3rd column with exh/inh
        cells_list = np.concatenate(np.argwhere(is_exh_inh[:, 0]))
        excitatory_ids = np.concatenate(np.argwhere(is_exh_inh[cells_list, 2]))  # list with excitatory cells

        if os.path.exists(input_path + 'df_f_exc.npy') and cache:
            self.df_f_exc = np.load(input_path + 'df_f_exc.npy')
            self.df_f_inh = np.load(input_path + 'df_f_inh.npy')
        else:
            self.df_f_exc = self.compute_df_f(excitatory_ids, input_path + 'df_f_exc.npy', mean_f_bsl)
            self.df_f_inh = self.compute_df_f(inhibitory_ids, input_path + 'df_f_inh.npy', mean_f_bsl)
            if self.filename == 4445:
                self.df_f_exc = correction_drift_fluo(self.df_f_exc, input_path + 'df_f_exc.npy')
                self.df_f_inh = correction_drift_fluo(self.df_f_inh, input_path + 'df_f_inh.npy')

        if os.path.exists(input_path + 'spks.npy'):
            spks = np.load(self.input_path + "spks.npy")
            self.spks_exc = spks[excitatory_ids]
            self.spks_inh = spks[inhibitory_ids]

        if os.path.exists(input_path + 'matrice_resp_exc.npy') and os.path.exists(input_path + 'matrice_resp_inh.npy'):
            self.matrices["EXC"]["Responsivity"] = np.load(self.input_path + "matrice_resp_exc.npy")
            self.matrices["INH"]["Responsivity"] = np.load(self.input_path + "matrice_resp_inh.npy")

    def compute_df_f(self, cell_ids, save_path, mean_f_bsl):
        """
        Compute ΔF/F and save it as a numpy.ndarray

        Note
        ---------
        Defining The Baseline F0:
        Portera-Cailliau baseline (He et al., Goel et al.): "baseline period is the 10-s period with the lowest variation
        (s.d.) in ΔF/F."
        Since to get the ΔF/F we already need to define a baseline, here the Baseline Period was defined as the 10s
        period with the lowest variation in F_neuropil_corrected

        Parameters
        ----------
        mean_f_bsl: boolean compute df/f with mean F for correction
        cell_ids: list
            ids of cells for which the df_f will be computed.
        save_path : str
            The folder where the df_f matrix will be saved (same as the input path)

        Returns
        -------
        df_f_percen : numpy.ndarray
            ΔF/F of all the cells selected (nb neurons * nb frames)
        """
        print("Calculation Delta F / F.")
        f_ = np.load(self.input_path + 'F.npy', allow_pickle=True)
        print(f"Nb of frames in F: {len(f_[0])}")
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
        if mean_f_bsl:
            f_baseline_2d_array = np.transpose([np.mean(f_nn_corrected, axis=1)]*len(f_nn_corrected[0]))
        # df/f for all cells for each frame
        df_f = np.divide(np.subtract(f_nn_corrected,  f_baseline_2d_array), f_baseline_2d_array)
        df_f_percen = df_f * 100
        np.save(save_path, df_f_percen)
        return df_f_percen

    def responsivity(self):
        """
        Compute responsivity matrices for both excitatory and inhibitory neurons. For a given neuron and a given
        stimulation, one integer among the following is added to the matrix :

        - 1 if there is an increase in the neuron's activity
        - 0 if there is no change in the neuron's activity
        - -1 if there is a decrease in the neuron's activity

        This method updates the 'Responsivity' matrices for the excitatory and inhibitory neurons in the 'matrices'
        attribute.
        """
        print("Calcul of repsonsivity.")
        self.matrices["EXC"]["Responsivity"] = np.array(resp_matrice(self, self.zscore_exc))
        self.matrices["INH"]["Responsivity"] = np.array(resp_matrice(self, self.zscore_inh))
        np.save(self.input_path + "matrice_resp_exc.npy", self.matrices["EXC"]["Responsivity"])
        np.save(self.input_path + "matrice_resp_inh.npy", self.matrices["INH"]["Responsivity"])

    def delay_onset_map(self):
        """
        Calculate the delay onset map for the excitatory and inhibitory neurons.

        This method updates the 'Delay_onset' matrices for the excitatory and inhibitory neurons in the 'matrices'
        attribute.
        """
        self.matrices["EXC"]["Delay_onset"] = delay_matrice(self, self.df_f_exc, self.stim_time,
                                                            self.matrices["EXC"]["Responsivity"])
        self.matrices["INH"]["Delay_onset"] = delay_matrice(self, self.df_f_inh, self.stim_time,
                                                            self.matrices["INH"]["Responsivity"])

    def auc(self):
        """
        Computes the zscore AUC matrices of the signal during the response period.

        Note
        ---------
        Computes the AUC between 0 (the neuron normalized baseline) and the positive part of the curve for positive
        responses, or between 0 and the negative part of the curve for negative responses.
        10 interpolated values are added between each frame for a more precise calculation of the AUC.


        This method updates the 'AUC' matrices for the excitatory and inhibitory neurons in the 'matrices' attribute.
        """
        self.matrices["EXC"]["AUC"] = auc_matrice(self, self.zscore_exc, self.matrices["EXC"]["Responsivity"])
        self.matrices["INH"]["AUC"] = auc_matrice(self, self.zscore_inh, self.matrices["INH"]["Responsivity"])

    def peak_delay_amp(self):
        """
        Calculate the zscore peak delay and peak amplitude for each neuron and each stimulation. The peak is the maximum
        amplitude when the responsivity is 1 or the minimum amplitude when the responsivity is -1. numpy.NaN is added to
        the matrix when the responsivity is 0.


        This method updates the "Peak_delay" and "Peak_amplitude" matrices for the excitatory and inhibitory neurons in
        the "matrices" attribute.
        """
        self.matrices["EXC"]["Peak_delay"], self.matrices["EXC"]["Peak_amplitude"] = peak_matrices(self,
                                                                                                   self.zscore_exc,
                                                                                                   self.matrices["EXC"][
                                                                                                       "Responsivity"])
        self.matrices["INH"]["Peak_delay"], self.matrices["INH"]["Peak_amplitude"] = peak_matrices(self,
                                                                                                   self.zscore_inh,
                                                                                                   self.matrices["INH"][
                                                                                                       "Responsivity"])


class RecordingStimulusOnly(Recording):
    """
    The RecordingStimulusOnly class represents a recording session for one mouse and provides methods to analyze the
    data.

    Attributes
    ----------
    analog : pandas.DataFrame
        Analog file stored as a pandas DataFrame.
    stim_time : list[int]
        A list of the stimulation times in ms.
    stim_ampl : list[float]
        A list of the stimulation amplitudes.
    """
    def __init__(self, input_path, inhibitory_ids, sf, correction=True):
        """
        This method initializes an instance of the class. It reads the 'analog.txt' file from the input path and checks
        if the "stim_ampl_time.csv" file exists. If it exists, it reads the stimulus time and amplitude data from it.
        If it doesn't exist, it calls the "synchronization_no_iti" method with the provided correction boolean.

        Parameters
        ----------
        input_path : str
            Path to the folder that contains the input files.
        inhibitory_ids : list
            The list of inhibitory neuron IDs.
        sf : float
            The sampling frequency of the recording session.
        correction : bool, optional
            Whether to perform correction. Default is True.
        """
        super().__init__(input_path, inhibitory_ids, sf)  # TODO: bad initialization of the instance
        self.analog = pd.read_csv(input_path + 'analog.txt', sep="\t")
        if os.path.exists(input_path + 'stim_ampl_time.csv'):
            print('Analog information already computed. Reading stimulus time and amplitude.')
            self.stim_time = pd.read_csv(input_path + 'stim_ampl_time.csv', usecols=['stim_time']).values.flatten()
            self.stim_ampl = pd.read_csv(input_path + 'stim_ampl_time.csv', usecols=['stim_ampl']).to_numpy().flatten()
        else:
            self.synchronization_no_iti(correction)

    def synchronization_no_iti(self, correction_shift):
        """
        Get the stimulus time and amplitude from the analog file and save a csv file with the stimulus amplitude and
        time (stimulus starting time) in ms.

        Note
        -------
        Define stimulus delivery based on the analog
        Stimulus = when analog[1]>=0.184
        First find the first peak for each stimulus (stimuli every 500ms)
        Then find the stimulus onset when amplitude is >0.184

        Parameters
        -------
        correction_shift : bool
            Indicate if the correction of the shift of frame should be executed.
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
            """ Perform a frames' correction. 8 is the number of frames that the last stimulus will be shifted before"""
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
    """
    Class for analyzing recording data with amplitude detection. Inherits from the Recording class.

    Attributes
    ----------
    analog : pandas.DataFrame
        Analog file stored as a pandas DataFrame.
    xls : pandas.DataFrame
        bpod file stored as a pandas DataFrame.
    stim_time : numpy.ndarray of int
        A 1D numpy.ndarray of the stimulation times in frames.
    stim_ampl : numpy.ndarray of float #TODO: why not int
        A 1D numpy.ndarray of the stimulation amplitudes.
    stim_durations : numpy.ndarray of float #TODO: why not int
        A 1D numpy.ndarray of the stimulation's durations
    reward_time : numpy.ndarray of int
        A 1D numpy.ndarray of the reward times in frames.
    timeout_time : numpy.ndarray of int
        A 1D numpy.ndarray of the timeout (no-go times) in frames.
    lick_time : numpy.ndarray of int
        A 1D numpy.ndarray of the lick times in frames.
    detected_stim : numpy.ndarray of bool
        A 1D numpy.ndarray of if the mouse responded to a stimulation.
    mlr_labels_exc :
        #TODO: to fill
    mlr_labels_inh :
        #TODO: to fill
    json : list[dict]
        Read of the file params_trials.json
    zscore_exc : numpy.ndarray of float
        A 2D numpy.ndarray of the zscore of each excitatory neuron at each frame of the recording
        (nb neurons * nb frames)
    zscore_inh : numpy.ndarray of float
        A 2D numpy.ndarray of the zscore of each inhibitory neuron at each frame of the recording
        (nb neurons * nb frames)
    """
    def __init__(self, input_path, starting_trial, rois_path, tuple_mesc=(0, 0), mean_f=False, correction=True,
                 cache=True):
        """
        Parameters
        ----------
        input_path : str
            Path to the folder that contains the input files.
        starting_trial : int
            The starting trial number.
        rois_path: str
            Path to the Excel ROI file.
        tuple_mesc : tuple, optional
            Tuple representing the measurements per second (mesc) values for extraction, default is (0, 0).
        correction : bool, optional
            Flag indicating whether to apply correction, default is True.
        cache: bool
            Boolean value indicating whether to use cached files if available.
        analog_sf : int, optional
            Sampling frequency of the analog data, default is 10000.
        """
        super().__init__(input_path, rois_path, mean_f, cache=cache)
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
            if (len(self.analog[0])/10000) > 600:
                self.analog[0] = (self.analog[0] * 10).astype(int)
                analog_sf = 10000
            else:
                self.analog[0] = (self.analog[0]).astype(int)
                analog_sf = 1000
            self.synchronization_with_iti(starting_trial, analog_sf, correction)

        self.zscore_exc = self.zscore(self.df_f_exc)
        self.zscore_inh = self.zscore(self.df_f_inh)

    def zscore(self, dff):
        """
        Computes the standardized zscore from the ΔF/F.

        Parameters
        ----------
        dff: numpy.ndarray
            delta f over f array from inh or exc

        Returns
        -------
        zscore: numpy.ndarray
            zscore for one set of neurons (inh or exc)
        """
        data = np.concatenate(np.stack(
            dff[:,
            np.linspace(self.stim_time - int(0.5 * self.sf), self.stim_time, num=int(0.5 * self.sf) + 1, dtype=int)],
            axis=2))
        mean_bsl = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        zsc = np.divide(np.subtract(dff, mean_bsl[:, np.newaxis]), std[:, np.newaxis])
        return zsc

    def synchronization_with_iti(self, starting_trial, analog_s, correction):
        """
        Update the analog file with information on stimulus time, reward time and timeout time

        Note
        -------
        Get the ITI2, stimulus and reward time from the Excel
        Synchronize the ITI2 from the Excel with the ITI from the analog


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
        licks = [np.round(((split[2][(split[0] == 'EVENT') & (split[4] == 94)].values - 2) * analog_s).astype(float))
                 for
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
                    lick_time = int((index_licks[0] / analog_s) * self.sf)
                    if correction:
                        lick_time = lick_time - int(((1 / self.sf) * (index_licks[0] / analog_s)) * (1 / 3))
                    self.lick_time.append(lick_time)
            if len(index_reward) != 0:
                self.reward_time.append(int((index_reward[0] / analog_s) * self.sf))
            if len(index_timeout) != 0:
                self.timeout_time.append(int((index_timeout[0] / analog_s) * self.sf))
                if len(index_stimulus) == 0:
                    timeout_trial = next(ampl_recording_iter)
            if len(index_stimulus) != 0:
                amp = next(ampl_recording_iter)
                index_stim = int((index_stimulus[0] / analog_s) * self.sf)
                if correction:
                    self.stim_time.append(index_stim - int(((1 / self.sf) * (index_stimulus[0] / analog_s)) * (1 / 3)))
                else:
                    self.stim_time.append(index_stim)
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
                durations[i] = int(0.5*self.sf)
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
                self.mlr_labels_exc = events["exc"]
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

    def stim_ampl_filter(self, stim_ampl="all", include_no_go=False):
        """
        Returns a vector of booleans to select the stimulations of desired amplitude.

        Parameters
        ----------
        stim_ampl : str or list[int]
            The amplitudes of stimulation that we want to select. List of absolute values or relative to threshold
            (threshold, supra, sub or all)

        include_no_go : bool (optional, default = False)
            Whether to include no-go trials (amplitude 0) or not.

        Returns
        -------
        numpy.ndarray[bool]
            A vector of booleans to select the stimulations of desired amplitude.
        """
        if include_no_go:
            all_ampl = np.arange(0, 14, 2)
        else:
            all_ampl = np.arange(2, 14, 2)

        if stim_ampl == "threshold":
            amplitudes = self.threshold
        elif stim_ampl == "supra":
            amplitudes = all_ampl[all_ampl >= self.threshold]
        elif stim_ampl == "sub":
            amplitudes = all_ampl[all_ampl < self.threshold]
        elif stim_ampl == "all":
            amplitudes = all_ampl
        else:
            amplitudes = np.array(stim_ampl)
        selected_stim = np.isin(self.stim_ampl, amplitudes)
        return selected_stim


if __name__ == '__main__':
    import percephone.plts.heatmap as hm

    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/Amplitude_Detection_DMSO_BMS/"
    roi_info = directory + "Fmko_bms&dmso_info.xlsx"
    folder = "20231108_5886_00_BMS_det_synchro"
    path_to_mesc = directory + "20231108_5886_BMS_det.mesc"

    extract_analog_from_mesc(path_to_mesc, (0, 0),  30.9609, 20000, directory + folder + "/")
    rec = RecordingAmplDet(directory + folder + "/", 0, roi_info, cache=False, correction=False)
    hm.interactive_heatmap(rec, rec.zscore_exc)
    #
    # from percephone.analysis.neuron_var import plot_heatmap, get_zscore
    # zsc,t_stim = get_zscore(rec, exc_neurons=True, sort=True, amp_sort=True)
    # plot_heatmap(rec,  zsc, sorted=True, amp_sorted=True)
