"""Théo Gauvrit, 05/09/2023
Perform multiple linear regression between  DECONVOLUED calcium neuronal signals and external signals(stim, reward, licj, etc )
Adapted from Banyuls summer school 2023
Adapted for devonclued spikes from suite2p
"""

import itertools
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import core as pc

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Qt5Agg")


def bootstrap(signal):
    """resample a time series with replacement.

    Parameters :
    ------------
    signal : array
        original time series, the first dimension corresponds to time, shape (n_timepoints,...)

    Return :
    --------
    resampled_signal : array
        bootstrapped time series
    """
    # ⌨️⬇️
    l = len(signal)
    random_indices = np.random.randint(l, size=l)
    resampled_signal = signal[random_indices]
    return resampled_signal


def detected_amp(filename, infos):
    name = int(filename[9:13])
    n_record = filename[14:16]
    row = infos[(infos["Number"] == name) & (infos["Recording number"] == int(n_record))]
    psycho = np.array(row["psycho"].values[0].split(", ")).astype(float)
    amps = np.array([2, 4, 6, 8, 10, 12])[psycho > 0.5]
    return amps


directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format/"
roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")

output = pd.DataFrame()

sf = 30.9609
tau_r = 0.07  # s0.014
tau_d = 0.236  # s
kernel_size = 10  # size of the kernel in units of tau
a = 5  # scaling factor for biexpo kernel
dt = 1 / sf  # spacing between successive timepoints
n_points = int(kernel_size * tau_d / dt)
kernel_times = np.linspace(-n_points * dt, n_points * dt,
                           2 * n_points + 1)  # linearly spaced array from -n_points*dt to n_points*dt with spacing dt
kernel_bi = a * (1 - np.exp(-kernel_times / tau_r)) * np.exp(-kernel_times / tau_d)
kernel_bi[kernel_times < 0] = 0

files = os.listdir(directory)
files = ["2022tauN_4456_00_synchro"]
for id, folder in enumerate(files):
    if os.path.isdir(directory + folder):
        path = directory + folder + '/'
        rec = pc.RecordingAmplDet(path, 0, folder, roi_info)
        dff = rec.spks_exc
        smooth_signal = ss.savgol_filter(dff, 5, 1)
        # stims
        detetected_amps = detected_amp(folder, roi_info)
        if len(detetected_amps) == 0:
            continue
        stims = np.concatenate([rec.stim_time[rec.stim_ampl == stim] for stim in detetected_amps])
        len(rec.stim_time)
        len(rec.reward_time)
        durations = np.zeros(len(rec.stim_time))
        for i, stim_t in enumerate(rec.stim_time):
            diff_ar = np.absolute(rec.reward_time - stim_t)
            if diff_ar[diff_ar.argmin()] >= int(0.5 * rec.sf) - 1:
                durations[i] = 15
            else:
                durations[i] = diff_ar[diff_ar.argmin()]
        s_durations = np.concatenate([durations[rec.stim_ampl == stim] for stim in detetected_amps])
        stim_vector = np.zeros(len(smooth_signal[0]))

        stim_index = [list(range(stim, int(stim + s_durations[i]))) for i, stim in enumerate(stims)]
        stim_vector[np.concatenate(stim_index)] = 100

        # reward
        reward_duration = 0.1  # s
        reward_vector = np.zeros(len(smooth_signal[0]))
        reward_index = [list(range(i, i + int(reward_duration * rec.sf))) for i in rec.reward_time]
        reward_vector[np.concatenate(reward_index)] = 200

        # timeout
        timeout_duration = 2  # s
        timeout_vector = np.zeros(len(smooth_signal[0]))
        timeout_index = np.concatenate([list(range(i, i + int(timeout_duration * rec.sf))) for i in rec.timeout_time])

        timeout_vector[timeout_index[timeout_index < len(timeout_vector)]] = 100

        # postreward
        postreward_duration = 2  # s
        postreward_vector = np.zeros(len(smooth_signal[0]))
        postreward_index = np.concatenate(
            [list(range(i + int(reward_duration * rec.sf), i + int(postreward_duration * rec.sf))) for i in
             rec.reward_time])
        postreward_vector[postreward_index[postreward_index < len(postreward_vector)]] = 100

        regressors = np.array([stim_vector, reward_vector, timeout_vector])

        n_regressors = len(regressors)
        df_times = np.linspace(0, len(dff[0]) / rec.sf, len(dff[0]))
        t1 = 0
        t2 = df_times[-1]

        # Linear regression
        reg = LinearRegression().fit(regressors.T, dff.T)
        coef = reg.coef_
        r2 = r2_score(dff.T, reg.predict(regressors.T), multioutput='raw_values')

        n_neurons = len(dff)  # Not sure, problem in the notebooks, don't understand where it came from
        n_resamples = 50
        r2_sbs = np.zeros(n_resamples * n_neurons)
        from corr_exter_signal import stationary_bootstrap, find_significant_neurons
        from tqdm import tqdm

        for i in tqdm(range(n_resamples)):
            dff_resampled = stationary_bootstrap(dff.T).T
            reg = LinearRegression().fit(regressors.T, dff_resampled.T)
            r2_sbs[i * n_neurons:(i + 1) * n_neurons] = r2_score(dff_resampled.T, reg.predict(regressors.T),
                                                                 multioutput='raw_values')
        indices_r2, mask_r2 = find_significant_neurons(r2, r2_sbs, max_neurons=len(dff))
        len(indices_r2)
        n_resamples = 100
        coef_bs = np.zeros((n_resamples, n_neurons, n_regressors))
        samples = np.concatenate([regressors, dff])

        for n in tqdm(range(n_resamples)):
            resampled = bootstrap(samples.T).T
            reg = LinearRegression().fit(resampled[:n_regressors].T, resampled[n_regressors:].T)
            coef_bs[n] = reg.coef_

        coef_se = np.zeros_like(coef)
        for i in range(n_neurons):
            coef_se[i] = np.sqrt(n_resamples / (n_resamples - 1)) * np.std(coef_bs[:, i, :], axis=0, ddof=1)

        neuron_labels = np.zeros_like(coef)
        for i in range(n_regressors):  # loop over all regressors
            neuron_labels[
                coef[:, i] > 2 * coef_se[:, i], i] = 1  # if the coefficients are larger than 2*SE we set the sign to 1
            neuron_labels[coef[:, i] < -2 * coef_se[:, i], i] = -1  # if they are smaller than -2*SE we set it to -1
        neuron_labels = neuron_labels[indices_r2]  # we only consider the neurons for which the fit is good

        all_possible_labels = list(
            itertools.product([0, 1, -1], repeat=n_regressors))  # get all possible combinations of 0, 1 and -1
        n_neurons_per_label = []
        text_labels = []
        for label in all_possible_labels:
            n_neurons_per_label.append(
                np.sum(np.all(neuron_labels == label, axis=1)))  # number of neurons with a certain label
            text_labels.append(str(label))  # each label saved as a string
        n_neurons_per_label = np.array(n_neurons_per_label)
        text_labels = np.array(text_labels)
        output["labels"] = text_labels
        output[folder] = n_neurons_per_label

output = output.loc[(output.iloc[:, 1:] != 0).any(axis=1)]
s = output.sum()
output_s = output.sort_index(key=output.sum(1).get)
output_s.to_csv("output_mlr_deconv.csv")
