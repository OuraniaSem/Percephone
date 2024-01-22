"""
Théo Gauvrit 18/01/2024
Multiple Linear regression analysis
"""

import os
import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
import core as pc
from Helper_Functions.mlr_functions import mlr
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")


def detected_amp(filename, infos):
    name = int(filename[9:13])
    n_record = filename[14:16]
    row = infos[(infos["Number"] == name) & (infos["Recording number"] == int(n_record))]
    psycho = np.array(row["Stimulus detection"].values[0].split(", ")).astype(float)
    amps = np.array([2, 4, 6, 8, 10, 12])[psycho > 0.5]
    return amps


directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
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
# files = ["2022tauN_4745_01_synchro"]
for id, folder in enumerate(files):
    if os.path.isdir(directory + folder):
        path = directory + folder + '/'
        rec = pc.RecordingAmplDet(path, 0, folder, roi_info)
        dff = rec.df_f_exc
        # stims

        durations = np.zeros(len(rec.stim_time))
        for i, stim_t in enumerate(rec.stim_time):
            diff_ar = np.absolute(rec.reward_time - stim_t)
            if diff_ar[diff_ar.argmin()] >= int(0.5 * rec.sf) - 1:
                durations[i] = 15
            else:
                durations[i] = diff_ar[diff_ar.argmin()]
        s_durations = durations

        # detected
        stim_det = rec.stim_time[rec.detected_stim]
        det_duration = s_durations[rec.detected_stim]
        stim_vector = np.zeros(len(dff[0]))
        stim_index = [list(range(stim, int(stim + det_duration[i]))) for i, stim in enumerate(stim_det)]
        stim_vector[np.concatenate(stim_index)] = 100
        conv_stim_det = np.convolve(stim_vector, kernel_bi, mode='same') * dt

        # undetected
        stim_undet = rec.stim_time[~rec.detected_stim]
        undet_duration = s_durations[~rec.detected_stim]
        stim_vector_undet = np.zeros(len(dff[0]))
        stim_index = [list(range(stim, int(stim + undet_duration[i]))) for i, stim in enumerate(stim_undet)]
        stim_vector_undet[np.concatenate(stim_index)] = 100
        conv_stim_undet = np.convolve(stim_vector_undet, kernel_bi, mode='same') * dt

        # reward
        reward_duration = 0.1  # s
        reward_vector = np.zeros(len(dff[0]))
        reward_index = [list(range(i, i + int(reward_duration * rec.sf))) for i in rec.reward_time]
        reward_vector[np.concatenate(reward_index)] = 200
        conv_reward = np.convolve(reward_vector, kernel_bi, mode='same') * dt

        # timeout
        timeout_duration = 2  # s
        timeout_vector = np.zeros(len(dff[0]))
        timeout_index = np.concatenate([list(range(i, i + int(timeout_duration * rec.sf))) for i in rec.timeout_time])

        timeout_vector[timeout_index[timeout_index < len(timeout_vector)]] = 100
        conv_timeout = np.convolve(timeout_vector, kernel_bi, mode='same') * dt

        # postreward
        # postreward_duration = 2  # s
        # postreward_vector = np.zeros(len(smooth_signal[0]))
        # postreward_index = np.concatenate(
        #     [list(range(i + int(reward_duration * rec.sf), i + int(postreward_duration * rec.sf))) for i in
        #      rec.reward_time])
        # postreward_vector[postreward_index[postreward_index < len(postreward_vector)]] = 100
        # conv_postreward = np.convolve(postreward_vector, kernel_bi, mode='same') * dt

        regressors = np.array([conv_stim_det, conv_stim_undet,  conv_reward, conv_timeout])

        text_labels, n_neurons_per_label = mlr(dff, regressors, rec.sf)
        output["labels"] = text_labels
        output[folder] = n_neurons_per_label

output = output.loc[(output.iloc[:, 1:] != 0).any(axis=1)]
s = output.sum()
output_s = output.sort_index(key=output.sum(1).get)
output_s.to_csv("output_mlr_tau02.csv")
