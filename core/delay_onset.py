"""Théo Gauvrit, 04/09/2023
Compute delay onset by linear regression and convolution"""

import matplotlib
import numpy as np
import pandas as pd
import core as pc
from sklearn.linear_model import LinearRegression
from Helper_Functions.Utils_core import kernel_biexp
import scipy.signal as ss
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")


def delay_(rec, df, stim_times, resp_mask):

    neuro_resp_delay = np.full((len(df), len(stim_times)), np.nan)
    st_index = [list(range(i, i+int(0.5*rec.sf))) for i in stim_times]
    st_vector = np.zeros(len(df[0]))
    for n_neuro, neuron_df in enumerate(df):
        for i, st_i in enumerate(stim_times):
            if not resp_mask[n_neuro][i]:
                continue
            rs_ = []
            if st_i + int(0.5 * rec.sf) + 10 > len(neuron_df):  # remove last stim to close to end of signal
                continue
            smooth_sig = ss.savgol_filter(neuron_df, 5, 1)
            sig = smooth_sig[st_i - 10:st_i + int(0.5 * rec.sf) + 10]
            baseline = np.mean(smooth_sig[st_i - 10:st_i])
            st_vector[np.concatenate(st_index)] = np.max(sig) + 0.50 * np.max(sig)
            conv_st = np.convolve(st_vector, kernel_biexp(rec.sf), mode='same') * 1/rec.sf
            conv_st = conv_st + baseline
            for y in range(15):
                conv_ = range(st_i - 5, st_i + int(0.5 * rec.sf))
                portion_sig = sig[0 + y: 20 + y]
                conv_ = np.array(conv_)[np.array(conv_) < len(conv_st)]
                reg_fit = LinearRegression().fit(conv_st[conv_][:, np.newaxis], portion_sig[:, np.newaxis])
                r_ = reg_fit.score(conv_st[conv_][:, np.newaxis], portion_sig[:, np.newaxis])
                rs_.append(r_)
            r_i_ = np.argmax(rs_)
            neuro_resp_delay[n_neuro, i] = r_i_
    return neuro_resp_delay


if __name__ == '__main__':
    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format/"
    roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
    folder = "20220715_4456_00_synchro"
    path = directory + folder + '/'
    recording = pc.RecordingAmplDet(path, 0, folder, roi_info)
    from zscore import zscore
    recording.zscore = zscore(recording)
    df = recording.zscore[1, :]  # recording.df_f_exc[1, :]
    original_signal = df
    stim_vector = np.zeros(len(original_signal))
    stim_index = [list(range(i, i+int(0.5*recording.sf))) for i in recording.stim_time[recording.stim_ampl == 12]]
    stim_vector[np.concatenate(stim_index)] = 100
    conv_stim = np.convolve(stim_vector, kernel_biexp(recording.sf), mode='same') * 1/recording.sf
    fig, ax = plt.subplots()
    times = np.linspace(0, len(original_signal)/recording.sf, len(original_signal))
    ax.plot(times, original_signal, label='original')
    ax.plot(times, conv_stim, label='convolved')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('signal')
    ax.legend()
    plt.show()
    stims = recording.stim_time[recording.stim_ampl == 12]
    for stim_i in stims[0:40]:
        rs = []
        smooth_signal = ss.savgol_filter(df, 5, 1)
        signal = smooth_signal[stim_i-10:stim_i + int(0.5 * recording.sf)+10]
        bsl = np.mean(smooth_signal[stim_i-10:stim_i])
        stim_vector[np.concatenate(stim_index)] = np.abs(np.max(signal)) + 0.50 * np.abs(np.max(signal))
        conv_stim = np.convolve(stim_vector, kernel_biexp(recording.sf), mode='same') * 1/recording.sf
        conv_stim = conv_stim + bsl
        for i in range(15):
            conv = range(stim_i-5, stim_i + int(0.5 * recording.sf))
            portion_signal = signal[0+i:20+i]
            reg = LinearRegression().fit(conv_stim[conv][:, np.newaxis], portion_signal[:, np.newaxis],)
            r = reg.score(conv_stim[conv][:, np.newaxis], portion_signal[:, np.newaxis])
            rs.append(r)
            # fig, ax = plt.subplots()
            # ax.plot(portion_signal, label='original')
            # ax.plot(conv_stim[conv], label='convolved')
            # ax.set_title(str(r) + " delay:" + str(i))
            # plt.show()
        r_i = np.argmax(rs)
        to_plot_s = smooth_signal[stim_i - 100 + r_i:stim_i + 100 + r_i]
        to_plot_c = conv_stim[stim_i - 95:stim_i + 105]
        fig, ax = plt.subplots()
        times = np.linspace(-95/recording.sf, 105 / recording.sf, len(to_plot_s))
        ax.plot(times,  to_plot_s, label='original')
        ax.plot(times,  to_plot_c, label='convolved')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('signal')
        ax.legend()
        ax.set_title(str(rs[r_i]) + " delay:" + str(r_i))
        plt.show()
