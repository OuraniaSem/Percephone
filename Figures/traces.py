import os

import numpy as np
import pandas as pd
from multiprocessing import cpu_count, pool

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

import percephone.core.recording as pc
import percephone.plts.stats as ppt
from Figures.response_decoding import get_activity_by_frame_df


def plot_mean_neuronal_trace(frame_df, rec=None, n_type="EXC", trial_avg=True, smoothing=0):
    """Plots the mean neuronal traces for hit and miss trials for all recs"""
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color]}
    data = frame_df[(frame_df.ID == rec) & (frame_df.n_type == n_type)].copy()
    genotype = data.Genotype.values[0]
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 16), constrained_layout=True)
    for i, pattern in enumerate([1, -1]):
        pat_data = data[data.resp == pattern].copy()
        pat_data = pat_data[pat_data.Amplitude != 0].drop(columns=["Genotype", "ID", "Threshold", "Duration", "Amplitude", "n_type", "resp", "n_ID"])
        # Without averaging per trial
        if trial_avg:
            pat_data = pat_data.groupby(["Trial", "Behavior"], as_index=False).mean()
        mean_traces = pat_data.groupby(["Behavior"]).mean().drop(columns=["Trial"]).iloc[:, 15:]
        # if smoothing > 0:
        #     mean_traces = mean_traces.rolling(window=smoothing, axis=1, center=True, min_periods=2).mean()
        x = np.arange(mean_traces.shape[1])
        y_hit = mean_traces.loc[True]
        y_miss = mean_traces.loc[False]
        if smoothing > 0:
            y_hit = savgol_filter(y_hit, window_length=smoothing, polyorder=3)
            y_miss = savgol_filter(y_miss, window_length=smoothing, polyorder=3)
        ax[i].plot(x, y_hit, label='Hit', color=color_dict[genotype][0])
        ax[i].plot(x, y_miss, label='Miss', color=color_dict[genotype][1])
        ax[i].axvline(x=15, ls="--", lw=2, color="red")
        ax[i].axvline(x=30, ls="--", lw=2, color="black")
        ax[i].set_title(f"{genotype} - pattern={pattern}", color=color_dict[genotype][0], fontsize=20)
        ax[i].set_xlabel("Time (s)")
        ax[i].set_xticks([0, 15, 30, 45])
        ax[i].set_xticklabels([-0.5, 0, 0.5, 1])
        ax[i].set_ylabel("Z-score (Mean)")
        ax[i].set_ylim([-1.5, 2])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    fig.suptitle(f"{rec} - pattern={pattern}, trial_avg={trial_avg}", fontsize=12)
    fig.canvas.manager.set_window_title(f"{rec}")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/8_Figure3/trace_{rec}_{n_type}_{trial_avg}.pdf", format="pdf")
    plt.show()
    return mean_traces


def plot_neuronal_corr(frame_df, rec=None, neurons=[], trials=[]):
    n_neurons = len(neurons)
    n_trials = len(trials)
    data = frame_df[(frame_df.ID == rec) & (frame_df.n_type == "EXC")].copy()
    numeric_cols = [c for c in data.columns if isinstance(c, int)]
    keep = [c for c in numeric_cols if 15 <= c < 30]
    data["trace"] = data[keep].values.tolist()
    data = data.drop(columns=numeric_cols)

    resp_col = {0: "#828282", 1: "#db4d00", -1: "#6a00e0"}
    trial_name_dict = {True: "Hit", False: "Miss"}

    fig, ax = plt.subplots(nrows=n_neurons, ncols=n_trials, figsize=(n_trials * 1, n_neurons * 1),
                           constrained_layout=True, sharex=True, sharey=True)
    for row, neuron in enumerate(neurons):
        for col, trial in enumerate(trials):
            data_row = data[(data.Trial == trial) & (data.n_ID == neuron)]
            trace = data_row.trace.values.tolist()[0]
            smoothed_trace = savgol_filter(trace, window_length=5, polyorder=3)
            ax[row, col].plot(np.arange(0, 15, 1), smoothed_trace, color=resp_col[data_row.resp.values[0]])
            # Frame
            ax[row, col].spines[["right", "top", "bottom"]].set_visible(False)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

            ax[row, col].axhline(y=0, ls="--", lw=1, color="grey")
            if row == 0:
                trial_label = trial_name_dict[data_row.Behavior.values[0]]
                ax[row, col].set_title(f"Trial n°{trial}\n{trial_label}", fontsize=10)

            if col == 0:
                ax[row, col].set_ylabel(f"Neuron {neuron}", fontsize=10)
    plt.show()
    return data

neurons_corr_data = plot_neuronal_corr(frame_df, 7553, neurons=[1, 2, 3, 4, 5, 6, 7, 8], trials=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


if __name__ == '__main__':
    BMS_analysis = False
    ### Initialisation of recs instances ###
    if BMS_analysis:
        directory = "C:/Users/cvandromme/Desktop/Tactile_detection/Data_DMSO_BMS/"
        roi_path = "C:/Users/cvandromme/Desktop/Tactile_detection/Fmko_bms&dmso_info.xlsx"
    else:
        directory = "C:/Users/cvandromme/Desktop/Tactile_detection/Data/"
        roi_path = "C:/Users/cvandromme/Desktop/Tactile_detection/FmKO_ROIs&inhibitory.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/"
    roi_info = pd.read_excel(roi_path)
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]
    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec
    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    if BMS_analysis:
        recs = {f"{ar.get().filename}-{ar.get().genotype.split("-")[1]}": ar.get() for ar in async_results}
    else:
        recs = {ar.get().filename: ar.get() for ar in async_results}

    frame_df = get_activity_by_frame_df(recs.values(), zscore=True)

    # for rec in [recs[7553], recs[5890]]:
    # for rec in recs.values():
    #     if rec.genotype != "KO":
    #         mean_traces_df = plot_mean_neuronal_trace(frame_df, rec=rec.filename, n_type="INH", trial_avg=True, smoothing=10)
