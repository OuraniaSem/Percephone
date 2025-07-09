import os

import numpy as np
import pandas as pd
from multiprocessing import cpu_count, pool

from matplotlib import pyplot as plt

import percephone.core.recording as pc
import percephone.plts.stats as ppt
from Figures.response_decoding import get_activity_by_frame_df


def plot_mean_neuronal_trace(frame_df, rec=None, n_type="EXC", pattern=1):
    """Plots the mean neuronal traces for hit and miss trials for all recs"""
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color]}
    data = frame_df[(frame_df.ID == rec) & (frame_df.n_type == n_type)].copy()
    genotype = data.Genotype.values[0]
    data = data[data.resp == pattern]
    data = data[data.Amplitude != 0].drop(columns=["Genotype", "ID", "Threshold", "Duration", "Amplitude", "n_type", "resp", "n_ID"])
    # Without averaging per trial
    mean_traces = data.groupby(["Behavior"]).mean().drop(columns=["Trial"])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    x = mean_traces.columns.astype(int)
    y_hit = mean_traces.loc[True]
    y_miss = mean_traces.loc[False]
    ax.plot(x, y_hit, label='Hit', color=color_dict[genotype][0])
    ax.plot(x, y_miss, label='Miss', color=color_dict[genotype][1])
    ax.axvline(x=30, ls="--", lw=1, color="red")
    ax.axvline(x=45, ls="--", lw=1, color="black")
    ax.set_title(genotype, color=color_dict[genotype][0], fontsize=20)
    ax.set_xlabel("Time (s)")
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xticklabels([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Z-score (Mean)")
    plt.show()
    return mean_traces


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

    frame_dff = get_activity_by_frame_df(recs.values(), zscore=True)

    mean_traces_df = plot_mean_neuronal_trace(frame_dff, rec=7553, n_type="EXC", pattern=1)