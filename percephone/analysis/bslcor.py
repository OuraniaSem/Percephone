"""Théo Gauvrit 01/03/2024
IS the prestimulus activity impacting the detection ?
"""
import numpy as np
import pandas as pd
import percephone.core.recording as pc
import os
import percephone.plts.behavior as pbh
import matplotlib
import percephone.plts.stats as ppt
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, pool
import percephone.plts.heatmap as hm
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")

user = "Théo"

if user == "Célien":
    directory = "C:/Users/cvandromme/Desktop/Data/"
    roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
elif user == "Théo":
    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
    roi_path = directory + "/FmKO_ROIs&inhibitory.xlsx"

roi_info = pd.read_excel(roi_path)
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]


def opening_rec(fil, i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path, mean_f=False, cache=False)

    return rec


workers = cpu_count()
if user == "Célien":
    pool = pool.ThreadPool(processes=workers)
elif user == "Théo":
    pool = Pool(processes=workers)
async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
recs = {ar.get().filename: ar.get() for ar in async_results}


def prestim_activity(n_type, ko):
    wt_det, wt_undet, ko_det, ko_undet = [], [], [], []
    for rec in recs.values():
        t_points = rec.stim_time[rec.detected_stim & (rec.stim_ampl == rec.threshold)]
        t_points_u = rec.stim_time[~rec.detected_stim & (rec.stim_ampl == rec.threshold)]
        bsl_n_det = np.mean(np.mean(np.mean(rec.df_f_exc[:, np.linspace(t_points-15, t_points, 15, dtype=int)], axis=1), axis=1))
        bsl_n_undet = np.mean(np.mean(np.mean(rec.df_f_exc[:, np.linspace(t_points_u-15, t_points_u, 15, dtype=int)], axis=1), axis=1))
        if n_type== "INH":
            bsl_n_det = np.mean(
                np.mean(np.mean(rec.df_f_inh[:, np.linspace(t_points - 15, t_points, 15, dtype=int)], axis=1), axis=1))
            bsl_n_undet = np.mean(
                np.mean(np.mean(rec.df_f_inh[:, np.linspace(t_points_u - 15, t_points_u, 15, dtype=int)], axis=1),
                        axis=1))

        if rec.genotype == "WT":
            wt_det.append(bsl_n_det)
            wt_undet.append(bsl_n_undet)
        elif rec.genotype =="KO-Hypo":
            ko_det.append(bsl_n_det)
            ko_undet.append(bsl_n_undet)
        elif rec.genotype == "KO" and ko =="KO":
            ko_det.append(bsl_n_det)
            ko_undet.append(bsl_n_undet)
    return wt_det, ko_det, wt_undet, ko_undet


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, type in enumerate(["EXC", "INH"]):
    wt_det, ko_det, wt_undet, ko_undet = prestim_activity(n_type=type, ko="KO-Hypo")
    ppt.paired_boxplot(axs[i, 0], wt_det, wt_undet, "Var DF/F", "", ylim=[-2, 2],
                       colors=[ppt.wt_color, ppt.light_wt_color])
    ppt.paired_boxplot(axs[i, 1], ko_det, ko_undet, "Var DF/F", "", ylim=[-2, 2])
    fig.suptitle("Comparaison of neurons activated between detected and undetected for all stimulus")
    fig.tight_layout()

