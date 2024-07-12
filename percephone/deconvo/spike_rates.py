
"""
4 JULY 2024
Theo Gauvrit
Use the deconvolution spiking data(spks.npy) to get teh spontaneous firing rate comparaison"""

import percephone.core.recording as pc
import percephone.plts.behavior as pbh
import percephone.plts.stats as ppt
from percephone.analysis.utils import get_zscore
import percephone.analysis.mlr_models as mlr_m
import numpy as np
import pandas as pd
import scipy.stats as ss
from multiprocessing import Pool, cpu_count, pool
import os
import matplotlib
import matplotlib.pyplot as plt
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib as mpl


plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")
matplotlib.rc('pdf', fonttype=42)
save_figure = True
user = "Théo"

if user == "Célien":
    directory = "C:/Users/cvandromme/Desktop/Data/"
    roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper/"
elif user == "Théo":
    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
    roi_path = directory + "/FmKO_ROIs&inhibitory.xlsx"
    server_address = "/run/user/1004/gvfs/smb-share:server=engram.local,share=data/Current_members/Ourania_Semelidou/2p/Figures_paper/"
elif user == "Adrien":
    directory = "C:/Users/acorniere/Desktop/percephone_data/"
    roi_path = directory + "FmKO_ROIs&inhibitory.xlsx"

roi_info = pd.read_excel(roi_path)
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]


def opening_rec(fil, i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path, mean_f=True, cache=True)
    return rec


workers = cpu_count()
if user == "Célien":
    pool = pool.ThreadPool(processes=workers)
elif user == "Théo":
   pool = Pool(processes=workers)
elif user == "Adrien":
    pool = pool.ThreadPool(processes=workers)
async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
recs = {ar.get().filename: ar.get() for ar in async_results}

# prestimulus
wt, ko = [], []
for rec in recs.values():
    print(rec.filename)
    idx_bsl = np.linspace(rec.stim_time-int(0.5*rec.sf), rec.stim_time, int(0.5*rec.sf)+1, dtype=int)
    spks_baseline = np.mean(np.mean(np.sum(rec.spks_exc[:, idx_bsl], axis=1), axis=0))
    if rec.genotype == "WT":
        wt.append(spks_baseline)
    elif rec.genotype == "KO-Hypo":
        ko.append(spks_baseline)
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
ppt.boxplot(ax, wt, ko, f"A.U deconv", ylim=[0, 250])
fig.tight_layout()

# supra/sub WT threshold prestim
wt_supra_hit, wt_supra_miss, ko_supra_hit, ko_supra_miss = [], [], [], []
for rec in recs.values():
    print(rec.filename)
    stim_idx_hit = rec.stim_time[rec.detected_stim & (rec.stim_ampl > 7)]
    stim_idx_miss = rec.stim_time[~rec.detected_stim & (rec.stim_ampl > 7)]

    idx_bsl_hit = np.linspace( stim_idx_hit - int(0.5*rec.sf),  stim_idx_hit, int(0.5*rec.sf)+1, dtype=int)
    idx_bsl_miss = np.linspace(stim_idx_miss - int(0.5*rec.sf), stim_idx_miss, int(0.5*rec.sf)+1, dtype=int)
    spks_supra_hit = np.mean(np.mean(np.sum(rec.spks_exc[:,  idx_bsl_hit], axis=1), axis=0))
    spks_supra_miss = np.mean(np.mean(np.sum(rec.spks_exc[:, idx_bsl_miss], axis=1), axis=0))
    if rec.genotype == "WT":
        wt_supra_hit.append(spks_supra_hit)
        wt_supra_miss.append(spks_supra_miss)
    elif rec.genotype == "KO-Hypo":
        ko_supra_hit.append(spks_supra_hit)
        ko_supra_miss.append(spks_supra_miss)
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
ppt.paired_boxplot(axs[0], wt_supra_hit, wt_supra_miss, f"A.U deconv", title="", ylim=[0, 250])
ppt.paired_boxplot(axs[1],  ko_supra_hit,  ko_supra_miss, f"A.U deconv", title="", ylim=[0, 250])
fig.tight_layout()

# supra/sub WT threshold stim
wt_supra_hit, wt_supra_miss, ko_supra_hit, ko_supra_miss = [], [], [], []
for rec in recs.values():
    stim_idx_hit = rec.stim_time[rec.detected_stim ]
    stim_idx_miss = rec.stim_time[~rec.detected_stim ]

    idx_bsl_hit = np.linspace( stim_idx_hit,  stim_idx_hit + int(0.5*rec.sf), int(0.5*rec.sf)+1, dtype=int)
    idx_bsl_miss = np.linspace(stim_idx_miss, stim_idx_miss + int(0.5*rec.sf), int(0.5*rec.sf)+1, dtype=int)
    spks_supra_hit = np.mean(np.mean(np.mean(rec.spks_exc[:,  idx_bsl_hit], axis=1), axis=0))
    spks_supra_miss = np.mean(np.mean(np.mean(rec.spks_exc[:, idx_bsl_miss], axis=1), axis=0))
    if rec.genotype == "WT":
        wt_supra_hit.append(spks_supra_hit)
        wt_supra_miss.append(spks_supra_miss)
        print(rec.filename)
    elif rec.genotype == "KO-Hypo":
        ko_supra_hit.append(spks_supra_hit)
        ko_supra_miss.append(spks_supra_miss)

fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharey="all")
ppt.paired_boxplot(axs[0], wt_supra_hit, wt_supra_miss, f"A.U deconv", title="", ylim=[0, 20])
ppt.paired_boxplot(axs[1],  ko_supra_hit,  ko_supra_miss, f"A.U deconv", title="", ylim=[0, 20])
fig.tight_layout()

