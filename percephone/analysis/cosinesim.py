"""Théo Gauvrit 11/03/2024
Compute cosine similarity matrix between trials to decipher if there is more trial by trial variability in the
recruitment of neurons in the KO group"""

import numpy as np
import pandas as pd
import percephone.core.recording as pc
import percephone.plts.behavior as pbh
import matplotlib
import percephone.plts.stats as ppt
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity
import os
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")

directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]


def cosine_matrix(ax, rec, amplitude):
    print("Cosine similarity computation")
    rec.responsivity()
    # stim_mat = np.zeros((len(rec.stim_time, rec.stim_time)))
    # for i, stim in enumerate(rec.stim_time):

    sim_mat = cosine_similarity(np.transpose(rec.matrices["EXC"]["Responsivity"])[~rec.detected_stim])#[rec.stim_ampl == amplitude])
    h = ax.imshow(sim_mat, cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Trial i")
    ax.set_ylabel("Trial j")
    ax.set_title(str(rec.filename) + " " + rec.genotype + " " + str(amp))


def opening_rec(fil, i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0, fil, roi_info)
    return rec


# workers = cpu_count()
# pool = Pool(processes=workers)
# async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
# recs = [ar.get() for ar in async_results]
#
#
# y, i = 0, 0
# amps = [2, 6, 4, 4, 4, 8, 4, 4, 12, 8, 6, 12, 12]  # manual selection of the threshold amp for each animal from psychometric curves
# fig, ax = plt.subplots(4, 7, figsize=(35, 20))
#
# for rec, amp in zip(recs, amps):
#         if rec.genotype == "WT":
#             pbh.psycho_like_plot(rec, roi_info, ax[0, i])
#             cosine_matrix(ax[1, i], rec, amp)
#             i = i + 1
#         else:
#             pbh.psycho_like_plot(rec, roi_info, ax[2, y])
#             cosine_matrix(ax[3, y], rec, amp)
#             y = y + 1
#
# np.ravel(ax)[-1].set_axis_off()
# ax[2, 6].set_axis_off()
# fig.suptitle('Cosine similarity for undetected trials', fontsize=26)
#
#
