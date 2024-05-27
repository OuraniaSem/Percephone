"""
04 MARCH 2024
Theo Gauvrit
Testing the higher baseline hypothesis to explain the no detection of tactile stimulus on KO mice.
"""

import numpy as np
import pandas as pd
import percephone.core.recording as pc
import os
import percephone.plts.behavior as pbh
import matplotlib
import percephone.plts.stats as ppt
import matplotlib.pyplot as plt
import percephone.analysis.mlr_models as mlr_m
from multiprocessing import Pool, cpu_count
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")

directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
roi_info = directory + "/FmKO_ROIs&inhibitory.xlsx"
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]

def opening_rec(fil,i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_info)
    return rec

workers = cpu_count()
pool = Pool(processes=workers)
async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
recs = {ar.get().filename: ar.get() for ar in async_results}

from matplotlib.widgets import Button, Slider
rec = recs[4939]
import percephone.plts.response as rr

# rr.superimposed_response(rec, rec.zscore_exc, rec.stim_time[2])

fig, ax = plt.subplots(1, 1, figsize=(35, 20))
fig.subplots_adjust(left=0.25, bottom=0.25)
axslide = fig.add_axes([0.25, 0.1, 0.65, 0.03])
amp = 10
stims = rec.stim_time[rec.stim_ampl == amp]
# rec.responsivity()
idx_responsive = np.where(np.sum(rec.matrices["EXC"]["Responsivity"], axis=1))

# from percephone.plts.heatmap import intereactive_heatmap
# intereactive_heatmap(rec, rec.zscore_exc)

def cor_map(val):

    val = int(val)
    ax.cla()
    exc = rec.zscore_exc[:,
          np.linspace(stims[val] - int(1 * rec.sf), stims[val], int(1 * rec.sf), dtype=int)]

    corr = np.corrcoef(exc)
    Z = linkage(corr, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    order_s = dn_exc["leaves"]
    ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.detected_stim[rec.stim_ampl == amp][val]) + "  " + str(rec.stim_ampl[rec.stim_ampl ==amp][val]))

cor_map(0)
trial_slider = Slider(
    ax=axslide,
    label='Trial',
    valmin=0,
    valmax=len(stims)-1,
    valstep=np.linspace(0, len(stims), len(stims)+1),
    valinit=0,
)
trial_slider.on_changed(cor_map)
