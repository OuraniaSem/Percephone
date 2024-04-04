"""
Théo Gauvrit 18/01/2024
plotting psychometric graph-like for behavior and detection
"""

import matplotlib
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.switch_backend("Qt5Agg")
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams["xtick.major.width"] = 3
plt.rcParams["ytick.major.width"] = 3

sampling_rate = 30.9609  # Hz
wt_color = "#326993"
ko_color = "#CC0000"


def psycho_like_plot(rec, roi_info, ax):
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    ax.plot([2, 4, 6, 8, 10, 12], converted_list)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])

def psycho_like_plot_and_synchro(rec, roi_info, ax):
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    to_plot = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        res = sum(rec.detected_stim[rec.stim_ampl == amp]) / len(rec.detected_stim[rec.stim_ampl == amp])
        to_plot.append(res)
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.plot([0, 2, 4, 6, 8, 10, 12], to_plot, linestyle='--')
    ax.set_xticks([0 ,2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])

