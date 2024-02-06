

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.signal as ssig
import scipy.ndimage as sn
import matplotlib.pyplot as plt
import percephone.core.recording as pc
import math
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams["xtick.major.width"] = 3
plt.rcParams["ytick.major.width"] = 3
wt_color = "#326993"
light_wt_color = "#8db7d8"
ko_color = "#CC0000"
light_ko_color = "#ff8080"
matplotlib.use("Qt5Agg")
plt.switch_backend("Qt5Agg")


def superimposed_response(rec, zscore, onset_stim):

    fig, ax = plt.subplots(figsize=(12, 8))
    time = np.linspace(-1,0.5,int(rec.sf)+int(rec.sf/2))
    data_sliced = zscore[:,onset_stim-int(rec.sf):onset_stim+int(rec.sf/2)]
    y = np.mean(data_sliced , axis=0)
    y_err = np.std(data_sliced, axis=0)/math.sqrt(len(data_sliced))
    # y_err = time.std() * np.sqrt(1 / len(time) +
    #                           (time - time.mean()) ** 2 / np.sum((time - time.mean()) ** 2))
    y_f = sn.gaussian_filter(y, sigma=0.5)
    ax.plot(time, y_f, color=wt_color)
    ax.fill_between(time, y_f - y_err, y_f + y_err, alpha=0.2)
    ax.grid(False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([-1, 0, 0.5])
    ax.set_ylim([-1,1])
    ax.axvline(0, 0, 1, color='#ffc266', linestyle='--', lw = 4, label = 'stim')
    plt.legend(loc='upper right', fontsize=25, frameon=False)
    fig.tight_layout()
    plt.show()


