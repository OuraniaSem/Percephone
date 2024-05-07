"""
Théo Gauvrit 07/05/2024
Using a logistic regression to classify hit or miss from zcore time points
"""

from unittest import result
import numpy as np
import pandas as pd
import percephone.core.recording as pc
import percephone.plts.stats as ppt
import os
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, pool
import warnings
import seaborn as sns
import copy

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from scipy.stats import mannwhitneyu

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")
warnings.filterwarnings('ignore')
fontsize = 30


def classification_graph(time_accuracy, title):
    """plot the Hit vs miss classification graph like in Fig3 of Rowland et al """
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_accuracy)
    ax.set_ylim([0, 1])
    fig.suptitle(title, fontsize=fontsize)
    plt.show()


def frame_model(rec, frame, amp):
    X = np.transpose(rec.zscore_exc[:, rec.stim_time+frame])
    Y = rec.detected_stim
    accuracies = cross_val_score(LogisticRegression(penalty='l2', max_iter=5000), X, Y, cv=8)
    accuracy = accuracies.mean()
    return accuracy


if __name__ == '__main__':
    user = "Théo"

    if user == "Célien":
        directory = "C:/Users/cvandromme/Desktop/Data/"
        roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    elif user == "Théo":
        directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
        roi_path = directory + "/FmKO_ROIs&inhibitory.xlsx"

    roi_info = roi_path
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]


    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        rec.peak_delay_amp()
        return rec


    workers = cpu_count()
    if user == "Célien":
        pool = pool.ThreadPool(processes=workers)
    elif user == "Théo":
        pool = Pool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}

    # rec = recs[4456]
    for rec in recs.values():
        t_acc = []
        for i in list(range(-15, 15)):
            acc = frame_model(rec, i, 4)
            t_acc.append(acc)
        classification_graph(t_acc, rec.filename)

    # rec.zscore_exc[:, rec.stim_time[rec.stim_ampl == 12] - 15]
    # rec.detected_stim[rec.stim_ampl == 4]