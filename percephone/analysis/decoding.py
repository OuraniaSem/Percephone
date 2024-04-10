"""
Théo Gauvrit 18/01/2024
decoding of the behaviors labels from neurons activity
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import percephone.core.recording as pc
import os
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import sklearn.model_selection as ms
import random as rnd
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")

directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]


def opening_rec(fil,i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0, fil, roi_info)
    return rec

workers = cpu_count()
pool = Pool(processes=workers)
async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
recs = {ar.get().filename: ar.get() for ar in async_results}

# predicting if the stim will be detected or not by the animal from the response vector
wt_ = []
ko_ = []
for rec in recs.values():
    # rec = recs[4756]
    resp = np.concatenate([rec.matrices["EXC"]["Responsivity"], rec.matrices["INH"]["Responsivity"]])
    exc = rec.zscore_exc[:,
          np.linspace(rec.stim_time - int(1 * rec.sf), rec.stim_time,
                      int(1 * rec.sf), dtype=int)]
    exc_ = np.std(exc, axis=1)
    X = np.array(np.transpose(resp))
    accuracies = ms.cross_val_score(LogisticRegression(penalty='l2', max_iter=5000), X, rec.detected_stim, cv=8)
    f, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(accuracies, vert=False, widths=.7)
    ax.scatter(accuracies, np.ones(8))
    ax.set(
        xlabel="Accuracy",
        yticks=[],
        title=f"Average test accuracy: {accuracies.mean():.2%} + {rec.filename}"
    )
    ax.spines["left"].set_visible(False)
    plt.show()
    if rec.genotype == "WT":
        wt_.append(accuracies.mean())
    elif rec.genotype == "KO-Hypo":
        ko_.append(accuracies.mean())
        print(rec.filename)
        print(accuracies.mean())
#     training_proportion = int(0.7*len(X))
#     rnd_trials = rnd.sample(range(len(X)), k=training_proportion)
#     train_label = rec.detected_stim[rnd_trials]
#     train_set = X[rnd_trials]
#     log_reg = LogisticRegression(penalty="l2", max_iter=5000)
#     log_reg.fit(train_set, train_label)
#     test_idx = list(set(set(range(len(X)))).difference(rnd_trials))
#     test_set = X[test_idx]
#     test_labels = rec.detected_stim[test_idx]
#     y_pred = log_reg.predict(test_set)
#     accuracy = (test_labels == y_pred).sum() / len(test_labels)
#     print(rec.filename)
#     if rec.genotype == "WT":
#         wt_.append(accuracy)
#     elif rec.genotype == "KO-Hypo":
#         ko_.append(accuracy)
#         print(rec.filename)
#         print(accuracy)
from percephone.plts.stats import boxplot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
boxplot(ax, np.array(wt_), np.array(ko_), "accuracy", ylim=[0, 1])


