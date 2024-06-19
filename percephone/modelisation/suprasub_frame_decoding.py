"""
Théo Gauvrit 22/05/2024
Using a logistic regression to classify supra hit, supra miss, sub hit, sub miss
"""

import numpy as np
import pandas as pd
import percephone.core.recording as pc
import percephone.plts.stats as ppt
import os
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, pool
import warnings
import copy
import imblearn as imb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, sem

plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")
warnings.filterwarnings('ignore')
fontsize = 30


def classification_graph_group(accus, title):
    """plot the Hit vs miss classification graph like in Fig3 of Rowland et al """
    y_err_hit_sup  = sem(accus[0], axis=1, nan_policy="omit")
    y_err_miss_sup = sem(accus[1], axis=1, nan_policy="omit")
    y_err_hit_sub  = sem(accus[2], axis=1, nan_policy="omit")
    y_err_miss_sub = sem(accus[3], axis=1, nan_policy="omit")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    times = np.linspace(-1, 1, int(2*30))
    # ax.plot(times, np.nanmean(accus[0], axis=1), label="Supra hit trials")
    ax.plot(times, np.nanmean(accus[1], axis=1), label="Supra miss trials")
    # ax.plot(times, np.nanmean(accus[2], axis=1), label="Sub hit trials")
    ax.plot(times, np.nanmean(accus[3], axis=1), label="Sub miss trials")
    # ax.fill_between(times, np.nanmean(accus[0], axis=1) - y_err_hit_sup,  np.nanmean(accus[0], axis=1) + y_err_hit_sup, alpha=0.2)
    ax.fill_between(times, np.nanmean(accus[1], axis=1) - y_err_miss_sup, np.nanmean(accus[1], axis=1) + y_err_miss_sup, alpha=0.2,)
    # ax.fill_between(times, np.nanmean(accus[2], axis=1) - y_err_hit_sub,  np.nanmean(accus[2], axis=1) + y_err_hit_sub, alpha=0.2)
    ax.fill_between(times, np.nanmean(accus[3], axis=1) - y_err_miss_sub, np.nanmean(accus[3], axis=1) + y_err_miss_sub, alpha=0.2, )
    ax.set_ylabel("Hit versus Miss classification")
    ax.set_xlabel("Time (s)")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.vlines(0, ymin=0, ymax=1, linestyle="--", color="red")
    ax.legend( fontsize=15)
    ax.set_title(title, fontsize=fontsize)
    fig.tight_layout()


def classification_graph(accus, title):
    """plot the Hit vs miss classification graph like in Fig3 of Rowland et al """
    # y_err_hit_sup  = sem(accus[0], axis=1, nan_policy="omit")
    # y_err_miss_sup = sem(accus[1], axis=1, nan_policy="omit")
    # y_err_hit_sub  = sem(accus[2], axis=1, nan_policy="omit")
    # y_err_miss_sub = sem(accus[3], axis=1, nan_policy="omit")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    times = np.linspace(-1, 1, int(2*30))
    # ax.plot(times, np.nanmean(accus[0], axis=1), label="Supra hit trials")
    ax.plot(times, accus[:,1], label="Supra miss trials")
    # ax.plot(times, np.nanmean(accus[2], axis=1), label="Sub hit trials")
    ax.plot(times, accus[:,3], label="Sub miss trials")
    # ax.fill_between(times, np.nanmean(accus[0], axis=1) - y_err_hit_sup,  np.nanmean(accus[0], axis=1) + y_err_hit_sup, alpha=0.2)
    # ax.fill_between(times, np.nanmean(accus[1], axis=1) - y_err_miss_sup, np.nanmean(accus[1], axis=1) + y_err_miss_sup, alpha=0.2,)
    # ax.fill_between(times, np.nanmean(accus[2], axis=1) - y_err_hit_sub,  np.nanmean(accus[2], axis=1) + y_err_hit_sub, alpha=0.2)
    # ax.fill_between(times, np.nanmean(accus[3], axis=1) - y_err_miss_sub, np.nanmean(accus[3], axis=1) + y_err_miss_sub, alpha=0.2, )
    ax.set_ylabel("Hit versus Miss classification")
    ax.set_xlabel("Time (s)")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.vlines(0, ymin=0, ymax=1, linestyle="--", color="red")
    ax.legend(fontsize=15)
    ax.set_title(title, fontsize=fontsize)
    fig.tight_layout()


def split_data(rec, frame, train_ratio=0.8, stratify=False, seed=None):
    record_dict= {}
    record_dict["X"] = np.row_stack((rec.df_f_exc[:, rec.stim_time+frame], rec.df_f_inh[:, rec.stim_time+frame])).T
    record_dict["y"] = np.zeros(len(rec.detected_stim))
    temp_y = rec.detected_stim
    record_dict["y"][np.logical_and(temp_y, rec.stim_ampl > 6)] = 1
    record_dict["y"][np.logical_and(~temp_y, rec.stim_ampl > 6)] = 2
    record_dict["y"][np.logical_and(temp_y, rec.stim_ampl <= 6)] = 3
    record_dict["y"][np.logical_and(~temp_y, rec.stim_ampl <= 6)] = 4
    print(record_dict["y"])
    if stratify:
        record_dict["X_train"], record_dict["X_test"], record_dict["y_train"], record_dict["y_test"] = train_test_split(record_dict["X"], record_dict["y"],
                                                                                                                        train_size=train_ratio,
                                                                                                                        stratify=record_dict["y"],
                                                                                                                        random_state=seed)
    else:
        record_dict["X_train"], record_dict["X_test"], record_dict["y_train"], record_dict["y_test"] = train_test_split(record_dict["X"], record_dict["y"],
                                                                                                                        train_size=train_ratio,
                                                                                                                        stratify=None,
                                                                                                                        random_state=seed)
    return record_dict


def resample(record_dict, resampler):
    if resampler == None:
        return record_dict
    record_dict["X_bal"], record_dict["y_bal"] = resampler.fit_resample(record_dict["X_train"], record_dict["y_train"])
    return record_dict


def frame_model(rec, frame, resampler):

    r_dict = split_data(rec, frame, train_ratio=0.8, stratify=True, seed=None)
    model = LogisticRegression(penalty='l2', max_iter=5000)
    # ros = imb.over_sampling.RandomOverSampler(sampling_strategy='auto', shrinkage=None)
    # smote = imb.over_sampling.SMOTE(sampling_strategy='auto')
    # adasyn = imb.over_sampling.ADASYN(sampling_strategy='auto')
    r_dict = resample(r_dict,  resampler)
    model.fit(r_dict["X_train"], r_dict["y_train"])
    y_pred = model.predict(r_dict["X_test"])
    print(y_pred)
    print(r_dict["y_test"])
    conf_matrices = multilabel_confusion_matrix(r_dict["y_test"], y_pred, labels=[1, 2, 3, 4])
    accuracies = []
    for conf in conf_matrices:
        TP = conf[1, 1]
        TN = conf[0, 0]
        FP = conf[0, 1]
        FN = conf[1, 0]
        hit_acc = TP / (TP + FN)
        miss_acc2 = FP / (FP + TN)
        miss_acc = TN / (TN + FP)
        accuracies.append((TP + TN) / (TP + TN + FP + FN))
    return accuracies


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
        print(fil)
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec

    workers = cpu_count()
    if user == "Célien":
        pool = pool.ThreadPool(processes=workers)
    elif user == "Théo":
        pool = Pool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}

    ## for single recording only
    # for rec in recs.values():
    #     acc_hit, acc_miss = [], []
    #     for i in list(range(-15, 15)):
    #         acc = frame_model(rec, i)
    #         acc_hit.append(acc[0])
    #         acc_miss.append(acc[1])
    #     classification_graph(acc_hit, acc_miss, rec.filename)

    ros = imb.over_sampling.RandomOverSampler(sampling_strategy='auto')
    smote = imb.over_sampling.SMOTE(sampling_strategy='auto', k_neighbors=4)
    adasyn = imb.over_sampling.ADASYN(sampling_strategy='auto')
    rus = imb.under_sampling.RandomUnderSampler(sampling_strategy='auto')
    resampler = rus

# per group
    wt_acc, ko_hypo_acc = [], []
    for rec in recs.values():
        print(rec.filename)
        acc = []
        for i in list(range(-30, 30)):
            acc_s = frame_model(rec, i, resampler)
            acc.append(acc_s)

        if rec.genotype == "WT":
            wt_acc.append(acc)
        elif rec.genotype == "KO-Hypo":
            ko_hypo_acc.append(acc)
        classification_graph(np.array(acc), f"{rec.filename} - {rec.genotype}")
    # classification_graph(np.array(wt_acc), f"WT")
    # classification_graph(np.array(ko_hypo_acc), f"KO-Hypo")
