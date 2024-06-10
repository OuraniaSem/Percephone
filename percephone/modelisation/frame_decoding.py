"""
Théo Gauvrit 07/05/2024
Using a logistic regression to classify hit or miss from zcore time points
"""

import numpy as np
import pandas as pd
import percephone.core.recording as pc
from percephone.analysis.utils import idx_resp_neur
import percephone.plts.stats as ppt
import os
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, pool
import warnings
import copy
import imblearn as imb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, sem

plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")
warnings.filterwarnings('ignore')
fontsize = 30


def classification_graph(hit_accuracy, miss_accuracy, title):
    """plot the Hit vs miss classification graph like in Fig3 of Rowland et al """
    hit_acc = np.nanmean(hit_accuracy, axis=0)
    miss_acc = np.nanmean(miss_accuracy, axis=0)
    y_err_hit = sem(hit_accuracy, axis=0, nan_policy="omit")
    y_err_miss = sem(miss_accuracy, axis=0, nan_policy="omit")
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    times = np.linspace(-1,1, int(2*30))
    ax.plot(times, hit_acc, label="Hit trials")
    ax.plot(times, miss_acc, label="Miss trials")
    ax.fill_between(times, hit_acc - y_err_hit,  hit_acc + y_err_hit, alpha=0.2)
    ax.fill_between(times, miss_acc - y_err_miss, miss_acc + y_err_miss, alpha=0.2,)
    ax.set_ylabel("Hit versus Miss classification")
    ax.set_xlabel("Time (s)")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.vlines(0, ymin=0, ymax=1, linestyle="--", color="red")
    ax.legend( fontsize=15)
    ax.set_title(title, fontsize=fontsize)
    fig.tight_layout()


def split_data(rec, frame, train_ratio=0.8, stratify=False, seed=None, neurons="all"):
    id_exc_act, id_exc_inh = idx_resp_neur(rec, n_type="EXC")
    id_inh_act, id_inh_inh = idx_resp_neur(rec, n_type="INH")
    full_exc_id = np.arange(rec.df_f_exc.shape[0])
    full_inh_id = np.arange(rec.df_f_inh.shape[0])
    id_exc_dict = {"all": full_exc_id, "activated": id_exc_act, "inhibited": id_exc_inh, "both": np.concatenate([id_exc_act, id_exc_inh])}
    id_inh_dict = {"all": full_inh_id, "activated": id_inh_act, "inhibited": id_inh_inh, "both": np.concatenate([id_inh_act, id_inh_inh])}
    exc_filter = np.isin(full_exc_id, id_exc_dict[neurons])
    inh_filter = np.isin(full_inh_id, id_inh_dict[neurons])
    # print(f"EXC : {id_exc_dict[neurons].shape[0]/ full_exc_id.shape[0]:^5.1%} ({id_exc_dict[neurons].shape[0]:^3}/{full_exc_id.shape[0]:^3}) - INH : {id_inh_dict[neurons].shape[0]/ full_inh_id.shape[0]:^5.1%} ({id_inh_dict[neurons].shape[0]:^3}/{full_inh_id.shape[0]:^3})")

    filtered_exc_dff = rec.df_f_exc[exc_filter]
    filtered_inh_dff = rec.df_f_inh[inh_filter]
    record_dict= {}
    record_dict["X"] = np.row_stack((filtered_exc_dff[:, rec.stim_time+frame], filtered_inh_dff[:, rec.stim_time+frame])).T
    record_dict["y"] = rec.detected_stim
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
    record_dict["X_bal"], record_dict["y_bal"] = resampler.fit_resample(record_dict["X_train"], record_dict["y_train"])
    return record_dict


def frame_model(rec, frame, resampler, neurons="all"):

    r_dict = split_data(rec, frame, train_ratio=0.8, stratify=False, seed=None, neurons=neurons)
    model = LogisticRegression(penalty='l2', max_iter=5000)
    r_dict = resample(r_dict,  resampler)
    model.fit(r_dict["X_bal"], r_dict["y_bal"])
    y_pred = model.predict(r_dict["X_test"])
    conf_matrix = confusion_matrix(r_dict["y_test"], y_pred, labels=[False, True])
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    hit_acc = TP / (TP + FN)
    miss_acc2 = FP/(FP + TN)
    miss_acc = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return (hit_acc, miss_acc2, accuracy)


if __name__ == '__main__':
    user = "Célien"
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
    resampler = ros

    neurons = "activated"

# per group
    wt_hit, wt_miss, ko_hypo_hit, ko_hypo_miss = [], [], [], []
    for rec in recs.values():
        try:
            print(f"{rec.filename} ({rec.genotype})")
            acc_hit, acc_miss = [], []
            for i in list(range(-30, 30)):
                acc = frame_model(rec, i, resampler, neurons=neurons)
                acc_hit.append(acc[0])
                acc_miss.append(acc[1])
            if rec.genotype == "WT":
                wt_hit.append(acc_hit)
                wt_miss.append(acc_miss)
            elif rec.genotype == "KO-Hypo":
                ko_hypo_hit.append(acc_hit)
                ko_hypo_miss.append(acc_miss)
        except ValueError:
            print(f"{rec.filename} -> Failed")
            continue

    classification_graph(wt_hit, wt_miss, f"WT ({resampler}/{neurons})")
    classification_graph(ko_hypo_hit, ko_hypo_miss, f"KO-Hypo ({resampler}/{neurons})")

    # for f in [6601, 6606, 6609, 6611,]:
    #     recs[f].responsivity()
    plt.show()
    print("Done")
