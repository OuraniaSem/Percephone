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
import scipy.stats as ss
import imblearn as imb
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, sem, pearsonr
from sklearn.linear_model import LinearRegression

plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")
warnings.filterwarnings('ignore')
fontsize = 30
server_address = "/run/user/1004/gvfs/smb-share:server=engram.local,share=data/Current_members/Ourania_Semelidou/2p/Figures_paper/"


def classification_graph(hit_accuracy, miss_accuracy, title, colors):
    """plot the Hit vs miss classification graph like in Fig3 of Rowland et al """
    hit_acc = np.nanmean(hit_accuracy, axis=0)
    miss_acc = np.nanmean(miss_accuracy, axis=0)
    y_err_hit = sem(hit_accuracy, axis=0, nan_policy="omit")
    y_err_miss = sem(miss_accuracy, axis=0, nan_policy="omit")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    times = np.linspace(-1, 1, int(2 * 30))
    ax.plot(times, hit_acc, label="Hit trials", color=colors[0])
    ax.plot(times, miss_acc, label="Miss trials", color=colors[1])
    ax.fill_between(times, hit_acc - y_err_hit, hit_acc + y_err_hit, alpha=0.2, color=colors[1])
    ax.fill_between(times, miss_acc - y_err_miss, miss_acc + y_err_miss, alpha=0.2, color=colors[1])
    ax.set_ylabel("Hit versus Miss classification")
    ax.set_xlabel("Time (s)")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.vlines(0, ymin=0, ymax=1, linestyle="--", color="red")
    ax.vlines(0.5, ymin=0, ymax=1, linestyle="--", color="black")
    ax.legend(fontsize=15)
    ax.set_title(title, fontsize=fontsize, pad=40)
    fig.tight_layout()
    fig.savefig(server_address + "Figure2/modelling/" + title + ".pdf")


def split_data(rec, frame, train_ratio=0.8, stratify=False, seed=None, neurons="all"):
    id_exc_act, id_exc_inh = idx_resp_neur(rec, n_type="EXC")
    id_inh_act, id_inh_inh = idx_resp_neur(rec, n_type="INH")
    full_exc_id = np.arange(rec.zscore_exc.shape[0])
    full_inh_id = np.arange(rec.zscore_inh.shape[0])
    id_exc_dict = {"all": full_exc_id, "activated": id_exc_act, "inhibited": id_exc_inh,
                   "both": np.concatenate([id_exc_act, id_exc_inh])}
    id_inh_dict = {"all": full_inh_id, "activated": id_inh_act, "inhibited": id_inh_inh,
                   "both": np.concatenate([id_inh_act, id_inh_inh])}
    exc_filter = np.isin(full_exc_id, id_exc_dict[neurons])
    inh_filter = np.isin(full_inh_id, id_inh_dict[neurons])
    # print(f"EXC : {id_exc_dict[neurons].shape[0]/ full_exc_id.shape[0]:^5.1%} ({id_exc_dict[neurons].shape[0]:^3}/{full_exc_id.shape[0]:^3}) - INH : {id_inh_dict[neurons].shape[0]/ full_inh_id.shape[0]:^5.1%} ({id_inh_dict[neurons].shape[0]:^3}/{full_inh_id.shape[0]:^3})")

    filtered_exc_dff = rec.df_f_exc[exc_filter]
    filtered_inh_dff = rec.df_f_inh[inh_filter]
    record_dict = {}
    record_dict["X"] = np.row_stack(
        (filtered_exc_dff[:, rec.stim_time + frame], filtered_inh_dff[:, rec.stim_time + frame])).T
    record_dict["y"] = rec.detected_stim
    if train_ratio == 1:
        record_dict["X_train"], record_dict["X_test"], record_dict["y_train"], record_dict["y_test"] = record_dict[
            "X"], [], record_dict["y"], []
    else:
        if stratify:
            record_dict["X_train"], record_dict["X_test"], record_dict["y_train"], record_dict[
                "y_test"] = train_test_split(
                record_dict["X"], record_dict["y"],
                train_size=train_ratio,
                stratify=record_dict["y"],
                random_state=seed)
        else:
            record_dict["X_train"], record_dict["X_test"], record_dict["y_train"], record_dict[
                "y_test"] = train_test_split(
                record_dict["X"], record_dict["y"],
                train_size=train_ratio,
                stratify=None,
                random_state=seed)
    return record_dict


def random_over_under(X_train, y_train):
    n_class_1 = sum(y_train)
    n_class_2 = len(y_train) - sum(y_train)
    if n_class_1 == n_class_2:
        return X_train, y_train
    elif n_class_1 > n_class_2:
        n_min = n_class_2
        n_maj = n_class_1
    else:
        n_min = n_class_1
        n_maj = n_class_2

    alpha_over = ((n_min + n_maj) / 2) / n_maj
    ros = imb.over_sampling.RandomOverSampler(sampling_strategy=alpha_over)
    rus = imb.under_sampling.RandomUnderSampler(sampling_strategy=1)
    X_bal_0, y_bal_0 = ros.fit_resample(X_train, y_train)
    X_bal, y_bal = rus.fit_resample(X_bal_0, y_bal_0)
    print(f"n_maj: {n_maj}, n_min: {n_min}, alpha_over: {alpha_over:.2f}, y_bal_0: {len(y_bal_0)}, y_bal: {len(y_bal)}")
    return X_bal, y_bal


def resample(record_dict, resampler):
    if resampler == "ROS-RUS":
        record_dict["X_bal"], record_dict["y_bal"] = random_over_under(record_dict["X_train"], record_dict["y_train"])
    record_dict["X_bal"], record_dict["y_bal"] = resampler.fit_resample(record_dict["X_train"], record_dict["y_train"])
    return record_dict


def get_sen_spe_acc(conf_matrix):
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = FP / (FP + TN)  # changed from dec_imbal_rep to fit graph representation
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return sensitivity, specificity, accuracy


def crossval_sen_spe_acc(model, X, y, kfold=5, stratify=True, shuffle=True, resampler=None, verbose=0):
    sensitivities = []
    specificities = []
    accuracies = []
    if stratify:
        kf = StratifiedKFold(n_splits=kfold, shuffle=shuffle)
    else:
        kf = KFold(n_splits=kfold, shuffle=shuffle)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if resampler == "ROS-RUS":
            X_bal, y_bal = random_over_under(X_train, y_train)
            model.fit(X_bal, y_bal)
        elif resampler is not None:
            X_bal, y_bal = resampler.fit_resample(X_train, y_train)
            if verbose != 0:
                print(
                    f"y_bal: {len(y_bal)} -> {len(y_bal) / (len(y_train) + len(y_test)) * 100:.1f}%, y_test: {len(y_test)} -> {len(y_test) / (len(y_train) + len(y_test)) * 100:.1f}%")
            model.fit(X_bal, y_bal)
        else:
            if verbose != 0:
                print(
                    f"y_train: {len(y_train)} -> {len(y_train) / (len(y_train) + len(y_test)) * 100:.1f}%, y_test: {len(y_test)} -> {len(y_test) / (len(y_train) + len(y_test)) * 100:.1f}%")
            model.fit(X_train, y_train)

        conf_mat = confusion_matrix(y_test, model.predict(X_test), labels=[False, True])
        sen, spe, acc = get_sen_spe_acc(conf_mat)
        sensitivities.append(sen)
        specificities.append(spe)
        accuracies.append(acc)
    return np.nanmean(sensitivities), np.nanmean(specificities), np.nanmean(accuracies)


def frame_model(rec, frame, resampler, cv=False, neurons="all"):
    model = LogisticRegression(penalty='l2', max_iter=5000)
    if cv:
        r_dict = split_data(rec, frame, train_ratio=1, stratify=True, seed=None, neurons=neurons)

        hit_acc, miss_acc2, accuracy = crossval_sen_spe_acc(model, r_dict["X"], r_dict["y"], kfold=cv,
                                                            resampler=resampler, verbose=0)
    else:
        r_dict = split_data(rec, frame, train_ratio=0.6, stratify=True, seed=None, neurons=neurons)
        r_dict = resample(r_dict, resampler)
        model.fit(r_dict["X_bal"], r_dict["y_bal"])
        y_pred = model.predict(r_dict["X_test"])
        conf_matrix = confusion_matrix(r_dict["y_test"], y_pred, labels=[False, True])
        TP = conf_matrix[1, 1]
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        hit_acc = TP / (TP + FN)
        miss_acc2 = FP / (FP + TN)
        miss_acc = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    return (hit_acc, miss_acc2, accuracy)


if __name__ == '__main__':
    import numpy as np

    np.random.seed(42)
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
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path, cache=True)
        return rec


    workers = cpu_count()
    if user == "Célien":
        pool = pool.ThreadPool(processes=workers)
    elif user == "Théo":
        pool = Pool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}

    # for single recording only

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
    smoteen = imb.combine.SMOTEENN(sampling_strategy='all')
    ros_rus = "ROS-RUS"
    resampler = rus

    neurons = "all"  # "activated"  #  # #

    # per group
    wt_hit, wt_miss, ko_hypo_hit, ko_hypo_miss, wt_stim_hit, wt_stim_miss, ko_hypo_stim_hit, ko_hypo_stim_miss = [], [], [], [], [], [], [], []
    for rec in recs.values():
        try:
            print(f"{rec.filename} ({rec.genotype})")
            acc_hit, acc_miss = [], []
            for i in list(range(-30, 30)):
                acc = frame_model(rec, i, resampler, neurons=neurons, cv=6)
                acc_hit.append(acc[0])
                acc_miss.append(acc[1])
            stim_hit = np.mean(acc_hit[30:45])
            stim_miss = np.mean(acc_miss[30:45])
            if rec.genotype == "WT":
                wt_hit.append(acc_hit)
                wt_miss.append(acc_miss)
                wt_stim_hit.append(stim_hit)
                wt_stim_miss.append(stim_miss)
            elif rec.genotype == "KO-Hypo":
                ko_hypo_hit.append(acc_hit)
                ko_hypo_miss.append(acc_miss)
                ko_hypo_stim_hit.append(stim_hit)
                ko_hypo_stim_miss.append(stim_miss)
        except ValueError:
            print(f"{rec.filename} -> Failed")
            continue

    classification_graph(wt_hit, wt_miss, f"WT ({resampler}-{neurons})", colors=[ppt.wt_color, ppt.wt_light_color])
    classification_graph(ko_hypo_hit, ko_hypo_miss, f"KO-Hypo ({resampler}-{neurons})",
                         colors=[ppt.hypo_color, ppt.hypo_light_color])

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    # ppt.boxplot(axs[0], np.subtract(wt_stim_hit, wt_stim_miss), np.subtract(ko_hypo_stim_hit,ko_hypo_stim_miss), "Hit accuracy", ylim=[0, 1])
    ppt.boxplot(axs[0], wt_stim_hit, ko_hypo_stim_hit, "Hit accuracy", ylim=[0, 1])

    ppt.boxplot(axs[1], wt_stim_miss, ko_hypo_stim_miss, "Miss error", ylim=[0, 1])
    fig.savefig(server_address + "Figure2/modelling/model_accuracy.pdf")
    plt.show()
    print("Done")

    # Correlation of frame model accuracy with cosine similarity of zscore (Trial-by-trial variability)

    cosine_file = np.load(server_address + "Figure2/cosine_similarity/cosine_sim.npy", allow_pickle=True)
    wt_cosine = cosine_file[0]  # first array should be wt
    hypo_cosine = cosine_file[1]

    fig1, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.scatter(wt_cosine, wt_stim_hit, color=ppt.wt_color, marker=".")
    x = np.linspace(0, 1, 30)
    reg = LinearRegression().fit(np.array(wt_cosine).reshape(-1, 1), wt_stim_hit)
    y_pred = reg.predict(np.linspace(0, 1, 30).reshape(-1, 1))
    ax.plot(x, y_pred, color=ppt.wt_color, lw=3, )

    ax.scatter(hypo_cosine, ko_hypo_stim_hit, color=ppt.hypo_color, marker=".")
    reg = LinearRegression().fit(np.array(hypo_cosine).reshape(-1, 1), ko_hypo_stim_hit)
    y_pred = reg.predict(np.linspace(0, 1, 30).reshape(-1, 1))
    ax.plot(x, y_pred, color=ppt.hypo_color, lw=3 )
    sstst, pvalwt = pearsonr(wt_cosine, wt_stim_hit)
    sstst, pvalko = pearsonr(hypo_cosine, ko_hypo_stim_hit)
    ax.set_title(f"wt pval:{pvalwt :.3f} ko: {pvalko: .3f}")
    ax.set_ylabel("Hit model accuracy")
    ax.set_xlabel("Cosine similarity (Tbt var)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0.4, 1])
    ax.spines[["right", "top"]].set_visible(False)
    fig1.savefig(server_address + "Figure2/modelling/correlation_model_accuracy_and_cosine_sim.pdf")

    # Correlation of frame model accuracy with number of driver cells
    from percephone.analysis.utils import idx_resp_neur
    drivers_len_wt, drivers_len_ko = [], []
    for rec in recs.values():
        nb_act, nb_inh = idx_resp_neur(rec, n_type="EXC")
        nb_i_act, nb_i_inh = idx_resp_neur(rec, n_type="INH")
        total_drivers = len(nb_act) + len(nb_inh) + len(nb_i_act) + len(nb_i_inh)
        if rec.genotype == "WT":
            drivers_len_wt.append( total_drivers)
        if rec.genotype == "KO-Hypo":
            drivers_len_ko.append( total_drivers)
    fig1, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.scatter( drivers_len_wt, wt_stim_hit, color=ppt.wt_color, marker=".")
    x = np.linspace(0, 100, 100)
    reg = LinearRegression().fit(np.array( drivers_len_wt).reshape(-1, 1), wt_stim_hit)
    y_pred = reg.predict(np.linspace(0, 100, 100).reshape(-1, 1))
    ax.plot(x, y_pred, color=ppt.wt_color, lw=3, )

    ax.scatter(drivers_len_ko, ko_hypo_stim_hit, color=ppt.hypo_color, marker=".")
    reg = LinearRegression().fit(np.array(drivers_len_ko).reshape(-1, 1), ko_hypo_stim_hit)
    y_pred = reg.predict(np.linspace(0, 100, 100).reshape(-1, 1))
    ax.plot(x, y_pred, color=ppt.hypo_color, lw=3)
    sstst, pvalwt = pearsonr(drivers_len_wt, wt_stim_hit)
    sstst, pvalko = pearsonr(drivers_len_ko, ko_hypo_stim_hit)
    ax.set_title(f"wt pval:{pvalwt :.3f} ko: {pvalko: .3f}")
    ax.set_ylabel("Hit model accuracy")
    ax.set_xlabel("Nb drivers neurons")
    ax.set_xlim([0, 110])
    ax.set_ylim([0.4, 1])
    ax.spines[["right", "top"]].set_visible(False)
    fig1.savefig(server_address + "Figure2/modelling/correlation_model_accuracy_and_n_drivers_neurons.pdf")

    # Correlation of frame model accuracy with number of responsive cells
    resp_wt, resp_ko = [], []
    for rec in recs.values():
        responsivity_exc = rec.matrices["EXC"]["Responsivity"][:, rec.detected_stim]
        responsivity_inh = rec.matrices["INH"]["Responsivity"][:, rec.detected_stim]
        fraction_n_exc = np.mean(np.count_nonzero(responsivity_exc, axis=0))
        fraction_n_inh = np.mean(np.count_nonzero(responsivity_inh, axis=0))
        total_n = rec.zscore_exc.shape[0] + rec.zscore_exc.shape[0]
        fraction_n = ((fraction_n_exc + fraction_n_inh) / total_n)*100
        if rec.genotype == "WT":
            resp_wt.append(fraction_n)
        if rec.genotype == "KO-Hypo":
            resp_ko.append(fraction_n)
    fig1, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.scatter( resp_wt, wt_stim_hit, color=ppt.wt_color, marker=".")
    x = np.linspace(0, 30, 30)
    reg = LinearRegression().fit(np.array( resp_wt).reshape(-1, 1), wt_stim_hit)
    y_pred = reg.predict(np.linspace(0, 30, 30).reshape(-1, 1))
    ax.plot(x, y_pred, color=ppt.wt_color, lw=3, )

    ax.scatter(resp_ko, ko_hypo_stim_hit, color=ppt.hypo_color, marker=".")
    reg_ko = LinearRegression().fit(np.array(resp_ko).reshape(-1, 1), ko_hypo_stim_hit)
    y_pred_ko = reg_ko.predict(np.linspace(0, 30, 30).reshape(-1, 1))
    ax.plot(x, y_pred_ko, color=ppt.hypo_color, lw=3)
    sstst, pvalwt = pearsonr(resp_wt, wt_stim_hit)
    sstst, pvalko = pearsonr(resp_ko, ko_hypo_stim_hit)
    ax.set_title(f"wt pval:{pvalwt :.3f} ko: {pvalko: .3f}")
    ax.set_ylabel("Hit model accuracy")
    ax.set_xlabel("Fraction resp neurons")
    ax.set_xlim([0, 30])
    ax.set_ylim([0.4, 1])
    ax.spines[["right", "top"]].set_visible(False)
    fig1.savefig(server_address + "Figure2/modelling/correlation_model_accuracy_and_resp_nueronss_neurons.pdf")


