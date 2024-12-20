"""
Théo Gauvrit 18/01/2024
Different plots link to behavior
"""
import percephone.core.recording as pc
import percephone.analysis.utils as pu
import percephone.plts.stats as ppt
import numpy as np
import pandas as pd
import scipy.stats as ss
from multiprocessing import Pool, cpu_count, pool
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font_s = 10


def psycho_like_plot(rec, roi_info, ax):
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = eval("[" + seq[0] + "]")
    # converted_list = [float(x) for x in seq[0].split(',')]
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[['right', 'top', 'bottom', 'left']].set_color("black")
    ax.tick_params(axis='both', labelsize=font_s)


def psycho_like_plot_and_synchro(rec, roi_info, ax):
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    # converted_list = eval("[" + seq[0] + "]")
    converted_list = [float(x) for x in seq[0].split(',')]
    to_plot = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        if len(rec.detected_stim[rec.stim_ampl == amp]) == 0:
            to_plot.append(0)
        else:
            res = sum(rec.detected_stim[rec.stim_ampl == amp]) / len(rec.detected_stim[rec.stim_ampl == amp])
            to_plot.append(res)
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.plot([0, 2, 4, 6, 8, 10, 12], to_plot, linestyle='--')
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=font_s)


def correlation_beh_neur(rec, roi_info, n_type="EXC", detected_trials=True, undetected_trials=True):
    """ Correlation with EXC and INH activity for behavior"""
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    neura_activity, behav = [], []
    resp_mat = np.array(rec.matrices[n_type]["Responsivity"])
    total_n = rec.zscore_exc.shape[0] if n_type == "EXC" else rec.zscore_inh.shape[0]
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        # neural activity
        if detected_trials and undetected_trials:
            stim_filter = rec.stim_ampl == amp
        elif detected_trials:
            stim_filter = np.logical_and(rec.detected_stim, rec.stim_ampl == amp)
        elif undetected_trials:
            stim_filter = np.logical_and(np.invert(rec.detected_stim), rec.stim_ampl == amp)
        trials = resp_mat[:, stim_filter]
        recruited_det = np.mean(np.count_nonzero(trials, axis=0))
        perc_n_det = (recruited_det / total_n) * 100
        neura_activity.append(perc_n_det)
    coef_cor, p_value = ss.pearsonr(neura_activity, converted_list)
    return coef_cor, p_value


def zscore_by_amp(rec, neuron_zscore):
    firing_curve = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        timings = rec.stim_time[rec.stim_ampl == amp]
        firing_curve.append(np.mean(neuron_zscore[np.linspace(timings, timings + 15, dtype=int)]))
    return firing_curve


def activation_proportion(rec, neur_id):
    proportion_by_amp = []
    for amp in np.unique(rec.stim_ampl):
        amp_filter = rec.stim_ampl_filter([amp])
        exc_mat = rec.matrices["EXC"]["Responsivity"]
        exc_mat = exc_mat[neur_id, amp_filter]
        exc_mat[exc_mat != 1] = 0
        exc = np.mean(np.count_nonzero(exc_mat, axis=0))
        proportion_by_amp.append(exc/sum(rec.stim_ampl==amp))
    return proportion_by_amp


def ei_ratio_per_amp(rec):
    ei_ratios = []
    for amp in np.unique(rec.stim_ampl):
        amp_filter = rec.stim_ampl_filter([amp])
        exc_mat = rec.matrices["EXC"]["Responsivity"]
        exc_mat = exc_mat[:, amp_filter]
        exc_mat[exc_mat != 1] = 0
        exc = np.mean(np.count_nonzero(exc_mat, axis=0))
        inh_mat = rec.matrices["INH"]["Responsivity"]
        inh_mat = inh_mat[:, amp_filter]
        inh_mat[inh_mat != 1] = 0
        inh = np.mean(np.count_nonzero(inh_mat, axis=0))
        if inh == 0:
            ei_ratios.append(1)
        else:
            ei_ratios.append(exc/inh)
    return np.array(ei_ratios)/max(ei_ratios)


def individuals_neurons_tuning(rec, roi_info):
    """Plot the psychometric curves. Compute the activity curve for single neurons for zscore and activation rate (proportion of trials per
    amplitude for which the neurons is considered active"""
    colors = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color}
    color_behavior = colors[rec.genotype]
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    fig, ax = plt.subplots(4, 1, figsize=(8, 20), sharex=True)
    x, y, k = sigmoid_fit(np.array([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1]), converted_list)
    ax[0].plot(x, y, color=color_behavior)
    ax[0].plot(np.array([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1]), converted_list, ".", color=color_behavior)
    ei_ratio = ei_ratio_per_amp(rec)
    if ei_ratio.all() != np.nan:
        ax[0].plot(np.array([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1]), ei_ratio)
    zscores = []
    ks = []
    act_n, desact_n = pu.idx_resp_neur(rec)
    for idx, zscore in enumerate(rec.zscore_exc):
        zsc = zscore_by_amp(rec, zscore)
        zsc_normalised = (np.array(zsc) - zsc[0]) / zsc[-1]
        zsc = zsc * -1 if decrease_activity(zsc) else zsc
        try:
            # discard neurons activity that are not fitting sigmoid fit
            x, y, k = sigmoid_fit(np.array([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1]), zsc)
            ks.append(k)
        except:
            continue
        ax[1].plot(x, y)
        # proportion activation neurons
        prop_act = activation_proportion(rec,idx)
        ax[2].plot([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1], prop_act)

        ax[3].plot([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1], zsc)
        zscores.append(zsc)
    ax[3].plot([0, 0.16, 0.33, 0.5, 0.66, 0.83, 1], np.average(zscores, axis=0), lw=5, linestyle="dashed", color="black")
    cor_coef = np.mean(np.corrcoef(zscores))
    ax[0].set_title(str(rec.filename) + "-" + str(rec.genotype) + str(round(cor_coef, 2)), fontsize=40)
    ax[0].set_ylim([0, 1])
    ax[2].set_ylim([0, 1])
    ax[0].set_xlim([0.16, 1])
    ax[1].set_title("Sigmoid fit with z-score for single neur")
    ax[2].set_title("Activition rate by amp for single neur")
    ax[3].set_title("Z-score by amp for single neur")
    fig.tight_layout()
    plt.show()
    return np.mean(ks), np.std(ks)


def decrease_activity(neur_act):
    if neur_act[-1] < neur_act[0]:
        return True
    else:
        return False


def sigmoid_fit(xdata, ydata):
    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

    def sigmoid_Hill(x, n, k):
        y = (x ** n) / (x ** n + k ** n)
        # y = 1/(1+(k/x)**n)
        return y

    popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=100000)
    fix_value = xdata[-1]  # + 1
    # slope, intercept = np.polyfit(x, y, 1)
    # slope = float("{:.2f}".format(slope))
    # Get r2 score
    xdata = xdata.astype(float)
    # y_pred = sigmoid(xdata, *popt)
    # r2 = r2_score(ydata, y_pred)
    # liste_r2.append(r2)

    x = np.linspace(0, fix_value, 50)
    y = sigmoid(x, *popt)
    return x, y, popt[0]


if __name__ == '__main__':
    user = "Théo"
    if user == "Célien":
        directory = "C:/Users/cvandromme/Desktop/Data/"
        roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
        server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper/"
    elif user == "Théo":
        directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
        roi_path = directory + "/FmKO_ROIs&inhibitory.xlsx"
        server_address = "/run/user/1004/gvfs/smb-share:server=engram.local,share=data/Current_members/Ourania_Semelidou/2p/Figures_paper/"

    roi_info = pd.read_excel(roi_path)
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]


    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec


    workers = cpu_count()
    if user == "Célien":
        pool = pool.ThreadPool(processes=workers)
    elif user == "Théo":
        pool = Pool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}
    wt, ko = [], []

    # neural tuning curves for individuals neurons
    for rec in recs.values():
        avg_k, std_k = individuals_neurons_tuning(rec, roi_info)
        if rec.genotype == "WT":
            wt.append([avg_k, std_k, rec.threshold])
        elif rec.genotype == "KO-Hypo":
            ko.append([avg_k, std_k, rec.threshold])

fig, axs = plt.subplots(ncols=2)
ppt.boxplot(axs[0], np.array(wt)[:, 0], np.array(ko)[:, 0], "average k")
print(f"Avg half-activation point: \n WT {np.mean(np.array(wt)[:, 0])}, KO {np.median(np.array(ko)[:, 0])}")
print(f"Detection threshold: \n WT {np.mean(np.array(wt)[:, 2])}, KO {np.mean(np.array(ko)[:, 2])}")
ppt.boxplot(axs[1], np.array(wt)[:, 1], np.array(ko)[:, 1], "std k")
fig.tight_layout()

# Correlation of the detection threshold and the average half activation of the neurons and
# the std half activation of the neurons

fig, axs = plt.subplots(nrows=2)
axs[0].plot(np.array(wt)[:, 0], np.array(wt)[:, 2], ".", color=ppt.wt_color)
axs[0].plot(np.array(ko)[:, 0], np.array(ko)[:, 2], ".", color=ppt.hypo_color)
axs[1].plot(np.array(wt)[:, 1], np.array(wt)[:, 2], ".", color=ppt.wt_color)
axs[1].plot(np.array(ko)[:, 1], np.array(ko)[:, 2], ".", color=ppt.hypo_color)
fig.tight_layout()
