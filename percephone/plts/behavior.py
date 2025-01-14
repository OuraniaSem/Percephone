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
    """
    Generates the psychometric curve from the data found in the ROI file.

    Parameters
    ----------
    rec : Recording
        The rec object to be plotted.
    roi_info : dict
        The dictionary that contains the data from the ROI file.
    ax
        The matplotlib axis object to be plotted.
    """
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    # Plotting the retrieved values
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[['right', 'top', 'bottom', 'left']].set_color("black")
    ax.tick_params(axis='both', labelsize=font_s)


def psycho_like_plot_and_synchro(rec, roi_info, ax):
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    to_plot = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        # 0 is plotted if there is no trial of the specified amplitude that has been detected
        if len(rec.detected_stim[rec.stim_ampl == amp]) == 0:
            to_plot.append(0)
        else:
            #TODO: verify this formula, what does it compute ?
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
    """
    Computes the Pearson correlation coefficient (linear relationship) between the number of responsive neurons and the
    detection level for all amplitudes.
    """
    assert detected_trials or undetected_trials, "Please select at least one trial type."
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    neura_activity = []
    resp_mat = np.array(rec.matrices[n_type]["Responsivity"])
    total_n = rec.zscore_exc.shape[0] if n_type == "EXC" else rec.zscore_inh.shape[0]
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        # Getting the neural responsivity corresponding to the selected trial type
        if detected_trials and undetected_trials:
            stim_filter = rec.stim_ampl == amp
        elif detected_trials:
            stim_filter = np.logical_and(rec.detected_stim, rec.stim_ampl == amp)
        elif undetected_trials:
            stim_filter = np.logical_and(np.invert(rec.detected_stim), rec.stim_ampl == amp)
        trials = resp_mat[:, stim_filter]
        # Counting and computing the percentage of responsive neurons for this amplitude
        recruited_det = np.mean(np.count_nonzero(trials, axis=0))
        perc_n_det = (recruited_det / total_n) * 100
        neura_activity.append(perc_n_det)
    # Testing if there is a linear relationship between the neural activity and the detection level
    coef_cor, p_value = ss.pearsonr(neura_activity, converted_list)
    return coef_cor, p_value


def zscore_by_amp(rec, neuron_zscore):
    """
    Computes the mean zscore during the trials of each amplitude for a single neuron.

    Parameters
    ----------
    rec
    neuron_zscore

    Returns
    -------
    list
        A list of the mean zscore of the neuron during the trials for each amplitude.
    """
    firing_curve = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        timings = rec.stim_time[rec.stim_ampl == amp]
        firing_curve.append(np.mean(neuron_zscore[np.linspace(timings, timings + 15, dtype=int)]))
    return firing_curve


def activation_proportion(rec, neur_id):
    """
    Computes the proportion of trials in which the neuron was activated for each amplitude.

    Parameters
    ----------
    rec
    neur_id

    Returns
    -------
    list
        A list of the proportion of trials in which the neuron was activated for each amplitude.
    """
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
    """
    Computes the level of E/I ratio compared to the maximum level for each amplitude.

    Parameters
    ----------
    rec

    Returns
    -------

    """
    ei_ratios = []
    for amp in np.unique(rec.stim_ampl):
        amp_filter = rec.stim_ampl_filter([amp])
        # Counting th mean number of activated EXC neurons per trial of the selected amplitude
        exc_mat = rec.matrices["EXC"]["Responsivity"]
        exc_mat = exc_mat[:, amp_filter]
        exc_mat[exc_mat != 1] = 0
        exc = np.mean(np.count_nonzero(exc_mat, axis=0))
        # Counting th mean number of activated INH neurons per trial of the selected amplitude
        inh_mat = rec.matrices["INH"]["Responsivity"]
        inh_mat = inh_mat[:, amp_filter]
        inh_mat[inh_mat != 1] = 0
        inh = np.mean(np.count_nonzero(inh_mat, axis=0))
        if inh == 0:
            ei_ratios.append(1)
        else:
            ei_ratios.append(exc/inh)
    return np.array(ei_ratios)/max(ei_ratios)


def individuals_neurons_tuning(rec, roi_info, normalize=False):
    """Plot the psychometric curves. Compute the activity curve for single neurons for zscore and activation rate (proportion of trials per
    amplitude for which the neurons is considered active"""
    colors = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color}
    color_behavior = colors[rec.genotype]
    # === Plot 1 ===
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    fig, ax = plt.subplots(4, 1, figsize=(8, 20), sharex=True)
    # Fitting a sigmoid curve on the data (psychometric curve)
    x_psy, y_psy, x0_psy, k_psy = sigmoid_fit(np.array(np.linspace(0, 1, 7)), converted_list)
    ax[0].plot(x_psy, y_psy, color=color_behavior)  # the fitted curve
    ax[0].plot(np.array(np.linspace(0, 1, 7)), converted_list, ".", color=color_behavior)  # the data points
    # If the detection threshold lies between 0 and 12µm, it is plotted on the graph and x0 is displayed
    if x0_psy < 1:
        ax[0].vlines(x=x0_psy, ymin=0, ymax=1, color='red', linestyle='dashed', lw=2)
        ax[0].text(x0_psy, -0.25, f"x0={x0_psy:.2f}", color='red', ha='center', fontsize=10)
    # Plotting the level of E/I ratio
    ei_ratio = ei_ratio_per_amp(rec)
    if ei_ratio.all() != np.nan:
        ax[0].plot(np.array(np.linspace(0, 1, 7)), ei_ratio)

    # === Plot 2 ===
    # Defining the annotation to display the k value when hovering the curve
    annot = ax[1].annotate(
        text='',
        fontsize=10,
        xy=(0, 0),
        xytext=(15, 15),
        textcoords='offset points',
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    # Initializing variables
    dtype = [("ID", "int"), ("x0", "float"), ("k", "float"), ("cluster", "int")]
    neurons_array = np.empty(0, dtype=dtype)
    cluster_dict = {0: ["No response", "gray"], 1: ["Binary activation", "purple"], 2: ["Amplitude related", "pink"]}
    zscores = []
    lines_and_ks = []
    skipped_neurons = 0
    act_n, desact_n = pu.idx_resp_neur(rec)
    # For each neuron:
    for idx, zscore in enumerate(rec.zscore_exc):
        # Computing the mean zscore for each amplitude
        zsc = zscore_by_amp(rec, zscore)
        # Normalization
        if normalize:
            # zsc_normalised_theo = (np.array(zsc) - zsc[0]) / zsc[-1]
            max_abs_value = np.max(np.abs(zsc)) * np.sign(zsc[np.argmax(np.abs(zsc))])
            zsc_normalised = np.array(zsc) / max_abs_value
        else:
            zsc_normalised = np.array(zsc)
        # Handling the case where the neuron is less active as the amplitude increases
        # zsc_normalised = zsc_normalised * -1 if decrease_activity(zsc_normalised) else zsc_normalised
        # Discard neurons activity that are not fitting sigmoid fit
        try:
            x, y, x0, k = sigmoid_fit(np.array(np.linspace(0, 1, 7)), zsc_normalised)
        except:
            skipped_neurons += 1
            continue
        # Clustering of neurons based on the steepness of the sigmoid (k)
        if abs(k) < 1:
            cluster = 0
        elif abs(k) > 10:
            cluster = 1
        else:
            cluster = 2
        # Adding the new neuron's data to the array
        new_row = np.array([(idx, x0, k, cluster)], dtype=dtype)
        neurons_array = np.append(neurons_array, new_row)
        # Storing the lines to be able to display the k values when hovering them
        line, = ax[1].plot(x, y, color=cluster_dict[cluster][1], lw=2, alpha=0.75)
        lines_and_ks.append((line, k))
        # === Plot 3 ===
        # Computing the proportion of trials in which the neuron was activated for each amplitude
        prop_act = activation_proportion(rec, idx)
        ax[2].plot(np.linspace(0, 1, 7), prop_act)

        # === Plot 4 ===
        # Plotting the raw mean zscore of the neuron for each amplitude
        ax[3].plot(np.linspace(0, 1, 7), zsc)
        zscores.append(zsc)

    def update_annotation(event):
        if event.inaxes == ax[1]:
            for line, k_value in lines_and_ks:
                cont, ind = line.contains(event)
                if cont:
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(f"k={k_value:.2f}")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            annot.set_visible(False)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", update_annotation)

    # Computing and plotting the average for all neurons of the mean zscore of each neuron for each amplitude
    ax[3].plot(np.linspace(0, 1, 7), np.average(zscores, axis=0), lw=5, linestyle="dashed", color="black")
    # Computing the mean correaltion coeficient to have an idea of how similarly the neurons respond to the different amplitudes of stimulation
    cor_coef = np.mean(np.corrcoef(zscores))
    ax[0].set_title(f"{rec.filename}-{rec.genotype} {round(cor_coef, 2)} {skipped_neurons}", fontsize=40)
    ax[0].set_ylim([0, 1])
    ax[2].set_ylim([0, 1])
    ax[0].set_xlim([0.16, 1])
    ax[1].set_title(f"Sigmoid fit with z-score for single neur norm={normalize}")
    ax[2].set_title("Activition rate by amp for single neur")
    ax[3].set_title("Z-score by amp for single neur")
    ax[3].set_xticks(np.linspace(0, 1, 7))
    ax[3].set_xticklabels([0, 2, 4, 6, 8, 10, 12])
    fig.tight_layout()
    plt.show()
    print(np.min(neurons_array["k"]), np.max(neurons_array["k"]))
    return neurons_array


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
    return x, y, popt[0], popt[1]

def group_tuning_comp(recs, normalize=False):
    dtype = [("ID", "str"), ("Genotype", "str"),
             ("0_nb", "int"), ("0_mean_x0", "float"), ("0_std_x0", "float"), ("0_mean_k", "float"), ("0_std_k", "float"),
             ("1_nb", "int"), ("1_mean_x0", "float"), ("1_std_x0", "float"), ("1_mean_k", "float"), ("1_std_k", "float"),
             ("2_nb", "int"), ("2_mean_x0", "float"), ("2_std_x0", "float"), ("2_mean_k", "float"), ("2_std_k", "float")]
    array = np.empty(0, dtype=dtype)

if __name__ == '__main__':
    user = "Célien"
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

    # neural tuning curves for individuals neurons
    wt, ko = [], []
    for rec in recs.values():
        avg_k, std_k = individuals_neurons_tuning(rec, roi_info, normalize=True)
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
    fig.show()

    # Correlation of the detection threshold and the average half activation of the neurons and
    # the std half activation of the neurons

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(np.array(wt)[:, 0], np.array(wt)[:, 2], ".", color=ppt.wt_color)
    axs[0].plot(np.array(ko)[:, 0], np.array(ko)[:, 2], ".", color=ppt.hypo_color)
    axs[1].plot(np.array(wt)[:, 1], np.array(wt)[:, 2], ".", color=ppt.wt_color)
    axs[1].plot(np.array(ko)[:, 1], np.array(ko)[:, 2], ".", color=ppt.hypo_color)
    fig.tight_layout()
    fig.show()
