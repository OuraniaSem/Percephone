# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as ss
from multiprocessing import cpu_count, pool
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols

import percephone.core.recording as pc
import percephone.plts.stats as ppt
# endregion ============================================================================================================
# region ======================================== Response features ====================================================

def get_features(recs):
    """
    Get the neuronal percentage of recruited neurons for both neuron type for all trials for all recordings and build a
    DataFrame.

    Parameters
    ----------
    recs

    Returns
    -------
    pd.DataFrame
    """
    # === === Building the DataFrame === ===
    rows = []
    for rec in recs:
        feature_vectors = {
            # --- Retrieving the target and covariate ---
            "behavior": rec.detected_stim,
            "amplitude": rec.stim_ampl,
            # --- Retrieving the different predictors ---
            # Percentage of recruited neurons
            "act_EXC_perc": rec.get_perc_resp(pattern=1, n_type="EXC"),
            "inh_EXC_perc": rec.get_perc_resp(pattern=-1, n_type="EXC"),
            "act_INH_perc": rec.get_perc_resp(pattern=1, n_type="INH"),
            "inh_INH_perc": rec.get_perc_resp(pattern=-1, n_type="INH"),
            # Mean peak amplitude for responsive neurons
            "act_EXC_amp": rec.get_mean_param(pattern=1, n_type="EXC", parameter="Peak_amplitude"),
            "inh_EXC_amp": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="Peak_amplitude"),
            "act_INH_amp": rec.get_mean_param(pattern=1, n_type="INH", parameter="Peak_amplitude"),
            "inh_INH_amp": rec.get_mean_param(pattern=-1, n_type="INH", parameter="Peak_amplitude"),
            # Mean peak delay for responsive neurons
            "act_EXC_delay": rec.get_mean_param(pattern=1, n_type="EXC", parameter="Peak_delay"),
            "inh_EXC_delay": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="Peak_delay"),
            "act_INH_delay": rec.get_mean_param(pattern=1, n_type="INH", parameter="Peak_delay"),
            "inh_INH_delay": rec.get_mean_param(pattern=-1, n_type="INH", parameter="Peak_delay"),
        }
        nb_trials = len(feature_vectors["behavior"])
        for trial_id in range(nb_trials):
            row = {"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.session_threshold}
            for feature, vector in feature_vectors.items():
                row[feature] = vector[trial_id]
            rows.append(row)
    return pd.DataFrame(rows)


def filter_amplitude(data, amplitude="all", no_go=False):
    if amplitude == "all":
        filt_data = data
    elif amplitude == "threshold":
        filt_data = data[data["amplitude"] == data["threshold"]]
    elif amplitude == "sub":
        filt_data = data[data["amplitude"] == data["threshold"] - 2]
    elif amplitude == "supra":
        filt_data = data[data["amplitude"] == data["threshold"] + 2]
    elif amplitude == "all_sub":
        filt_data = data[data["amplitude"] < data["threshold"]]
    elif amplitude == "all_supra":
        filt_data = data[data["amplitude"] > data["threshold"]]
    elif isinstance(amplitude, list):
        filt_data = data[data["amplitude"].isin(amplitude)]
    else:
        filt_data = None
    if not no_go:
        filt_data = filt_data[filt_data["amplitude"] != 0]
    return filt_data

def get_sub_supra_threshold(data, behavior_filter=None, genotype="WT", comparison="sub"):
    # Filtering the genotype
    data = data[data["Genotype"] == genotype]
    # Filtering the behavior
    if behavior_filter == None:
        data.drop(columns=["behavior"])
        grouping_cols = ["Genotype", "ID"]
    else:
        data = data[data["behavior"] == behavior_filter]
        grouping_cols = ["Genotype", "ID", "behavior"]
    # Filtering the amplitude
    threshold_trials = filter_amplitude(data, amplitude="threshold").groupby(grouping_cols, as_index=False).mean()
    genotype_threshold_average = 2 * round(threshold_trials["threshold"].mean() / 2) if 0 < threshold_trials["threshold"].mean() < 12 else 12
    amplitude = comparison if comparison != "mean_genotype" else genotype_threshold_average
    non_threshold_trials = filter_amplitude(data, amplitude=amplitude).groupby(grouping_cols, as_index=False).mean()
    # Filtering out the no-go trials
    non_threshold_trials = non_threshold_trials[non_threshold_trials["amplitude"] != 0]
    return threshold_trials, non_threshold_trials


def compare_sub_supra_within(data, behavior_filter=None, genotype="WT", comparison="sub"):
    threshold_trials, non_threshold_trials = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=genotype, comparison=comparison)
    colors_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                   "KO": [ppt.ko_color, ppt.ko_light_color]}
    # Asserting that the match between threshold and non-threshold trials for each animal
    common_IDs = set(threshold_trials["ID"]).intersection(non_threshold_trials["ID"])
    threshold_trials = threshold_trials[threshold_trials["ID"].isin(common_IDs)]
    non_threshold_trials = non_threshold_trials[non_threshold_trials["ID"].isin(common_IDs)]
    threshold_trials = threshold_trials.sort_values("ID").reset_index(drop=True)
    non_threshold_trials = non_threshold_trials.sort_values("ID").reset_index(drop=True)
    # Plotting the comparisons
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold"]
    variables = [col for col in data.columns if col not in not_variables]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    for variable, ax in zip(variables, axes_flat):
        ppt.boxplot(ax, threshold_trials[variable], non_threshold_trials[variable], ylabel=variable, paired=True, title="", ylim=[],
                    colors=colors_dict[genotype], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison in {genotype} of threshold trials and {comparison} trials"
                 f"\n[behavior filter={behavior_filter}] n={len(non_threshold_trials)}", fontsize=20)
    fig.canvas.manager.set_window_title(f"comp_{genotype}_threshold_{comparison}_{behavior_filter}")
    plt.show()


def compare_sub_supra_between(data, behavior_filter=None, gp1="WT", gp2="KO-Hypo", gp1_amps="sub", gp2_amps="sub",
                              colors=[ppt.wt_color, ppt.hypo_color]):
    gp1_threshold, gp1_non_threshold = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp1, comparison=gp1_amps if gp1_amps != "threshold" else "sub")
    if gp1_amps == "mean_genotype" and gp2_amps == "gp1_threshold":
        gp2_amps = [gp1_non_threshold["amplitude"].values[0]]
    gp2_threshold, gp2_non_threshold = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp2, comparison=gp2_amps if gp2_amps != "threshold" else "sub")
    # Plotting the comparisons
    data_gp1 = gp1_threshold if gp1_amps == "threshold" else gp1_non_threshold
    data_gp2 = gp2_threshold if gp2_amps == "threshold" else gp2_non_threshold
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold"]
    variables = [col for col in data.columns if col not in not_variables]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    for variable, ax in zip(variables, axes_flat):
        ppt.boxplot(ax, data_gp1[variable], data_gp2[variable], ylabel=variable, paired=False, title="", ylim=[],
                    colors=colors, det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison between {gp1_amps} trials of {gp1} & {gp2_amps} trials of {gp2}"
                 f"\n[behavior filter={behavior_filter}] n={len(data_gp1)}{gp1}/{len(data_gp2)}{gp2}", fontsize=20)
    fig.canvas.manager.set_window_title(f"comp_{gp1_amps}({gp1})_{gp2_amps}({gp2})_{behavior_filter}")
    plt.show()


def compare_det_undet(data, genotype="WT", amplitude="all"):
    colors_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                   "KO": [ppt.ko_color, ppt.ko_light_color]}
    grouping_cols = ["Genotype", "ID", "behavior"]
    # Filtering the amplitude
    genotype_data = data[data["Genotype"] == genotype]
    ampl_data = filter_amplitude(genotype_data, amplitude=amplitude, no_go=False).groupby(grouping_cols, as_index=False).mean()
    det_data = ampl_data[ampl_data["behavior"] == True]
    undet_data = ampl_data[ampl_data["behavior"] == False]
    # Plotting the comparisons
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold"]
    variables = [col for col in data.columns if col not in not_variables]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    for variable, ax in zip(variables, axes_flat):
        ppt.boxplot(ax, det_data[variable], undet_data[variable], ylabel=variable, paired=False, title="", ylim=[],
                    colors=colors_dict[genotype], det_marker=True, force_markers_identity=False)
    fig.suptitle(f"Comparison in {genotype} of detected trials and undetected trials"
                 f"\n[amplitude filter={amplitude}]", fontsize=20)
    fig.canvas.manager.set_window_title(f"comp_{genotype}_det_undet_{amplitude}")
    plt.show()
    return det_data, undet_data



def group_comp_param(recs, parameter, ko_hypo_only=False, stim_ampl="all", ylim=[]):
    """
    Compare a given parameter between WT and KO groups across neuron types and response types.

    This function generates boxplots to compare a specified neuronal parameter across different
    conditions, including neuron type (EXC, INH), response type (-1, 1), and stimulus detection status
    (detected vs. undetected). The results are plotted in a 2x4 subplot figure.

    Parameters
    ----------
    recs : dict
        Dictionary containing recording data, where keys are identifiers and values are recording objects.
    parameter : str
        The neuronal parameter to analyze (e.g., "Peak_delay", "Peak_amplitude").
    ko_hypo_only : bool, optional
        If True, only KO-Hypo data is used for KO comparisons (default is False).
    stim_ampl : str or list, optional
        Specifies the stimulation amplitude(s) to include. Can be "all" or a specific value/list of values (default is "all").
    ylim : list, optional
        Specifies the y-axis limits for the plots. If an empty list is provided, the limits are determined automatically.

    Notes
    -----
    - The function iterates through the dataset, filtering neurons based on genotype and response type.
    - It computes the mean value of the given parameter for detected and undetected stimuli.
    - Boxplots are generated to compare WT and KO groups.
    - The function assumes the presence of external plotting utilities from `ppt` (e.g., `ppt.boxplot`).

    Returns
    -------
    None
        The function displays a matplotlib figure with boxplots and saves it as a PDF if `save_figure` is enabled.

    Raises
    ------
    AttributeError
        If `recs` does not contain the expected attributes (`genotype`, `matrices`, `stim_ampl_filter`, etc.).
    KeyError
        If the requested `parameter` is not found in `recs.matrices`.

    Example
    -------
    >>> group_comp_param(recs, parameter="Peak_delay", ko_hypo_only=True, stim_ampl="threshold", ylim=[-10, 10])
    (Displays and saves a figure comparing Peak Delay between WT and KO-Hypo across different conditions.)
    """
    fig, axs = plt.subplots(2, 4, figsize=(24, 16), constrained_layout=True)
    for i, neuron_type in enumerate(["EXC", "INH"]):
        for j, response_type in enumerate([-1, 1]):
            auto_ylim = ylim
            if ylim != [] and response_type == -1 and parameter != "Peak_delay":
                auto_ylim = [-i for i in ylim][::-1]
            wt_det, wt_undet, ko_det, ko_undet = [], [], [], []
            if ko_hypo_only:
                ko_type = "KO-Hypo"
                color_ko = ppt.hypo_color
                light_color_ko = ppt.hypo_light_color
            else:
                ko_type = "(KO + KO-Hypo)"
                color_ko = ppt.all_ko_color
                light_color_ko = ppt.all_ko_light_color

            for rec in recs.values():
                if ko_hypo_only and rec.genotype == "KO":
                    continue
                else:
                    # Filtering stimulation amplitudes
                    stim_filter = rec.stim_ampl_filter(stim_ampl)
                    # Responsivity and parameter matrices building or retrieving according to neuron type
                    resp_mat = rec.matrices[neuron_type]["Responsivity"]
                    para_mat = rec.matrices[neuron_type][parameter]
                    # For detected stimuli
                    stim_thre_det = np.logical_and(stim_filter, rec.detected_stim)
                    resp_detected = resp_mat[:,stim_thre_det]
                    detected = para_mat[:,stim_thre_det]
                    det = np.where(resp_detected == response_type, detected, np.nan)
                    # For undetected stimuli
                    stim_thre_undet = np.logical_and(stim_filter, np.invert(rec.detected_stim))
                    resp_undetected = resp_mat[:,stim_thre_undet]
                    undetected = para_mat[:,stim_thre_undet]
                    undet = np.where(resp_undetected == response_type, undetected, np.nan)
                    if rec.genotype == "WT":
                        wt_det.append(np.nanmean(np.nanmean(det, axis=1)))
                        wt_undet.append(np.nanmean(np.nanmean(undet, axis=1)))
                    else:
                        ko_det.append(np.nanmean(np.nanmean(det, axis=1)))
                        ko_undet.append(np.nanmean(np.nanmean(undet, axis=1)))
            ppt.boxplot(axs[i, 2*j], wt_det, ko_det, paired=False, ylabel=f"{parameter}", ylim=auto_ylim, colors=[ppt.wt_color, color_ko], det_marker=True)
            ppt.boxplot(axs[i, 2*j+1], wt_undet, ko_undet, paired=False, ylabel=f"{parameter}", ylim=auto_ylim, colors=[ppt.wt_light_color, light_color_ko], det_marker=False)
            axs[i, 2*j].set_title(f"Det {neuron_type}({response_type})")
            axs[i, 2*j+1].set_title(f"Undet {neuron_type}({response_type})")
    title = f"{parameter}[WT_{ko_type}]_{stim_ampl}_amp"
    fig.suptitle(f"Mean {parameter} between WT and {ko_type}. Amplitude(s): {stim_ampl}", fontsize=10)
    fig.canvas.manager.set_window_title(title)
    plt.show()


def det_comp_param(recs, parameter, stim_ampl="all", ylim=[]):
    """
    Compare a given neuronal parameter between detected and undetected stimuli.

    This function generates boxplots to analyze the differences in a specified neuronal parameter
    across detected and undetected stimuli, considering neuron types (EXC, INH) and response types (-1, 1).
    Comparisons are performed across WT, KO (including KO-Hypo), and KO-Hypo groups.

    Parameters
    ----------
    recs : dict
        Dictionary containing recording data, where keys are identifiers and values are recording objects.
    parameter : str
        The neuronal parameter to analyze (e.g., "Peak_delay", "Peak_amplitude").
    stim_ampl : str or list, optional
        Specifies the stimulation amplitude(s) to include. Can be "all" or a specific value/list of values (default is "all").
    ylim : list, optional
        Specifies the y-axis limits for the plots. If an empty list is provided, the limits are determined automatically.

    Notes
    -----
    - The function filters data based on stimulus detection status.
    - It computes the mean parameter values for detected and undetected stimuli.
    - Boxplots compare detected vs. undetected stimuli within WT, KO (KO + KO-Hypo), and KO-Hypo groups.
    - Results are saved as CSV files if required.
    - The function assumes the presence of external plotting utilities from `ppt` (e.g., `ppt.boxplot`).

    Returns
    -------
    None
        The function displays a matplotlib figure with boxplots and saves it as a PDF if `save_figure` is enabled.

    Raises
    ------
    AttributeError
        If `recs` does not contain the expected attributes (`genotype`, `matrices`, `stim_ampl_filter`, etc.).
    KeyError
        If the requested `parameter` is not found in `recs.matrices`.

    Example
    -------
    >>> det_comp_param(recs, parameter="Peak_delay", stim_ampl=[10, 12], ylim=[-50, 50])
    (Displays and saves a figure comparing Peak Delay for detected vs. undetected stimuli.)
    """
    fig, axs = plt.subplots(2, 6, figsize=(36, 16), constrained_layout=True)
    for i, neuron_type in enumerate(["EXC", "INH"]):
        for j, response_type in enumerate([-1, 1]):
            auto_ylim = ylim
            if ylim != [] and response_type == -1 and parameter != "Peak_delay":
                auto_ylim = [-i for i in ylim][::-1]
            wt_det, wt_undet, ko_det, ko_undet, hypo_det, hypo_undet = [], [], [], [], [], []
            for rec in recs.values():
                # Filtering of stimulations
                stim_filter = rec.stim_ampl_filter(stim_ampl)
                # Responsivity and parameter matrices building or retrieving according to neuron type
                resp_mat = rec.matrices[neuron_type]["Responsivity"]
                para_mat = rec.matrices[neuron_type][parameter]
                # For detected stimuli
                stim_thre_det = np.logical_and(stim_filter, rec.detected_stim)
                resp_detected = resp_mat[:, stim_thre_det]
                detected = para_mat[:, stim_thre_det]
                det = np.where(resp_detected == response_type, detected, np.nan)
                # For undetected stimuli
                stim_thre_undet = np.logical_and(stim_filter, np.invert(rec.detected_stim))
                resp_undetected = resp_mat[:, stim_thre_undet]
                undetected = para_mat[:, stim_thre_undet]
                undet = np.where(resp_undetected == response_type, undetected, np.nan)
                if rec.genotype == "WT":
                    wt_det.append(np.nanmean(np.nanmean(det, axis=1)))
                    wt_undet.append(np.nanmean(np.nanmean(undet, axis=1)))
                else:
                    ko_det.append(np.nanmean(np.nanmean(np.nanmean(det, axis=1))))
                    ko_undet.append(np.nanmean(np.nanmean(undet, axis=1)))
                if rec.genotype == "KO-Hypo":
                    hypo_det.append(np.nanmean(np.nanmean(np.nanmean(det, axis=1))))
                    hypo_undet.append(np.nanmean(np.nanmean(undet, axis=1)))
            ppt.boxplot(axs[i, 3 * j], wt_det, wt_undet, ylabel=parameter, paired=True,
                        title=f"WT {neuron_type}({response_type})", ylim=auto_ylim,
                        colors=[ppt.wt_color, ppt.wt_light_color])
            ppt.boxplot(axs[i, 3 * j + 1], ko_det, ko_undet, ylabel=parameter, paired=True,
                        title=f"KO + KO-Hypo {neuron_type}({response_type})", ylim=auto_ylim,
                        colors=[ppt.all_ko_color, ppt.all_ko_light_color])
            ppt.boxplot(axs[i, 3 * j + 2], hypo_det, hypo_undet, ylabel=parameter, paired=True,
                        title=f"KO-Hypo {neuron_type}({response_type})", ylim=auto_ylim,
                        colors=[ppt.hypo_color, ppt.hypo_light_color])
            # excel_df_WT = pd.DataFrame(data={"WT Det": wt_det, "WT Undet": wt_undet})
            # excel_df_KO = pd.DataFrame(data={"Hypo Det": hypo_det, "Hypo Undet": hypo_undet})
            # excel_df_WT.to_csv(f"{server_address}data/det_{parameter}_{neuron_type}_{stim_ampl}_{response_type}_WT.csv", sep=",")
            # excel_df_KO.to_csv(f"{server_address}data/det_{parameter}_{neuron_type}_{stim_ampl}_{response_type}_KO.csv", sep=",")
    title = f"{parameter}[det_undet]_{stim_ampl}_amp"
    fig.suptitle(f"Mean {parameter} for detected vs. undetected stimuli. Amplitude(s): {stim_ampl}", fontsize=10)
    fig.canvas.manager.set_window_title(title)
    plt.show()


# endregion ============================================================================================================
# region ======================================== Responsivity =========================================================
def nb_neurons(recs):
    rows = []
    for rec in recs:
        n_EXC = rec.zscore_exc.shape[0]
        n_INH = rec.zscore_inh.shape[0]
        perc_EXC = n_EXC / (n_EXC + n_INH) * 100
        perc_INH = n_INH / (n_EXC + n_INH) * 100
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "n_EXC": n_EXC, "n_INH": n_INH, "perc_EXC": perc_EXC, "perc_INH": perc_INH})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 16), constrained_layout=True)
    for i, n_type in enumerate(["EXC", "INH"]):
        for j, metric in enumerate(["n", "perc"]):
            wt = data[data["Genotype"] == "WT"][f"{metric}_{n_type}"].values
            ko = data[data["Genotype"].isin(["KO", "KO-Hypo"])][f"{metric}_{n_type}"].values
            ylim = [0, 100] if metric == "perc" else [0, 150]
            ppt.boxplot(ax[i, j], wt, ko, ylabel=f"{n_type} neurons ({metric})", paired=False, title="", ylim=ylim,
                        colors=[ppt.wt_color, ppt.all_ko_color], det_marker=False, force_markers_identity=False)
    fig.suptitle("Number and percentage of neurons in the field of view", fontsize=10)
    fig.canvas.manager.set_window_title("Nb neurons FOV")
    plt.show()
    return data


def fraction_resp(pattern, n_type, ko_hypo_only=True, stim_ampl="all", no_go_normalize=True):
    """
    Compute the fraction of responsive neurons based on a given response pattern.

    This function calculates the percentage of neurons responding to detected and undetected stimuli
    for a specified neuron type (EXC or INH). The response is computed for WT, KO-Hypo, and optionally KO groups.

    Parameters
    ----------
    pattern : int
        The response pattern to analyze. If 0, the function considers any nonzero response as activation.
    n_type : str
        The neuron type to analyze ("EXC" for excitatory, "INH" for inhibitory).
    ko_hypo_only : bool, optional
        If True, only KO-Hypo data is included for KO comparisons (default is True).
    stim_ampl : str or list, optional
        Specifies the stimulation amplitude(s) to include. Can be "all" or a specific value/list of values (default is "all").
    no_go_normalize : bool, optional
        If True, normalizes the response rates by subtracting the number of responsive neurons in no-go trials (default is True).

    Returns
    -------
    wt_det : np.ndarray
        Array of percentages of detected responsive neurons in the WT group.
    ko_det : np.ndarray
        Array of percentages of detected responsive neurons in the KO group (or KO-Hypo if `ko_hypo_only=True`).
    wt_undet : np.ndarray
        Array of percentages of undetected responsive neurons in the WT group.
    ko_undet : np.ndarray
        Array of percentages of undetected responsive neurons in the KO group (or KO-Hypo if `ko_hypo_only=True`).

    Notes
    -----
    - Responses are extracted from the "Responsivity" matrix of the recording data.
    - Responses are filtered based on stimulation amplitude and whether the stimulus was detected.
    - If `pattern == 0`, all nonzero responses are treated as recruitment (binary response).
    - If `no_go_normalize=True`, responses are adjusted by subtracting the average number of responsive neurons in no-go trials.
    - The total number of neurons is used to compute response percentages.

    Raises
    ------
    AttributeError
        If `recs` does not contain expected attributes (`genotype`, `matrices`, `stim_ampl_filter`, etc.).
    KeyError
        If `n_type` is not found in `recs.matrices` or if the `pattern` does not exist.

    Example
    -------
    >>> wt_det, ko_det, wt_undet, ko_undet = fraction_resp(pattern=1, n_type="EXC", stim_ampl=[10, 20])
    (Returns the percentage of responsive neurons for detected/undetected stimuli in each genotype.)
    """
    wt_det, wt_undet, ko_det, ko_undet = [], [], [], []
    for rec in recs.values():
        resp_mat = np.array(rec.matrices[n_type]["Responsivity"])
        no_go_filter = rec.stim_ampl_filter(stim_ampl=[0], include_no_go=True)
        ampl_filter = rec.stim_ampl_filter(stim_ampl=stim_ampl, include_no_go=False)
        ampl_det_filt = np.logical_and(rec.detected_stim, ampl_filter)
        ampl_undet_filt = np.logical_and(np.invert(rec.detected_stim), ampl_filter)
        # detected
        trials_no_go = resp_mat[:, no_go_filter]
        trials_detected = resp_mat[:, ampl_det_filt]
        trials_undetected = resp_mat[:, ampl_undet_filt]
        if pattern == 0:
            trials_detected[trials_detected != 0] = 1
            trials_undetected[trials_undetected != 0] = 1
            trials_no_go[trials_no_go != 0] = 1
        else:
            trials_detected[trials_detected != pattern] = 0
            trials_undetected[trials_undetected != pattern] = 0
            trials_no_go[trials_no_go != pattern] = 0
        # The total number of neurons is computed
        total_n = rec.zscore_exc.shape[0] if n_type == "EXC" else rec.zscore_inh.shape[0]
        # Computation of the number of responsive neurons
        recruited_det = np.mean(np.count_nonzero(trials_detected, axis=0))
        recruited_undet = np.mean(np.count_nonzero(trials_undetected, axis=0))
        # Normalization by the number of responsive neurons for no-go trials
        if no_go_normalize:
            recruited_no_go = np.mean(np.count_nonzero(trials_no_go, axis=0))
            recruited_det -= recruited_no_go
            recruited_undet -= recruited_no_go
            recruited_det = 0 if recruited_det < 0 else recruited_det
            recruited_undet = 0 if recruited_undet < 0 else recruited_undet
        # Computation of the percentage of responsive neurons
        perc_n_det = (recruited_det / total_n) * 100
        perc_n_undet = (recruited_undet / total_n) * 100
        # Storing the computed percentage in the corresponding list
        if rec.genotype == "WT":
            wt_det.append(perc_n_det)
            wt_undet.append(perc_n_undet)
        elif rec.genotype == "KO-Hypo":
            ko_det.append(perc_n_det)
            ko_undet.append(perc_n_undet)
        elif rec.genotype == "KO" and not ko_hypo_only:
            ko_det.append(perc_n_det)
            ko_undet.append(perc_n_undet)
    return np.array(wt_det), np.array(ko_det), np.array(wt_undet), np.array(ko_undet)


def plot_neuron_frac_wt_ko(pattern, ko_hypo_only=True, stim_ampl="all", ylim=[], no_go_normalize=True):
    """
    Plot the fraction of responsive neurons in WT vs. KO groups.

    This function visualizes the percentage of neurons responding to detected and undetected stimuli
    for both excitatory (EXC) and inhibitory (INH) neuron types. It compares WT and KO (or KO-Hypo) groups
    using boxplots.

    Parameters
    ----------
    pattern : int
        The response pattern to analyze. If 0, the function considers both activation (1) and inhibition (-1).
        If 1, only activated neurons are considered. If -1, only inhibited neurons are considered.
    ko_hypo_only : bool, optional
        If True, only KO-Hypo data is used for KO comparisons (default is True).
    stim_ampl : str or list, optional
        Specifies the stimulation amplitude(s) to include. Can be "all" or a specific value/list of values (default is "all").
    ylim : list, optional
        Specifies the y-axis limits for the plots. If an empty list is provided, the limits are determined automatically.
    no_go_normalize : bool, optional
        If True, normalizes the response rates by subtracting the number of responsive neurons in no-go trials (default is True).

    Returns
    -------
    None
        The function displays a matplotlib figure with boxplots and saves it as a PDF if `save_figure` is enabled.

    Notes
    -----
    - The function calls `fraction_resp()` to compute the percentage of responsive neurons.
    - Boxplots are generated for detected and undetected stimuli across neuron types (EXC, INH).
    - The KO group can include either KO-Hypo only or both KO and KO-Hypo depending on `ko_hypo_only`.
    - The function assumes external plotting utilities from `ppt` (e.g., `ppt.boxplot`).

    Raises
    ------
    ValueError
        If `pattern` is not 0, 1, or -1.
    AttributeError
        If `fraction_resp()` fails due to missing attributes in `recs`.

    Example
    -------
    >>> plot_neuron_frac_wt_ko(pattern=1, ko_hypo_only=False, stim_ampl="all", ylim=[0, 50])
    (Displays and saves a figure comparing neuron recruitment between WT and KO groups.)
    """
    if ko_hypo_only:
        ko_type = "KO-Hypo"
        color_ko = ppt.hypo_color
        light_color_ko = ppt.hypo_light_color
    else:
        ko_type = "(KO + KO-Hypo)"
        color_ko = ppt.all_ko_color
        light_color_ko = ppt.all_ko_light_color
    fig, axs = plt.subplots(2, 2, figsize=(12, 16), constrained_layout=True)
    for y_index, n_type in enumerate(["EXC", "INH"]):
        wt_det, ko_det, wt_undet, ko_undet = fraction_resp(pattern=pattern, n_type=n_type, ko_hypo_only=ko_hypo_only,
                                                           stim_ampl=stim_ampl, no_go_normalize=no_go_normalize)
        ppt.boxplot(axs[y_index, 0], wt_det, ko_det, paired=False, ylabel="Neurons(%)", title=f"{n_type} Detected",
                    ylim=ylim,
                    colors=[ppt.wt_color, color_ko], det_marker=True)
        ppt.boxplot(axs[y_index, 1], wt_undet, ko_undet, paired=False, ylabel="Neurons(%)",
                    title=f"{n_type} Undetected", ylim=ylim,
                    colors=[ppt.wt_light_color, light_color_ko], det_marker=False)
    t_pattern = "recruited (1 and -1)" if pattern == 0 else ("activated (1)" if pattern == 1 else "inhibited (-1)")
    fig.suptitle(
        f"Percentage of neurons {t_pattern} during hit and miss trials (amplitude: {stim_ampl}) - WT vs. {ko_type}",
        fontsize=5)
    plt.show()


def plot_neuron_frac_det_undet(pattern, ko_hypo_only=True, stim_ampl="all", ylim=[], no_go_normalize=True):
    """
    Compare the fraction of responsive neurons between detected and undetected stimuli.

    This function visualizes the percentage of neurons responding to detected and undetected stimuli
    for both excitatory (EXC) and inhibitory (INH) neuron types. It compares WT and KO (or KO-Hypo) groups
    using boxplots.

    Parameters
    ----------
    pattern : int
        The response pattern to analyze. If 0, both activation (1) and inhibition (-1) are included.
        If 1, only activated neurons are considered. If -1, only inhibited neurons are considered.
    ko_hypo_only : bool, optional
        If True, only KO-Hypo data is used for KO comparisons (default is True).
    stim_ampl : str or list, optional
        Specifies the stimulation amplitude(s) to include. Can be "all" or a specific value/list of values (default is "all").
    ylim : list, optional
        Specifies the y-axis limits for the plots. If an empty list is provided, the limits are determined automatically.
    no_go_normalize : bool, optional
        If True, normalizes the response rates by subtracting the number of responsive neurons in no-go trials (default is True).

    Returns
    -------
    None
        The function displays a matplotlib figure with boxplots and saves it as a PDF if `save_figure` is enabled.

    Notes
    -----
    - The function calls `fraction_resp()` to compute the percentage of responsive neurons.
    - Boxplots compare detected vs. undetected stimuli within WT and KO (KO-Hypo) groups.
    - The KO group can include either KO-Hypo only or both KO and KO-Hypo depending on `ko_hypo_only`.
    - The function assumes external plotting utilities from `ppt` (e.g., `ppt.boxplot`).
    - Results are saved as CSV files if needed.

    Raises
    ------
    ValueError
        If `pattern` is not 0, 1, or -1.
    AttributeError
        If `fraction_resp()` fails due to missing attributes in `recs`.

    Example
    -------
    >>> plot_neuron_frac_det_undet(pattern=1, ko_hypo_only=False, stim_ampl=[10, 20], ylim=[0, 50])
    (Displays and saves a figure comparing detected vs. undetected neuron recruitment in WT and KO groups.)
    """
    if ko_hypo_only:
        ko_type = "KO-Hypo"
        color_ko = ppt.hypo_color
        light_color_ko = ppt.hypo_light_color
    else:
        ko_type = "(KO + KO-Hypo)"
        color_ko = ppt.all_ko_color
        light_color_ko = ppt.all_ko_light_color

    fig, axs = plt.subplots(2, 2, figsize=(12, 16), constrained_layout=True)
    for y_index, n_type in enumerate(["EXC", "INH"]):
        wt_det, ko_det, wt_undet, ko_undet = fraction_resp(pattern=pattern, n_type=n_type, ko_hypo_only=ko_hypo_only,
                                                           stim_ampl=stim_ampl, no_go_normalize=no_go_normalize)
        ppt.boxplot(axs[y_index, 0], wt_det, wt_undet, paired=True, ylabel="Neurons(%)", title=f"{n_type} - WT",
                    ylim=ylim,
                    colors=[ppt.wt_color, ppt.wt_light_color])
        ppt.boxplot(axs[y_index, 1], ko_det, ko_undet, paired=True, ylabel="Neurons(%)", title=f"{n_type} - {ko_type}",
                    ylim=ylim,
                    colors=[color_ko, light_color_ko])
        # excel_df_WT = pd.DataFrame(data={"WT Det": wt_det, "WT Undet": wt_undet})
        # excel_df_KO = pd.DataFrame(data={"Hypo Det": hypo_det, "Hypo Undet": hypo_undet})
        # excel_df_WT.to_csv(f"{server_address}data/frac_{n_type}_{stim_ampl}_{pattern}_WT.csv", sep=",")
        # excel_df_KO.to_csv(f"{server_address}data/frac_{n_type}_{stim_ampl}_{pattern}_KO.csv", sep=",")
    t_pattern = "recruited (1 and -1)" if pattern == 0 else ("activated (1)" if pattern == 1 else "inhibited (-1)")
    fig.suptitle(
        f"Comparison of neurons {t_pattern} between hit and miss trials (amplitude: {stim_ampl}) - WT & {ko_type} [No-go norm={no_go_normalize}]",
        fontsize=5)
    title = f"n_resp_({pattern})_{stim_ampl}_amp_[WT_{ko_type}]"
    fig.canvas.manager.set_window_title(title)
    plt.show()


def resp_contrast(pattern="recruited", stim_ampl="all", method="ratio", ylim=[]):
    """
    Compare the contrast in neuronal responses between detected and undetected trials.

    This function computes and visualizes the contrast (either as a ratio or delta) of responsive neurons
    between detected and undetected stimuli for excitatory (EXC) and inhibitory (INH) neurons. WT and KO-Hypo
    groups are compared using boxplots.

    Parameters
    ----------
    pattern : str, optional
        Specifies the type of neuronal response to analyze. Must be one of:
        - `"recruited"` (both activation and inhibition, default)
        - `"activated"` (only activation)
        - `"inhibited"` (only inhibition)
    stim_ampl : str or list, optional
        Specifies the stimulation amplitude(s) to include. Can be `"all"` or a specific value/list of values (default is `"all"`).
    method : str, optional
        The method used to compute response contrast. Must be one of:
        - `"ratio"` (ratio of detected to undetected response, default)
        - `"delta"` (difference between detected and undetected response)
    ylim : list, optional
        Specifies the y-axis limits for the plots. If an empty list is provided, the limits are determined automatically.

    Returns
    -------
    None
        The function displays a matplotlib figure with boxplots and saves it as a PDF if `save_figure` is enabled.

    Notes
    -----
    - Calls `fraction_resp()` to compute the percentage of responsive neurons.
    - Boxplots compare detected vs. undetected trials across neuron types (EXC, INH).
    - The contrast is computed either as a ratio (`detected / undetected`) or as a difference (`detected - undetected`).
    - The function assumes external plotting utilities from `ppt` (e.g., `ppt.boxplot`).
    - Results can be saved as CSV files if needed.

    Raises
    ------
    ValueError
        If `pattern` is not `"recruited"`, `"activated"`, or `"inhibited"`, or if `method` is not `"ratio"` or `"delta"`.
    AttributeError
        If `fraction_resp()` fails due to missing attributes in `recs`.

    Example
    -------
    >>> resp_contrast(pattern="activated", stim_ampl=[10, 20], method="ratio", ylim=[0, 5])
    >>> resp_contrast(pattern="inhibited", stim_ampl=[10, 20], method="delta", ylim=[0, 5])
    (Displays and saves a figure comparing the contrast of activated neurons between detected and undetected trials.)
    """
    pat_dict = {"recruited": 0, "activated": 1, "inhibited": -1}
    fig, axs = plt.subplots(1,2,figsize=(12,8), constrained_layout=True)
    for i, type in enumerate(["EXC", "INH"]):
        wt_det, ko_det, wt_undet, ko_undet = fraction_resp(pattern=pat_dict[pattern], n_type=type, ko_hypo_only=True, stim_ampl=stim_ampl)
        if method == "ratio":
            wt_nan = np.logical_and(wt_det>0, wt_undet>0)
            ko_nan = np.logical_and(ko_det>0, ko_undet>0)
            wt_det, ko_det, wt_undet, ko_undet = wt_det[wt_nan], ko_det[ko_nan], wt_undet[wt_nan], ko_undet[ko_nan]
            wt = wt_det/wt_undet
            ko = ko_det/ko_undet
        elif method == "delta":
            wt = wt_det - wt_undet
            ko = ko_det - ko_undet
        ppt.boxplot(axs[i], wt, ko, ylabel=f"{method} nb neuron Hit/Miss", title=type, paired=False, ylim=ylim)
        fig.suptitle(f"Comparaison of {method} of {pattern} neurons between detected and undetected trials for {stim_ampl} stimulus", fontsize=10)
        # excel_df_WT = pd.DataFrame(data={"WT": wt})
        # excel_df_KO = pd.DataFrame(data={"Hypo": ko})
        # excel_df_WT.to_csv(f"{server_address}data/{method}_{type}_{stim_ampl}_{pattern}_WT.csv", sep=",")
        # excel_df_KO.to_csv(f"{server_address}data/{method}_{type}_{stim_ampl}_{pattern}_KO.csv", sep=",")
        title = f"{method}({pattern})_{stim_ampl}_amp"
        fig.canvas.manager.set_window_title(title)
    plt.show()


def plot_neuron_perc_amp(recs, pattern="recruited", detected_trials=True, undetected_trials=True, ylim=[], sign_exc_inh=[[], []],
                         transformation=None, normality=[False, False], homogeneity=[False, False]):
    pat_dict = {"recruited": 0, "activated": 1, "inhibited": -1}
    assert pattern in pat_dict.keys()
    assert detected_trials or undetected_trials
    trials_name = "all" if (detected_trials and undetected_trials) else "detected" if detected_trials else "undetected" if undetected_trials else "none"
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 12), constrained_layout=True)
    amps = np.arange(start=2, stop=13, step=2)
    results = {}
    for i, n_type in enumerate(["EXC", "INH"]):
        rows = []
        for rec in recs:
            # Computing the total number of neurons
            total_n = rec.zscore_exc.shape[0] if n_type == "EXC" else rec.zscore_inh.shape[0]
            # Retrieving the responsivity matrix
            resp_mat = np.array(rec.matrices[n_type]["Responsivity"])
            # For each amplitude, computing the number of recruited neurons
            for amp in amps:
                if detected_trials and undetected_trials:
                    stim_filter = rec.stim_ampl == amp
                elif detected_trials:
                    stim_filter = np.logical_and(rec.detected_stim, rec.stim_ampl == amp)
                elif undetected_trials:
                    stim_filter = np.logical_and(np.invert(rec.detected_stim), rec.stim_ampl == amp)
                trials = resp_mat[:, stim_filter]
                if pattern != "recruited":
                    trials[trials != pat_dict[pattern]] = 0
                recruited_det = np.mean(np.count_nonzero(trials, axis=0))
                # The no-go trials are used to normalized
                # trials_no_go = resp_mat[:, rec.stim_ampl_filter(stim_ampl=[0], include_no_go=True)]
                # trials_no_go[trials_no_go != 1] = 0
                # recruited_no_go = np.mean(np.count_nonzero(trials_no_go, axis=0))
                # recruited_det -= recruited_no_go
                # recruited_det = 0 if recruited_det < 0 else recruited_det
                perc_n_det = (recruited_det / total_n) * 100
                row = {"ID": rec.filename, "Genotype": rec.genotype, "Amplitude": amp, f"perc_{n_type}": perc_n_det}
                rows.append(row)
        data = pd.DataFrame(rows)
        data_nan = data.fillna(0)
        test, post_hoc = ppt.curveplot(ax[i], data_nan, between="Genotype", within="Amplitude", variable=f"perc_{n_type}",
                                       title=f"Percentage of {pattern} {n_type} neurons",
                                       ylabel=None, xlabel=None, ylim=None, colors=["#c57c9a", "firebrick", "#326993"],
                                       id_display=True, legend_display=True, qq_show=True, transformation=transformation, consider_normality=normality[i],
                                       consider_homogeneity=homogeneity[i])
        results[f"data_{n_type}"] = data_nan
        results[f"test_{n_type}"] = test
        results[f"post_{n_type}"] = post_hoc
    title = f"ampcurv_{pattern}_{trials_name}_trials"
    fig.suptitle(f"Percentage of {pattern} neurons for {trials_name} trials", fontsize=15)
    fig.canvas.manager.set_window_title(title)
    plt.show()
    return results


# endregion ============================================================================================================


if __name__ == '__main__':
    ### Initialisation of recs instances ###
    directory = "C:/Users/cvandromme/Desktop/Data/"
    roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper/"
    roi_info = pd.read_excel(roi_path)
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]
    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec
    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}
    # ====== Building a summary table ======
    summary = []
    for rec in recs.values():
        summary.append({"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.session_threshold,
                        "n_trials": len(rec.detected_stim), "n_EXC": len(rec.zscore_exc), "n_INH": len(rec.zscore_inh)})
    summary = pd.DataFrame(summary)

    # ====== Response features ======
    for rec in recs.values():
        rec.peak_delay_amp()
        rec.auc()
    full_data = get_features(recs.values())
    data = full_data[full_data["ID"] != 5886]
    compare_sub_supra_within(data, behavior_filter=None, genotype="WT", comparison="all")
    compare_sub_supra_between(data, behavior_filter=False, gp1="KO", gp2="KO-Hypo", gp1_amps="mean", gp2_amps="threshold", colors=[ppt.ko_color, ppt.hypo_color])
    det, undet = compare_det_undet(data, genotype="KO-Hypo", amplitude="all")
    mean_det = np.mean(det.drop(columns="Genotype"), axis=0)
    mean_undet = np.mean(undet.drop(columns="Genotype"), axis=0)


    # ====== Responsivity ======
    # neurons = nb_neurons(recs.values())
    # plot_neuron_frac_wt_ko(pattern=0, ko_hypo_only=True, stim_ampl="all", no_go_normalize=True, ylim=[0, 60])
    # plot_neuron_frac_det_undet(pattern=-1, ko_hypo_only=True, stim_ampl="session_threshold", no_go_normalize=True, ylim=[0, 60])
    # resp_contrast(pattern="recruited", stim_ampl="session_threshold", method="delta", ylim=[-10, 30])
    #
    # results = plot_neuron_perc_amp(recs.values(), pattern="inhibited", detected_trials=False, undetected_trials=True, ylim=[0, 30],
    #                                transformation="yeojohnson", normality=[False, False], homogeneity=[False, False])
    # post_EXC = results["post_EXC"]
    # post_EXC_btw = post_EXC["between"]
    # post_INH = results["post_INH"]
    # post_INH_btw = post_INH["between"]



