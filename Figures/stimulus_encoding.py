# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as ss
from multiprocessing import cpu_count, pool
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols, mixedlm
from tqdm import tqdm

import percephone.core.recording as pc
import percephone.plts.stats as ppt
import percephone.plts.style as sty
# endregion ============================================================================================================
# region ======================================== Response features ====================================================

def get_features(recs, amp_delay=True, auc=False):
    """
    Extract neuronal response features from multiple recordings and compile them into a structured DataFrame.
    For each trial in each recording, compute the percentage of recruited neurons, optionally their mean peak
    amplitude, peak delay, and AUC-based measures, separated by neuron type and response pattern.

    Parameters
    ----------
    recs : list
        List of recording objects containing neuronal response data and metadata. Each recording must implement
        attributes (`detected_stim`, `stim_ampl`, `filename`, `genotype`, `session_threshold`, `bounded_x0`)
        and methods (`get_perc_resp`, `get_mean_param`).
    amp_delay : bool, default=True
        Whether to include mean peak amplitude and peak delay features for responsive neurons
    auc : bool, default=False
        Whether to include mean AUC and cumulative AUC features for responsive neurons (only considered if
        `amp_delay` is True)

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a trial in a recording, containing:
            - Recording metadata (ID, Genotype, threshold, bounded_x0, Trial)
            - Behavioral outcome and stimulus amplitude
            - Neuronal recruitment percentages by neuron type and response pattern
            - (Optional) mean peak amplitude and peak delay
            - (Optional) mean AUC and cumulative AUC
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
            "rec_EXC_perc": rec.get_perc_resp(pattern=2, n_type="EXC"),
            "act_INH_perc": rec.get_perc_resp(pattern=1, n_type="INH"),
            "inh_INH_perc": rec.get_perc_resp(pattern=-1, n_type="INH"),
            "rec_INH_perc": rec.get_perc_resp(pattern=2, n_type="INH")}
        if amp_delay:
            feature_vectors.update({
                # Mean peak amplitude for responsive neurons
                "act_EXC_amp": rec.get_mean_param(pattern=1, n_type="EXC", parameter="Peak_amplitude"),
                "inh_EXC_amp": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="Peak_amplitude"),
                "rec_EXC_amp": rec.get_mean_param(pattern=2, n_type="EXC", parameter="Peak_amplitude"),
                "act_INH_amp": rec.get_mean_param(pattern=1, n_type="INH", parameter="Peak_amplitude"),
                "inh_INH_amp": rec.get_mean_param(pattern=-1, n_type="INH", parameter="Peak_amplitude"),
                "rec_INH_amp": rec.get_mean_param(pattern=2, n_type="INH", parameter="Peak_amplitude"),
                # Mean peak delay for responsive neurons
                "act_EXC_delay": rec.get_mean_param(pattern=1, n_type="EXC", parameter="Peak_delay"),
                "inh_EXC_delay": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="Peak_delay"),
                "rec_EXC_delay": rec.get_mean_param(pattern=2, n_type="EXC", parameter="Peak_delay"),
                "act_INH_delay": rec.get_mean_param(pattern=1, n_type="INH", parameter="Peak_delay"),
                "inh_INH_delay": rec.get_mean_param(pattern=-1, n_type="INH", parameter="Peak_delay"),
                "rec_INH_delay": rec.get_mean_param(pattern=2, n_type="INH", parameter="Peak_delay")})
            if auc:
                feature_vectors.update({
                    # Mean AUC for responsive neurons
                    "act_EXC_auc": rec.get_mean_param(pattern=1, n_type="EXC", parameter="AUC"),
                    "inh_EXC_auc": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="AUC"),
                    "act_INH_auc": rec.get_mean_param(pattern=1, n_type="INH", parameter="AUC"),
                    "inh_INH_auc": rec.get_mean_param(pattern=-1, n_type="INH", parameter="AUC"),
                    # Mean cumulted AUC for responsive neurons
                    "act_EXC_cum_auc": rec.get_mean_param(pattern=1, n_type="EXC", parameter="cum_AUC"),
                    "inh_EXC_cum_auc": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="cum_AUC"),
                    "act_INH_cum_auc": rec.get_mean_param(pattern=1, n_type="INH", parameter="cum_AUC"),
                    "inh_INH_cum_auc": rec.get_mean_param(pattern=-1, n_type="INH", parameter="cum_AUC")})
        nb_trials = len(feature_vectors["behavior"])
        for trial_id in range(nb_trials):
            row = {"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.session_threshold, "bounded_x0": rec.bounded_x0, "Trial": trial_id}
            for feature, vector in feature_vectors.items():
                row[feature] = vector[trial_id]
            rows.append(row)
    return pd.DataFrame(rows)


# Not used in the final paper
def compare_x0_psy_threshold(recs):
    """
    Compare the difference between psychometric x0 (x0_psy) and session threshold across genotypes.
    For each recording, compute the absolute difference between x0_psy and threshold, and visualize
    the distributions using customized boxplots for pairwise genotype comparisons.
    → Useless because threshold as to be set to 12 for KO-Hypo even if the computed x0_psy was way greater

    Parameters
    ----------
    recs : list
        List of recording objects. Each recording must provide attributes
        (`filename`, `genotype`, `x0_psy`, `session_threshold`).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing, for each recording:
            - Recording ID
            - Genotype
            - Psychometric x0 (x0_psy)
            - Session threshold
            - Absolute difference (Delta) between x0_psy and threshold
    """
    colors_dict = {"WT": sty.wt_color, "KO": sty.ko_color, "KO-Hypo": sty.hypo_color}
    rows = []
    for rec in recs:
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "x0": rec.x0_psy, "Threshold": rec.session_threshold})
    data = pd.DataFrame(rows)
    data["Delta"] = abs(data["x0"] - data["Threshold"])
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8), constrained_layout=True)
    for ax_id, (geno1, geno2) in enumerate([["WT", "KO-Hypo"], ["WT", "KO"], ["KO", "KO-Hypo"]]):
        ppt.boxplot(ax[ax_id], data[data["Genotype"] == geno1]["Delta"].values, data[data["Genotype"] == geno2]["Delta"].values,
                    ylabel="Delta x0_psy/Threshold", paired=False, title=f"{geno1}/{geno2}", ylim=[], colors=[colors_dict[geno1], colors_dict[geno2]],
                    det_marker=False, force_markers_identity=False)
    plt.show()
    return data


def filter_amplitude(data, amplitude="all", no_go=False):
    """
    Filter trials from a dataset based on their stimulus amplitude relative to the detection threshold.
    Supports selecting specific conditions (threshold, sub-threshold, supra-threshold, all below, all above)
    or custom amplitude values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least the columns `amplitude` and `threshold`
    amplitude : {"all", "threshold", "sub", "supra", "all_sub", "all_supra"} or list of int, default="all"
        Filtering condition:
            - "all"       : keep all amplitudes
            - "threshold" : keep only trials at threshold amplitude
            - "sub"       : keep only trials at (threshold - 2)
            - "supra"     : keep only trials at (threshold + 2)
            - "all_sub"   : keep all trials below threshold
            - "all_supra" : keep all trials above threshold
            - list        : keep trials matching specific amplitude values
    no_go : bool, default=False
        If False, exclude "no-go" trials (amplitude = 0). If True, keep them

    Returns
    -------
    pd.DataFrame or None
        Filtered DataFrame. Returns None if an invalid `amplitude` argument is provided
    """
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
    """
    Retrieve threshold and non-threshold trials for a given genotype, with optional filtering by behavior
    and different methods for selecting comparison amplitudes (sub-/supra-threshold or genotype-level
    mean/median thresholds).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least the columns:
        `Genotype`, `ID`, `amplitude`, `threshold`, `bounded_x0`, and optionally `behavior`
    behavior_filter : bool or None, default=None
        If provided, keep only trials matching the specified behavior.
        If None, all behaviors are included
    genotype : str, default="WT"
        Genotype to filter the dataset on
    comparison : {"sub", "supra", "rounded_mean_genotype", "real_mean_genotype",
                  "rounded_median_genotype", "real_median_genotype"} or list of int, default="sub"
        Method to select non-threshold trials:
            - "sub"                     : threshold - 2
            - "supra"                   : threshold + 2
            - "rounded_mean_genotype"   : rounded mean threshold amplitude across genotype
            - "real_mean_genotype"      : rounded mean bounded_x0 across genotype
            - "rounded_median_genotype" : rounded median threshold amplitude across genotype
            - "real_median_genotype"    : rounded median bounded_x0 across genotype
            - list of int               : custom amplitude values

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - threshold_trials     : DataFrame of averaged trials at threshold amplitude
        - non_threshold_trials : DataFrame of averaged trials at comparison amplitude(s)
    """
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
    if comparison == "rounded_mean_genotype":
        amplitude = [2 * round(threshold_trials["threshold"].mean() / 2) if 0 < threshold_trials["threshold"].mean() < 12 else 12]
    elif comparison == "real_mean_genotype":
        amplitude = [2 * round(threshold_trials["bounded_x0"].mean() / 2) if 0 < threshold_trials["bounded_x0"].mean() < 12 else 12]
        print(f"Mean bounded x0: {threshold_trials["bounded_x0"].mean()}")
    elif comparison == "rounded_median_genotype":
        amplitude = [2 * round(threshold_trials["threshold"].median() / 2) if 0 < threshold_trials["threshold"].median() < 12 else 12]
    elif comparison == "real_median_genotype":
        amplitude = [2 * round(threshold_trials["bounded_x0"].median() / 2) if 0 < threshold_trials["bounded_x0"].median() < 12 else 12]
    else:
        amplitude = comparison
    non_threshold_trials = filter_amplitude(data, amplitude=amplitude).groupby(grouping_cols, as_index=False).mean()
    # Filtering out the no-go trials
    non_threshold_trials = non_threshold_trials[non_threshold_trials["amplitude"] != 0]
    return threshold_trials, non_threshold_trials


def compare_sub_supra_within(data, behavior_filter=None, genotype="WT", comparison="sub"):
    """
    Compare neuronal features between threshold and non-threshold trials within a given genotype.
    The function extracts trials at threshold and at a specified comparison amplitude, ensures matching
    animal IDs between conditions, and visualizes paired comparisons for all feature variables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing trial-level features. Must include columns:
        `ID`, `Genotype`, `amplitude`, `threshold`, `bounded_x0`, and optionally `behavior`,
        along with feature columns to be compared
    behavior_filter : bool or None, default=None
        If provided, keep only trials matching the specified behavior.
        If None, all behaviors are included
    genotype : str, default="WT"
        Genotype to filter the dataset on
    comparison : {"sub", "supra", "rounded_mean_genotype", "real_mean_genotype",
                  "rounded_median_genotype", "real_median_genotype"} or list of int, default="sub"
        Method to select non-threshold trials (see `get_sub_supra_threshold` for details)

    Returns
    -------
    None
        Displays a multi-panel figure (4x4 grid) of boxplots comparing threshold vs non-threshold trials
        across all feature variables for the specified genotype
    """
    threshold_trials, non_threshold_trials = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=genotype, comparison=comparison)
    # Asserting that the match between threshold and non-threshold trials for each animal
    common_IDs = set(threshold_trials["ID"]).intersection(non_threshold_trials["ID"])
    threshold_trials = threshold_trials[threshold_trials["ID"].isin(common_IDs)]
    non_threshold_trials = non_threshold_trials[non_threshold_trials["ID"].isin(common_IDs)]
    threshold_trials = threshold_trials.sort_values("ID").reset_index(drop=True)
    non_threshold_trials = non_threshold_trials.sort_values("ID").reset_index(drop=True)
    # Plotting the comparisons
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold", "bounded_x0"]
    variables = [col for col in data.columns if col not in not_variables]
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(24, 32), constrained_layout=True)
    axes_flat = axes.flatten()
    for variable, ax in zip(variables, axes_flat):
        ppt.boxplot(ax, threshold_trials[variable], non_threshold_trials[variable], ylabel=variable, paired=True, title="", ylim=[],
                    colors=sty.color_dict[genotype], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison in {genotype} of threshold trials and {comparison} trials"
                 f"\n[behavior filter={behavior_filter}] n={len(non_threshold_trials)}", fontsize=20)
    title = f"comp_{genotype}_threshold_{comparison}_{behavior_filter}"
    fig.canvas.manager.set_window_title(title)
    # if save_fig:
        # plt.savefig(f"{server_address}Threshold_analysis/{title}.pdf")
    plt.show()


def compare_sub_supra_between(data, behavior_filter=None, gp1="WT", gp2="KO-Hypo", gp1_amps="sub", gp2_amps="sub",
                              colors=[sty.wt_color, sty.hypo_color]):
    """
    Compare neuronal features between two genotypes at threshold or non-threshold amplitudes.
    The function extracts trials for each genotype based on specified amplitude conditions,
    aligns them, and visualizes comparisons of feature distributions across groups.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing trial-level features. Must include columns:
        `ID`, `Genotype`, `amplitude`, `threshold`, `bounded_x0`, and optionally `behavior`,
        along with feature columns to be compared
    behavior_filter : bool or None, default=None
        If provided, keep only trials matching the specified behavior.
        If None, all behaviors are included
    gp1 : str, default="WT"
        Name of the first genotype to compare
    gp2 : str, default="KO-Hypo"
        Name of the second genotype to compare
    gp1_amps : {"threshold", "sub", "supra", "rounded_mean_genotype", "real_mean_genotype",
                "rounded_median_genotype", "real_median_genotype"} or list of int, default="sub"
        Amplitude selection rule for the first genotype (see `get_sub_supra_threshold` for details)
    gp2_amps : {"threshold", "sub", "supra", "gp1_threshold",
                "rounded_mean_genotype", "real_mean_genotype",
                "rounded_median_genotype", "real_median_genotype"} or list of int, default="sub"
        Amplitude selection rule for the second genotype. If `"gp1_threshold"`, uses the amplitude
        selected for gp1
    colors : list of str, default=[sty.wt_color, sty.hypo_color]
        Colors used to represent the two genotypes in the boxplots

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - data_gp1 : DataFrame of selected trials for the first genotype
        - data_gp2 : DataFrame of selected trials for the second genotype
    """
    data = data.drop(columns=["Trial"])
    gp1_threshold, gp1_non_threshold = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp1, comparison=gp1_amps if gp1_amps != "threshold" else "sub")
    if (gp1_amps in ["rounded_mean_genotype", "real_mean_genotype", "rounded_median_genotype", "real_median_genotype"] and gp2_amps == "gp1_threshold"):
        gp2_amps = [gp1_non_threshold["amplitude"].values[0]]
    gp2_threshold, gp2_non_threshold = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp2, comparison=gp2_amps if gp2_amps != "threshold" else "sub")
    # Plotting the comparisons
    data_gp1 = gp1_threshold if gp1_amps == "threshold" else gp1_non_threshold
    data_gp2 = gp2_threshold if gp2_amps == "threshold" else gp2_non_threshold
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold", "bounded_x0"]
    variables = [col for col in data.columns if col not in not_variables]
    variables = [var for var in variables if var.split("_")[-1] != "auc"]
    det_marker = True if behavior_filter in [None, True] else False
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(36, 24), constrained_layout=True)
    axes_flat = axes.flatten()
    for variable, ax in zip(variables, axes_flat):
        if variable.split("_")[-1] == "perc":
            ylim = [0, 80]
        elif variable.split("_")[-1] == "delay":
            ylim = [0, 15]
        elif variable.split("_")[-1] in ["amp", "auc"]:
            ylim = [-5, 5]
        if variable.split("_")[-1] != "auc":
            ppt.boxplot(ax, data_gp1[variable], data_gp2[variable], ylabel=variable, paired=False, title="", ylim=ylim,
                        colors=colors, det_marker=det_marker, force_markers_identity=False)
    fig.suptitle(f"Comparison between {gp1_amps} trials of {gp1} & {gp2_amps} trials of {gp2}"
                 f"\n[behavior filter={behavior_filter}] n={len(data_gp1)}{gp1}/{len(data_gp2)}{gp2}", fontsize=20)
    title = f"comp_{gp1_amps}({gp1})_{gp2_amps}({gp2})_{behavior_filter}"
    fig.canvas.manager.set_window_title(title)
    # if save_fig:
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/Including 5886/{title}.pdf", format="pdf")
    # plt.show()
    return data_gp1, data_gp2


# Not used in the final paper
def compare_sub_supra_deltas(data, behavior_filter=None, gp1="WT", gp2="KO-Hypo", delta="both"):
    """
    Compare feature differences (deltas) between threshold, sub-threshold, and supra-threshold trials
    across two genotypes. Instead of comparing raw trial values, this function computes the absolute
    difference between conditions (sub–threshold, supra–threshold, or sub–supra) and visualizes them
    for each feature.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing trial-level features. Must include columns:
        `ID`, `Genotype`, `amplitude`, `threshold`, `bounded_x0`, and optionally `behavior`,
        along with feature columns to be compared
    behavior_filter : bool or None, default=None
        If provided, keep only trials matching the specified behavior.
        If None, all behaviors are included
    gp1 : str, default="WT"
        Name of the first genotype to compare
    gp2 : str, default="KO-Hypo"
        Name of the second genotype to compare
    delta : {"sub", "supra", "both"}, default="both"
        Type of delta to compare:
            - "sub"   : difference between sub-threshold and threshold trials
            - "supra" : difference between supra-threshold and threshold trials
            - "both"  : difference between supra-threshold and sub-threshold trials

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - delta1 : DataFrame of absolute deltas for the first genotype
        - delta2 : DataFrame of absolute deltas for the second genotype
    """
    # Retrieving the data from threshold, sub (and supra threshold) for both groups
    gp1_threshold, gp1_sub = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp1, comparison="sub")
    _, gp1_supra = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp1, comparison="supra")
    gp2_threshold, gp2_sub = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp2, comparison="sub")
    _, gp2_supra = get_sub_supra_threshold(data, behavior_filter=behavior_filter, genotype=gp2, comparison="supra")
    # Setting the ID as index to perform the computation of deltas by subtracting dataframes
    gp1_threshold = gp1_threshold.set_index('ID').drop(columns=["Genotype", "behavior"])
    gp1_sub = gp1_sub.set_index('ID').drop(columns=["Genotype", "behavior"])
    gp1_supra = gp1_supra.set_index('ID').drop(columns=["Genotype", "behavior"])
    gp2_threshold = gp2_threshold.set_index('ID').drop(columns=["Genotype", "behavior"])
    gp2_sub = gp2_sub.set_index('ID').drop(columns=["Genotype", "behavior"])
    gp2_supra = gp2_supra.set_index('ID').drop(columns=["Genotype", "behavior"])
    # !!! Taking the absolute values for the deltas !!!
    # Computing the delta sub / threshold
    diff_sub1 = gp1_threshold.subtract(gp1_sub).reset_index().abs()
    diff_sub2 = gp2_threshold.subtract(gp2_sub).reset_index().abs()
    # Computing the delta supra / threshold
    diff_supra1 = gp1_supra.subtract(gp1_threshold).reset_index().abs()
    diff_supra2 = gp2_supra.subtract(gp2_threshold).reset_index().abs()
    # Computing the delta sub / supra
    diff_both1 = gp1_supra.subtract(gp1_sub).reset_index().abs()
    diff_both2 = gp2_supra.subtract(gp2_sub).reset_index().abs()

    # Plotting the deltas
    color_dict = {"WT": sty.wt_color, "KO-Hypo": sty.hypo_color, "KO": sty.ko_color}
    if delta == "both":
        delta1 = diff_both1
        delta2 = diff_both2
    elif delta == "sub":
        delta1 = diff_sub1
        delta2 = diff_sub2
    elif delta == "supra":
        delta1 = diff_supra1
        delta2 = diff_supra2
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold", "bounded_x0"]
    variables = [col for col in data.columns if col not in not_variables]
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(24, 32), constrained_layout=True)
    axes_flat = axes.flatten()
    for variable, ax in zip(variables, axes_flat):
        ppt.boxplot(ax, delta1[variable], delta2[variable], ylabel=variable, paired=False, title="", ylim=[],
                    colors=[color_dict[gp1], color_dict[gp2]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison between deltas (sub/supra: {delta}) trials of {gp1} & {gp2}"
                 f"\n[behavior filter={behavior_filter}]", fontsize=20)
    fig.canvas.manager.set_window_title(f"delta_comp_{delta}_{behavior_filter}_{gp1}_{gp2}")
    plt.show()
    return delta1, delta2


def compare_det_undet(data_df, genotype="WT", amplitude="all"):
    """
    Compare feature values between detected and undetected trials, either within a single genotype
    or across WT vs KO under pharmacological conditions (DMSO/BMS). Trials can be filtered by amplitude
    before computing group-level averages and paired/unpaired comparisons are performed depending on
    the genotype specification.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing trial-level features. Must include columns:
        `ID`, `Genotype`, `behavior`, `amplitude`, `threshold`, and `bounded_x0`,
        along with feature columns to be compared
    genotype : str, default="WT"
        Defines the comparison to perform:
            - If set to a simple genotype (e.g. "WT", "KO", "KO-Hypo"), compares detected vs undetected
              trials within that group (paired comparison across matching IDs)
            - If set to a condition with suffix "det" or "undet" (e.g. "DMSO_det", "BMS_undet"),
              compares WT vs KO for detected or undetected trials under that pharmacological condition
              (unpaired comparison)
    amplitude : str or numeric, default="all"
        Amplitude filter applied to trials before averaging.
        Passed directly to `filter_amplitude`

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - gp1 : DataFrame of detected (or WT detected) trials
        - gp2 : DataFrame of undetected (or KO detected) trials
    """
    colors_dict = sty.color_dict.copy()
    colors_dict.update({"DMSO_det": [sty.wt_color, sty.hypo_color],
                        "DMSO_undet": [sty.wt_light_color, sty.hypo_light_color],
                        "BMS_det": [sty.wt_bms_color, sty.hypo_bms_color],
                        "BMS_undet": [sty.wt_bms_light_color, sty.hypo_bms_light_color]})
    grouping_cols = ["Genotype", "ID", "behavior"]
    data = data_df.drop(columns=["Trial"]).copy()
    # Filtering the amplitude
    if genotype[-3:] == "det":
        condition = genotype.split("_")[0]
        detection = genotype.split("_")[1]
        wt_data = data[data["Genotype"] == f"WT-{condition}"]
        ko_data = data[data["Genotype"] == f"KO-{condition}"]
        wt_ampl_data = filter_amplitude(wt_data, amplitude=amplitude, no_go=False).groupby(grouping_cols, as_index=False).mean()
        ko_ampl_data = filter_amplitude(ko_data, amplitude=amplitude, no_go=False).groupby(grouping_cols, as_index=False).mean()
        if detection == "det":
            gp1 = wt_ampl_data[wt_ampl_data["behavior"] == True]
            gp2 = ko_ampl_data[ko_ampl_data["behavior"] == True]
            trials = "detected"
            det_marker = True
            force_markers_identity = True
        else:
            gp1 = wt_ampl_data[wt_ampl_data["behavior"] == False]
            gp2 = ko_ampl_data[ko_ampl_data["behavior"] == False]
            trials = "undetected"
            det_marker = False
        paired = False
        suptitle = f"Comparison of {trials} trials in {condition} conditions between WT and KO"
    else:
        genotype_data = data[data["Genotype"] == genotype]
        ampl_data = filter_amplitude(genotype_data, amplitude=amplitude, no_go=False).groupby(grouping_cols, as_index=False).mean()
        gp1 = ampl_data[ampl_data["behavior"] == True].copy()
        gp2 = ampl_data[ampl_data["behavior"] == False].copy()
        # keep only matching IDs
        common_ids = np.intersect1d(gp1["ID"].values, gp2["ID"].values)
        gp1 = gp1[gp1["ID"].isin(common_ids)].sort_values("ID")
        gp2 = gp2[gp2["ID"].isin(common_ids)].sort_values("ID")
        assert np.all(gp1.ID.values == gp2.ID.values), f"The IDs do not match"
        det_marker = True
        paired = True
        suptitle = f"Comparison in {genotype} of detected trials and undetected trials"
    # Plotting the comparisons
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold", "bounded_x0"]
    variables = [col for col in data.columns if col not in not_variables]
    variables = [var for var in variables if var.split("_")[-1] != "auc"]
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(36, 24), constrained_layout=True)
    axes_flat = axes.flatten()
    for i, (variable, ax) in enumerate(zip(variables, axes_flat)):
        if variable.split("_")[-1] == "perc":
            ylim = [-10, 80]
        elif variable.split("_")[-1] == "delay":
            ylim = [0, 15]
        elif variable.split("_")[-1] in ["amp", "auc"]:
            ylim = [-5, 5]
        if paired:
            na_filter = (~np.isnan(gp1[variable].values) & ~np.isnan(gp2[variable].values))
            gp1_plot = gp1[variable].values[na_filter]
            gp2_plot = gp2[variable].values[na_filter]
        else:
            gp1_plot = gp1[variable].values
            gp2_plot = gp2[variable].values
        ppt.boxplot(ax, gp1_plot, gp2_plot, ylabel=variable, paired=paired, title="", ylim=ylim,
                    colors=colors_dict[genotype], det_marker=det_marker, force_markers_identity=False)
    used = len(variables)
    for extra_ax in axes_flat[used:]:
        extra_ax.set_axis_off()
    fig.suptitle(f"{suptitle}\n[amplitude filter={amplitude}]", fontsize=15)
    title = f"comp_{genotype}_det_undet_{amplitude}"
    fig.canvas.manager.set_window_title(title)
    # if save_fig:
    # plt.savefig(f"{server_address}stimulus_encoding/{title}.pdf")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/{title}.pdf", format="pdf")
    # plt.show()
    return gp1, gp2


def compare_det_undet_between(data_df, gp1="WT", gp2="KO-Hypo", amplitude="all", behavior_filter=None):
    """
    Compare feature values between two genotypes (gp1 vs gp2) for either all trials, only detected
    (hits), or only undetected (misses). Trials are filtered by amplitude and averaged per subject
    before unpaired comparisons are plotted.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing trial-level features. Must include columns:
        `ID`, `Genotype`, `behavior`, `amplitude`, `threshold`, and `bounded_x0`,
        along with feature columns to be compared.
    gp1 : str, default="WT"
        First genotype group (e.g. "WT", "KO", "KO-Hypo").
    gp2 : str, default="KO-Hypo"
        Second genotype group to compare against gp1.
    amplitude : str or numeric, default="all"
        Amplitude filter applied to trials before averaging.
        Passed directly to `filter_amplitude`.
    behavior_filter : bool or None, default=None
        If provided, keep only trials matching the specified behavior.
        If None, all behaviors are included

    Returns
    -------
    tuple of (str, str)
        The genotype identifiers (`gp1`, `gp2`) used in the comparison.
    """
    grouping_cols = ["Genotype", "ID", "behavior"]
    data = data_df.drop(columns=["Trial"]).copy()
    # Filtering the amplitude
    gp1_full_data = data[data["Genotype"] == gp1]
    gp1_all = filter_amplitude(gp1_full_data, amplitude=amplitude, no_go=False).groupby(grouping_cols, as_index=False).mean()
    gp1_hit = gp1_all[gp1_all["behavior"] == True].copy()
    gp1_miss = gp1_all[gp1_all["behavior"] == False].copy()
    gp2_full_data = data[data["Genotype"] == gp2]
    gp2_all = filter_amplitude(gp2_full_data, amplitude=amplitude, no_go=False).groupby(grouping_cols, as_index=False).mean()
    gp2_hit = gp2_all[gp2_all["behavior"] == True].copy()
    gp2_miss = gp2_all[gp2_all["behavior"] == False].copy()
    gp_dict = {None: [gp1_all, gp2_all], True: [gp1_hit, gp2_hit], False: [gp1_miss, gp2_miss]}
    # Plotting the comparisons
    not_variables = ["ID", "Genotype", "behavior", "amplitude", "threshold", "bounded_x0"]
    variables = [col for col in data.columns if col not in not_variables]
    variables = [var for var in variables if var.split("_")[-1] != "auc"]
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(36, 24), constrained_layout=True)
    axes_flat = axes.flatten()
    det_marker = True if behavior_filter in ([None, True]) else False
    color_behavior_id = 0 if behavior_filter in ([None, True]) else 1
    for i, (variable, ax) in enumerate(zip(variables, axes_flat)):
        if variable.split("_")[-1] == "perc":
            ylim = [-10, 80]
        elif variable.split("_")[-1] == "delay":
            ylim = [0, 15]
        elif variable.split("_")[-1] in ["amp", "auc"]:
            ylim = [-5, 5]
        if variable.split("_")[-1] != "auc":
            ppt.boxplot(ax, gp_dict[behavior_filter][0][variable].values, gp_dict[behavior_filter][1][variable].values,
                        ylabel=variable, paired=False, title="", ylim=ylim,
                        colors=[sty.color_dict[gp1][color_behavior_id], sty.color_dict[gp2][color_behavior_id]], det_marker=det_marker, force_markers_identity=False)
    used = len(variables)
    for extra_ax in axes_flat[used:]:
        extra_ax.set_axis_off()
    fig.suptitle(f"Comparison of {"all" if behavior_filter is None else behavior_filter} trials between {gp1} and {gp2}\n[amplitude filter={amplitude}]", fontsize=15)
    title = f"comp_{gp1}_{gp2}_det_undet_{amplitude}_{behavior_filter}"
    fig.canvas.manager.set_window_title(title)
    # if save_fig:
    # plt.savefig(f"{server_address}stimulus_encoding/{title}.pdf")
    # plt.show()
    return gp1, gp2


# Not used, previous version
def group_comp_param(recs, parameter, ko_hypo_only=False, stim_ampl="all", ylim=[]):
    """
    Compare a neuronal parameter between WT and KO groups across neuron and response types,
    separately for detected and undetected stimuli.

    Parameters
    ----------
    recs : dict
        Dictionary of recording objects. Each object must provide:
        - `genotype` (str): recording genotype ("WT", "KO", "KO-Hypo").
        - `matrices` (dict): nested dict with neuronal parameters and responsivity.
        - `stim_ampl_filter` (callable): function to filter trials by amplitude.
        - `detected_stim` (np.ndarray): boolean mask of detected stimuli.
    parameter : str
        Parameter name to analyze (e.g., "Peak_delay", "Peak_amplitude").
        Must exist in `rec.matrices[neuron_type][parameter]`.
    ko_hypo_only : bool, default=False
        If True, restrict KO comparisons to KO-Hypo. Otherwise pool KO and KO-Hypo.
    stim_ampl : str or list, default="all"
        Stimulation amplitude(s) to include. Passed to `stim_ampl_filter`.
    ylim : list, default=[]
        Y-axis limits for plots. If empty, determined automatically.
        For inhibitory response types with negative values, limits are inverted.

    Returns
    -------
    None
        Displays a matplotlib figure (2x4 subplots) with WT vs KO boxplots,
        comparing detected and undetected responses across EXC/INH neurons.
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
                color_ko = sty.hypo_color
                light_color_ko = sty.hypo_light_color
            else:
                ko_type = "(KO + KO-Hypo)"
                color_ko = sty.all_ko_color
                light_color_ko = sty.all_ko_light_color

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
            ppt.boxplot(axs[i, 2*j], wt_det, ko_det, paired=False, ylabel=f"{parameter}", ylim=auto_ylim, colors=[sty.wt_color, color_ko], det_marker=True)
            ppt.boxplot(axs[i, 2*j+1], wt_undet, ko_undet, paired=False, ylabel=f"{parameter}", ylim=auto_ylim, colors=[sty.wt_light_color, light_color_ko], det_marker=False)
            axs[i, 2*j].set_title(f"Det {neuron_type}({response_type})")
            axs[i, 2*j+1].set_title(f"Undet {neuron_type}({response_type})")
    title = f"{parameter}[WT_{ko_type}]_{stim_ampl}_amp"
    fig.suptitle(f"Mean {parameter} between WT and {ko_type}. Amplitude(s): {stim_ampl}", fontsize=10)
    fig.canvas.manager.set_window_title(title)
    plt.show()


# Not used, previous version
def det_comp_param(recs, parameter, stim_ampl="all", ylim=[]):
    """
    Compare a neuronal parameter between detected and undetected stimuli,
    across WT, KO (WT + KO-Hypo), and KO-Hypo groups.

    Parameters
    ----------
    recs : dict
        Dictionary of recording objects. Each object must provide:
        - `genotype` (str): recording genotype ("WT", "KO", "KO-Hypo").
        - `matrices` (dict): nested dict with neuronal parameters and responsivity.
        - `stim_ampl_filter` (callable): function to filter trials by amplitude.
        - `detected_stim` (np.ndarray): boolean mask of detected stimuli.
    parameter : str
        Parameter name to analyze (e.g., "Peak_delay", "Peak_amplitude").
        Must exist in `rec.matrices[neuron_type][parameter]`.
    stim_ampl : str or list, default="all"
        Stimulation amplitude(s) to include. Passed to `stim_ampl_filter`.
    ylim : list, default=[]
        Y-axis limits for plots. If empty, determined automatically.
        For inhibitory response types with negative values, limits are inverted.

    Returns
    -------
    None
        Displays a matplotlib figure (2x6 subplots) with paired boxplots
        comparing detected vs. undetected responses across neuron types (EXC/INH),
        response types (-1/1), and genotypes (WT, KO, KO-Hypo).
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
                        colors=[sty.wt_color, sty.wt_light_color])
            ppt.boxplot(axs[i, 3 * j + 1], ko_det, ko_undet, ylabel=parameter, paired=True,
                        title=f"KO + KO-Hypo {neuron_type}({response_type})", ylim=auto_ylim,
                        colors=[sty.all_ko_color, sty.all_ko_light_color])
            ppt.boxplot(axs[i, 3 * j + 2], hypo_det, hypo_undet, ylabel=parameter, paired=True,
                        title=f"KO-Hypo {neuron_type}({response_type})", ylim=auto_ylim,
                        colors=[sty.hypo_color, sty.hypo_light_color])
            # excel_df_WT = pd.DataFrame(data={"WT Det": wt_det, "WT Undet": wt_undet})
            # excel_df_KO = pd.DataFrame(data={"Hypo Det": hypo_det, "Hypo Undet": hypo_undet})
            # excel_df_WT.to_csv(f"{server_address}data/det_{parameter}_{neuron_type}_{stim_ampl}_{response_type}_WT.csv", sep=",")
            # excel_df_KO.to_csv(f"{server_address}data/det_{parameter}_{neuron_type}_{stim_ampl}_{response_type}_KO.csv", sep=",")
    title = f"{parameter}[det_undet]_{stim_ampl}_amp"
    fig.suptitle(f"Mean {parameter} for detected vs. undetected stimuli. Amplitude(s): {stim_ampl}", fontsize=10)
    fig.canvas.manager.set_window_title(title)
    plt.show()


def nogo_fa_cr(recs, condition=None):
    """
    Compare neuronal recruitment between genotypes during no-go trials, distinguishing False Alarms (FA) from Correct Rejections (CR).
    No-go trials are defined as trials with stimulus amplitude 0. A trial is considered a FA if a lick occurs within 75 ms
    after stimulus onset, otherwise it is a CR. The function computes the percentage of recruited neurons (EXC and INH,
    activated or inhibited) for each trial, aggregates the data by recording, genotype, and FA/CR outcome,
    and plots the results as boxplots in a 3x6 figure.

    Parameters
    ----------
    recs : list
        List of recording objects. Each recording must have the following attributes and methods:
        - filename : str, identifier of the recording
        - genotype : str, genotype label
        - stim_ampl : list or np.array, stimulus amplitudes per trial
        - stim_time : list or np.array, stimulus onset times per trial
        - lick_time : list or np.array, lick timestamps
        - get_perc_resp(pattern, n_type) : method returning the percentage of responsive neurons for a given pattern
          (1 = activation, -1 = inhibition) and neuron type ("EXC" or "INH")
    condition : str or None, optional
        Selects the genotype comparison:
        - None (default) → "WT" vs "KO-Hypo"
        - "DMSO" → "WT-DMSO" vs "KO-DMSO"
        - "BMS" → "WT-BMS" vs "KO-BMS"

    Returns
    -------
    pd.DataFrame
        Aggregated data containing mean recruitment metrics per recording, genotype, and FA/CR outcome.
        Columns include:
        - ID : str, recording identifier
        - Genotype : str
        - FA : bool, True for False Alarm, False for Correct Rejection
        - act_EXC_perc, inh_EXC_perc, rec_EXC_perc : float, percentage of activated, inhibited, or total responsive excitatory neurons
        - act_INH_perc, inh_INH_perc, rec_INH_perc : float, percentage of activated, inhibited, or total responsive inhibitory neurons
    """
    if condition is None:
        wt_geno = "WT"
        ko_geno = "KO-Hypo"
    elif condition == "DMSO":
        wt_geno = "WT-DMSO"
        ko_geno = "KO-DMSO"
    elif condition == "BMS":
        wt_geno = "WT-BMS"
        ko_geno = "KO-BMS"
    colors = [sty.color_dict[wt_geno][0], sty.color_dict[ko_geno][0]]
    rows = []
    for rec in recs:
        licks = rec.lick_time
        act_EXC_perc = rec.get_perc_resp(pattern=1, n_type="EXC")
        inh_EXC_perc = rec.get_perc_resp(pattern=-1, n_type="EXC")
        act_INH_perc = rec.get_perc_resp(pattern=1, n_type="INH")
        inh_INH_perc = rec.get_perc_resp(pattern=-1, n_type="INH")
        for trial_id, (trial_amp, time) in enumerate(zip(rec.stim_ampl, rec.stim_time)):
            if trial_amp == 0:
                # Defining if the no-go is a FA or CR
                time_diff_vector = licks - ([time] * len(licks))
                is_fa = np.any((time_diff_vector >= 0) & (time_diff_vector < 75))
                rows.append({"ID": rec.filename, "Genotype": rec.genotype, "FA": is_fa,
                             "act_EXC_perc": act_EXC_perc[trial_id], "inh_EXC_perc": inh_EXC_perc[trial_id],
                             "rec_EXC_perc": act_EXC_perc[trial_id] + inh_EXC_perc[trial_id],
                             "act_INH_perc": act_INH_perc[trial_id], "inh_INH_perc": inh_INH_perc[trial_id],
                             "rec_INH_perc": act_INH_perc[trial_id] + inh_INH_perc[trial_id]})
    data = pd.DataFrame(rows)
    gp_data_global = data.drop(columns="FA").groupby(["ID", "Genotype"], as_index=False).mean()
    gp_data = data.groupby(["ID", "Genotype", "FA"], as_index=False).mean()
    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(36, 24), constrained_layout=True)
    for col, metric in enumerate(gp_data.columns[-6:]):
        ppt.boxplot(ax[0, col], gp_data_global[gp_data_global["Genotype"] == wt_geno][metric].values,
                    gp_data_global[gp_data_global["Genotype"] == ko_geno][metric].values, ylabel=metric,
                    paired=False, title=f"All No-Go trials", ylim=[-10, 80],
                    colors=colors)
        ppt.boxplot(ax[1, col], gp_data[(gp_data["Genotype"] == wt_geno) & (gp_data["FA"] == True)][metric].values,
                    gp_data[(gp_data["Genotype"] == ko_geno) & (gp_data["FA"] == True)][metric].values, ylabel=metric,
                    paired=False, title=f"False Alarm", ylim=[-10, 80],
                    colors=colors)
        ppt.boxplot(ax[2, col], gp_data[(gp_data["Genotype"] == wt_geno) & (gp_data["FA"] == False)][metric].values,
                    gp_data[(gp_data["Genotype"] == ko_geno) & (gp_data["FA"] == False)][metric].values, ylabel=metric,
                    paired=False, title=f"Correct Rejection", ylim=[-10, 80],
                    colors=colors)
    fig.suptitle("Comparison between genotypes of neuronal recruitment during no-go trials")
    fig.canvas.manager.set_window_title("Recruitment NoGo")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/Recruitment_NoGo_{condition}.pdf", format="pdf")
    # plt.show()
    return gp_data


def delta_hit_miss_comp(feature_df, threshold_only=False, wt_threshold=False, condition=None):
    """
    Compute and compare the delta in neuronal recruitment and amplitude between hit and miss trials,
    as well as hit and no-go trials, across genotypes.

    The function calculates the difference (Δ) for each metric between hits and misses and between hits and
    no-go trials, aggregates by recording and genotype, and plots boxplots for visual comparison.
    Metrics include both recruitment percentages and mean amplitudes for excitatory (EXC) and inhibitory (INH) neurons.

    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame containing trial-level features for each recording, including:
        - 'ID' : str, recording identifier
        - 'Genotype' : str
        - 'behavior' : bool, True for hit, False for miss
        - 'amplitude' : float, stimulus amplitude
        - recruitment and amplitude metrics for EXC and INH neurons (e.g., 'act_EXC_perc', 'inh_INH_amp', etc.)
    threshold_only : bool, optional
        If True, only threshold trials are used for hit-miss comparisons (default is False).
    wt_threshold : bool, optional
        If True and threshold_only is True, WT threshold is set to 4 instead of using the session threshold (default is False).
    condition : str or None, optional
        Selects the genotype comparison:
        - None → 'WT' vs 'KO-Hypo'
        - 'DMSO' → 'WT-DMSO' vs 'KO-DMSO'
        - 'BMS' → 'WT-BMS' vs 'KO-BMS'

    Returns
    -------
    pd.DataFrame
        DataFrame containing delta metrics for each recording and genotype. Columns include:
        - 'ID' : str, recording identifier
        - 'Genotype' : str
        - 'Δ<metric>' : float, difference between hit and miss trials for each metric
        - 'Δ<metric>_nogo' : float, difference between hit and no-go trials for each metric

    This function also generates boxplots for each metric showing Δ Hit–Miss and Δ Hit–NoGo
    for WT vs KO, facilitating comparison of neuronal recruitment and amplitude changes.
    """
    if condition is None:
        wt_geno = "WT"
        ko_geno = "KO-Hypo"
    elif condition == "DMSO":
        wt_geno = "WT-DMSO"
        ko_geno = "KO-DMSO"
    elif condition == "BMS":
        wt_geno = "WT-BMS"
        ko_geno = "KO-BMS"
    colors = [sty.color_dict[wt_geno][0], sty.color_dict[ko_geno][0]]
    # Filtering out no go trials
    if threshold_only:
        if wt_threshold:
            data = feature_df[feature_df["amplitude"] == 4]
        else:
            data = feature_df[feature_df["amplitude"] == feature_df["threshold"]]
    else:
        data = feature_df[feature_df["amplitude"] != 0]
    data = data.drop(columns=["Trial"])
    hits = data[data["behavior"] == True]
    miss = data[data["behavior"] == False]
    nogo = feature_df[feature_df["amplitude"] == 0]
    metrics_perc = ['act_EXC_perc', 'inh_EXC_perc', 'rec_EXC_perc', 'act_INH_perc', 'inh_INH_perc', 'rec_INH_perc']
    metrics_amp = ['act_EXC_amp', 'inh_EXC_amp', 'rec_EXC_amp', 'act_INH_amp', 'inh_INH_amp', 'rec_INH_amp']
    metrics_list = metrics_perc + metrics_amp
    def mean_by_group(sub, label):
        gp = sub.groupby(['ID', 'Genotype'])[metrics_list].mean().add_suffix(f'_{label}')
        return gp
    gp_hit = mean_by_group(hits, 'hit')
    gp_miss = mean_by_group(miss, 'miss')
    gp_nogo = mean_by_group(nogo, 'nogo')
    joined = gp_hit.join(gp_miss, how='inner').join(gp_nogo, how='inner')
    delta = pd.DataFrame(index=joined.index)
    for metric in metrics_list:
        delta[f'Δ{metric}'] = joined[f'{metric}_hit'] - joined[f'{metric}_miss']
        delta[f'Δ{metric}_nogo'] = joined[f'{metric}_hit'] - joined[f'{metric}_nogo']
    delta = delta.reset_index()
    fig, axes = plt.subplots(4, 6, figsize=(36, 32), constrained_layout=True)
    for big_row, metrics, ylim in zip([0, 2], [metrics_perc, metrics_amp], [[-30, 50], [-5, 5]]):
        for col_idx, m in enumerate(metrics):
            ax1 = axes[big_row + 0, col_idx]
            ax2 = axes[big_row + 1, col_idx]
            wt = delta[delta['Genotype'] == wt_geno][f'Δ{m}']
            ko = delta[delta['Genotype'] == ko_geno][f'Δ{m}']
            wt_nogo = delta[delta['Genotype'] == wt_geno][f'Δ{m}_nogo']
            ko_nogo = delta[delta['Genotype'] == ko_geno][f'Δ{m}_nogo']
            ppt.boxplot(ax1, wt, ko, ylabel=m, title='Δ Hit–Miss',
                        colors=colors, paired=False, ylim=ylim)
                        # colors=[sty.wt_color, sty.hypo_color], paired=False, ylim=[-4, 4])
            ppt.boxplot(ax2, wt_nogo, ko_nogo, ylabel=m, title='Δ Hit–NoGo',
                        colors=colors, paired=False, ylim=ylim)
                        # colors=[sty.wt_color, sty.hypo_color], paired=False, ylim=[-4, 4])
    fig.suptitle(f"Delta in recruitment and amplitude of neurons ({wt_geno} vs. {ko_geno})\n[Only threshold={threshold_only} (WT={wt_threshold})]")
    # fig.suptitle(f"Delta in peak delay of neurons (WT vs. KO-Hypo)\n[Only threshold={threshold_only} (WT={wt_threshold})]")
    fig.canvas.manager.set_window_title(f"Recruitment Delta (cond={condition})[threshold={threshold_only}_WT={wt_threshold}]")
    # fig.canvas.manager.set_window_title(f"Peak delay Delta [threshold={threshold_only}_WT={wt_threshold}]")
    # plt.savefig(f"{server_address}stimulus_encoding/delta_{condition}_thre={threshold_only}_wt={wt_threshold}.pdf")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/delta_{condition}_thre={threshold_only}_wt={wt_threshold}.pdf",format="pdf")
    # plt.show()
    return delta

# endregion ============================================================================================================
# region ======================================== Responsivity =========================================================
def nb_neurons(recs):
    """
    Compute and compare the number and percentage of excitatory (EXC) and inhibitory (INH) neurons
    across WT and KO genotypes.

    The function iterates through all recordings, counts the number of EXC and INH neurons, calculates
    their respective percentages, aggregates results into a DataFrame, and generates boxplots comparing
    WT and KO (KO + KO-Hypo) for both absolute counts and percentages.

    Parameters
    ----------
    recs : list
        List of recording objects. Each recording object is expected to have the following attributes:
        - 'filename' : str, recording identifier
        - 'genotype' : str, genotype of the subject ('WT', 'KO', 'KO-Hypo')
        - 'zscore_exc' : np.ndarray, data for excitatory neurons
        - 'zscore_inh' : np.ndarray, data for inhibitory neurons

    Returns
    -------
    pd.DataFrame
        DataFrame containing, for each recording:
        - 'ID' : str, recording identifier
        - 'Genotype' : str
        - 'n_EXC' : int, number of excitatory neurons
        - 'n_INH' : int, number of inhibitory neurons
        - 'perc_EXC' : float, percentage of excitatory neurons
        - 'perc_INH' : float, percentage of inhibitory neurons

    The function also produces a 2x2 grid of boxplots comparing WT vs KO across neuron type (EXC, INH)
    and metric type (absolute number and percentage).
    """
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
                        colors=[sty.wt_color, sty.all_ko_color], det_marker=True, force_markers_identity=False)
    fig.suptitle("Comparison between WT and Fmr1KO of the number and percentage of neurons in the field of view", fontsize=10)
    fig.canvas.manager.set_window_title("Nb neurons FOV")
    plt.show()
    return data


def plot_neuron_perc_amp(recs, pattern="recruited", detected_trials=True, undetected_trials=True, nogo_norm=False, ylim=[],
                         transformation=None, normality=[False, False], homogeneity=[False, False], qq_show=True,
                         colors=[sty.ko_color, sty.hypo_color, sty.wt_color]):
    """
    Plot the percentage of responsive neurons as a function of stimulation amplitude.

    This function computes and visualizes the proportion of excitatory (EXC) and inhibitory (INH) neurons
    recruited across stimulation amplitudes. Neurons can be analyzed for detected, undetected, or both trial types.
    Percentages are computed relative to the total number of neurons in each recording, with optional normalization
    against no-go trials. Results are plotted as amplitude–response curves using `ppt.curveplot`.

    Parameters
    ----------
    recs : list
        List of recording objects containing neuronal activity and metadata.
    pattern : str, optional
        Type of neuronal response to analyze. Must be one of:
        - "recruited" (both activation and inhibition, default)
        - "activated" (only activation)
        - "inhibited" (only inhibition)
    detected_trials : bool, optional
        If True, includes detected trials in the analysis (default is True).
    undetected_trials : bool, optional
        If True, includes undetected trials in the analysis (default is True).
    nogo_norm : bool, optional
        If True, subtracts the number of recruited neurons in no-go trials for normalization (default is False).
    ylim : list, optional
        Y-axis limits for the plots. If empty, limits are determined automatically.
    transformation : str or None, optional
        Statistical transformation applied to the data (default is None).
    normality : list of bool, optional
        Whether to consider normality for each neuron type [EXC, INH] in statistical tests (default is [False, False]).
    homogeneity : list of bool, optional
        Whether to consider variance homogeneity for each neuron type [EXC, INH] (default is [False, False]).
    qq_show : bool, optional
        If True, displays QQ-plots for normality diagnostics (default is True).
    colors : list, optional
        List of colors for plotting different genotypes. Default is `[sty.ko_color, sty.hypo_color, sty.wt_color]`.

    Returns
    -------
    dict
        A dictionary containing per-neuron-type results:
        - "data_EXC"/"data_INH" : DataFrame with percentages per amplitude and genotype
        - "test_EXC"/"test_INH" : Results of the statistical test
        - "post_EXC"/"post_INH" : Post-hoc test results
    """
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
            if rec.genotype != "KO-Hypo" and len(rec.genotype.split("-")) > 1:    # Handling the case of DMSO-BMS analysis
                rec_id = f"{rec.filename}-{rec.genotype.split('-')[1]}"
            else:
                rec_id = rec.filename
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
                if nogo_norm:
                    # The no-go trials are used to normalized
                    trials_no_go = resp_mat[:, rec.stim_ampl_filter(stim_ampl=[0], include_no_go=True)]
                    trials_no_go[trials_no_go != 1] = 0
                    recruited_no_go = np.mean(np.count_nonzero(trials_no_go, axis=0))
                    recruited_det -= recruited_no_go
                    recruited_det = 0 if recruited_det < 0 else recruited_det
                perc_n_det = (recruited_det / total_n) * 100
                row = {"ID": rec_id, "Genotype": rec.genotype, "Amplitude": amp, f"perc_{n_type}": perc_n_det}
                rows.append(row)
        data = pd.DataFrame(rows)
        data_nan = data.fillna(0)
        test, post_hoc = ppt.curveplot(ax[i], data_nan, between="Genotype", within="Amplitude", variable=f"perc_{n_type}",
                                       title=f"Percentage of {pattern} {n_type} neurons", data_points=False,
                                       ylabel=None, xlabel=None, ylim=ylim, colors=colors,
                                       id_display=True, legend_display=False, qq_show=qq_show, transformation=transformation, consider_normality=normality[i],
                                       consider_homogeneity=homogeneity[i])
        results[f"data_{n_type}"] = data_nan
        results[f"test_{n_type}"] = test
        results[f"post_{n_type}"] = post_hoc
    title = f"ampcurv_{pattern}_{trials_name}_trials"
    fig.suptitle(f"Percentage of {pattern} neurons for {trials_name} trials\n[no-go normalization == {nogo_norm}]", fontsize=15)
    fig.canvas.manager.set_window_title(title)
    plt.show()
    return results


# Not used in the final paper
def plot_response_variance(features_df, variable="act_EXC_perc"):
    """
    Plot the variance of neuronal responses as a function of stimulation amplitude.

    For each recording, this function computes the variance of a specified feature
    (e.g., percentage of activated excitatory neurons) across trials, aligned by
    the difference between stimulation amplitude and the threshold of the recording.
    Amplitudes are sampled from -10 to +10 µm relative to threshold in steps of 2 µm.
    Variance values are also normalized by the maximum variance within the recording
    to yield relative variance. The function generates two plots: one for raw variance
    and one for relative variance, displaying both individual curves and genotype
    averages.

    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame containing extracted neuronal features with at least the columns:
        "ID", "Genotype", "threshold", "amplitude", and the specified `variable`.
    variable : str, optional
        Column name of the feature for which to compute variance
        (default is "act_EXC_perc").

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per recording, containing:
        - "ID" : Recording identifier
        - "Genotype" : Genotype label
        - "Threshold" : Stimulation threshold of the recording
        - "Variance" : List of variances per amplitude relative to threshold
        - "Relative_Variance" : List of normalized variances (divided by maximum variance)
    """
    gp_data = features_df.drop(columns=["bounded_x0", "behavior"]).groupby(["ID", "Genotype", "threshold", "amplitude"], as_index=False).std()
    rows = []
    for rec_id in gp_data["ID"].unique():
        rec_data = gp_data[gp_data["ID"] == rec_id]
        threshold = rec_data.threshold.values[0]
        var_list = []
        for diff_to_threshold in np.arange(-10, 11, 2):
            amp = threshold + diff_to_threshold
            if 0 < amp <=12:
                var_list.append(rec_data[rec_data.amplitude == amp][variable].values[0])
            else:
                var_list.append(np.nan)
        rows.append({"ID": rec_id, "Genotype": rec_data.Genotype.values[0], "Threshold": threshold, "Variance": var_list})
    var_data = pd.DataFrame(rows)
    var_data["Relative_Variance"] = var_data["Variance"].apply(lambda lst: np.array(lst, dtype=float)).apply(lambda arr: arr / np.nanmax(arr))
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), constrained_layout=True)
    for _, rec_row in var_data.iterrows():
        ax[0].scatter(np.arange(-10, 11, 2), rec_row["Variance"], color=sty.color_dict[rec_row["Genotype"]][0], lw=2, alpha=0.5, s=5)
        ax[0].plot(np.arange(-10, 11, 2), rec_row["Variance"], color=sty.color_dict[rec_row["Genotype"]][0], lw=2, alpha=0.5)
        ax[1].scatter(np.arange(-10, 11, 2), rec_row["Relative_Variance"], color=sty.color_dict[rec_row["Genotype"]][0], lw=2, alpha=0.5, s=5)
        ax[1].plot(np.arange(-10, 11, 2), rec_row["Relative_Variance"], color=sty.color_dict[rec_row["Genotype"]][0], lw=2, alpha=0.5)
    for ax_id in range(2):
        ax[ax_id].set_xlabel("Amplitude difference to threshold (in µm)", fontsize=10)
        ax[ax_id].set_xticks(np.arange(-10, 11, 2))
    ax[0].set_ylabel(f"Std in {variable}", fontsize=10)
    ax[1].set_ylabel(f"Relative Std in {variable}", fontsize=10)
    # Plotting the mean curves per genotype
    for geno in var_data.Genotype.unique():
        geno_subset = var_data[var_data["Genotype"] == geno]
        var_array = np.stack(geno_subset["Variance"].values)
        rel_var_array = np.stack(geno_subset["Relative_Variance"].values)
        mean_var = np.nanmean(var_array, axis=0)
        rel_mean_var = np.nanmean(rel_var_array, axis=0)
        ax[0].plot(np.arange(-10, 11, 2), mean_var, color=sty.color_dict[geno][0], lw=4, alpha=0.75)
        ax[1].plot(np.arange(-10, 11, 2), rel_mean_var, color=sty.color_dict[geno][0], lw=4, alpha=0.75)
    fig.suptitle(f"Variance in {variable} across amplitude of stimulation", fontsize=12)
    fig.canvas.manager.set_window_title(f"Var_{variable}_ampl")
    plt.show()
    return var_data


# Not used in the final paper
def get_perc_non_recruited_neurons_trials(rec, amplitude_filter=None, n_type="EXC"):
    """
    Compute the percentage of neurons that show no recruitment across trials.

    This function extracts the responsivity matrix for a given neuron type and
    computes the proportion of neurons that remain non-recruited (no activation
    or inhibition across all selected trials). The trials can be filtered either
    by a specific stimulation amplitude or by the session threshold.

    Parameters
    ----------
    rec : object
        Recording object containing neuronal data, with attributes:
        - `stim_ampl` : array of stimulation amplitudes
        - `session_threshold` : threshold amplitude for the session
        - `matrices` : dictionary containing responsivity matrices per neuron type.
    amplitude_filter : {"threshold", int, None}, optional
        Defines which trials to include:
        - `"threshold"` : select trials at the session threshold
        - `int` : select trials at a specific amplitude
        - `None` (default) : include all trials
    n_type : {"EXC", "INH"}, optional
        Neuron type to analyze (default is `"EXC"`).

    Returns
    -------
    float
        Percentage of neurons that are non-recruited across the selected trials.
    """
    resp = rec.matrices[n_type]["Responsivity"]
    # Filtering the trials of desired amplitude
    if amplitude_filter == "threshold":
        amp_filt = rec.stim_ampl == rec.session_threshold
    elif isinstance(amplitude_filter, int):
        amp_filt = rec.stim_ampl == amplitude_filter
    else:
        amp_filt = True * len(rec.stim_ampl)
    resp = resp[:, amp_filt]
    return (np.all(resp == 0, axis=1).sum() / resp.shape[0]) * 100




# endregion ============================================================================================================
# region ===================================== Neuronal clusters =======================================================

# Not used in the final paper
def get_concat_act(rec, n_type="EXC", zscore=True, pre_stim=False):
    """
    Concatenate neuronal activity across all trials for a given recording.

    This function extracts either z-scored or raw fluorescence traces from
    excitatory or inhibitory neurons, aligns them to the stimulation onset,
    and concatenates them trial by trial into a single array. Optionally,
    a fixed pre-stimulation period can be included.

    Parameters
    ----------
    rec : object
        Recording object containing neuronal activity, with attributes:
        - `zscore_exc` / `zscore_inh` : z-scored activity matrices
        - `df_f_exc` / `df_f_inh` : raw fluorescence activity matrices
        - `stim_time` : list or array of stimulation onset frames
        - `stim_durations` : list or array of stimulation durations (in frames).
    n_type : {"EXC", "INH"}, optional
        Neuron type to analyze (default is `"EXC"`).
    zscore : bool, optional
        Whether to use z-scored activity (`True`, default) or raw fluorescence (`False`).
    pre_stim : bool, optional
        If `True`, shifts the window 15 frames before the stimulation onset
        for each trial (default is `False`).

    Returns
    -------
    np.ndarray
        Concatenated activity array of shape `(n_neurons, total_frames)` for all trials.
    """
    if n_type == "EXC":
        activity = rec.zscore_exc if zscore else rec.df_f_exc
    elif n_type == "INH":
        activity = rec.zscore_inh if zscore else rec.df_f_inh

    start_times = np.array(rec.stim_time, dtype=int)
    if pre_stim:
        start_times -= 15
    durations = np.array(rec.stim_durations, dtype=int)
    frames_trials = []
    for time, duration in zip(start_times, durations):
        frames_trials.append(activity[:, time:time + duration])
    return np.concatenate(frames_trials, axis=1)


# Not used in the final paper
def pca_neurons(recs, n_type="EXC", min_trials=5, pre_stim=False):
    """
    Perform PCA on neuronal activity to explore clustering of neurons.

    This function extracts neuronal activity across trials, performs a principal
    component analysis (PCA), and visualizes neurons in a 3D PCA space. Neurons
    are colored according to their recruitment pattern: activated (1, red), inhibited
    (-1, blue), both activated and inhibited (2, purple), or non-recruited (0, gray).
    Recruitment is defined by having at least `min_trials` activations and/or inhibitions.
    PCA is performed with 3 components using `svd_solver="arpack"` and whitening enabled.

    Parameters
    ----------
    recs : list
        List of recording objects, each containing:
        - `matrices[n_type]["Responsivity"]` : responsivity matrix
        - `stim_time` : stimulation onset frames
        - `stim_durations` : stimulation durations
        - `zscore_exc` / `zscore_inh` or `df_f_exc` / `df_f_inh` : neuronal activity
        - `filename`, `genotype`, `threshold` : metadata for plotting and output.
    n_type : {"EXC", "INH"}, optional
        Neuron type to analyze (default is `"EXC"`).
    min_trials : int, optional
        Minimum number of trials required for a neuron to be considered recruited
        (default is 5).
    pre_stim : bool, optional
        If `True`, include 15 frames before stimulation onset when extracting activity
        (default is `False`).

    Returns
    -------
    pd.DataFrame
        Summary table with one row per recording, including:
        - `Genotype`
        - `ID` (filename)
        - `Threshold`
        - Variance explained by `PC1` and `PC2`.
    """
    pattern_color = {-1: "blue", 0: "gray", 1: "red", 2: "purple"}
    rows = []
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(24, 12), sharex=True, constrained_layout=True, subplot_kw={'projection': '3d'})
    ax = axs.flatten()
    for ax_id, rec in enumerate(recs):
        # Labelling the neurons according to their recruitment pattern
        act_counts = np.sum(rec.matrices[n_type]["Responsivity"] == 1, axis=1)
        inh_counts = np.sum(rec.matrices[n_type]["Responsivity"] == -1, axis=1)
        pattern_arr = np.where((act_counts >= min_trials) & (inh_counts >= min_trials), 2, np.where(inh_counts >= min_trials, -1, np.where(act_counts >= min_trials, 1, 0)))
        # Retrieving the neuronal activity during trials
        concat_act = get_concat_act(rec, n_type=n_type, zscore=False, pre_stim=pre_stim)
        # Performing a PCA
        pca = PCA(n_components=3, svd_solver='arpack', whiten=True)
        X_pca = pca.fit_transform(concat_act)
        explained_var = pca.explained_variance_ratio_
        # Storing the values in a DataFrame and plotting them
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
        pca_df["Pattern"] = pattern_arr
        for pattern_id, pattern_label in enumerate(sorted(pca_df["Pattern"].unique(), reverse=True)):
            subset = pca_df[pca_df["Pattern"] == pattern_label]
            ax[ax_id].scatter(subset["PC1"], subset["PC2"], subset["PC3"], c=pattern_color[pattern_label], label=str(pattern_label),
                              alpha=0.7, s=5)
        ax[ax_id].set_xlabel(f"PC1 ({explained_var[0]:.1%})", fontsize=10)
        ax[ax_id].set_ylabel(f"PC2 ({explained_var[1]:.1%})", fontsize=10)
        ax[ax_id].set_zlabel(f"PC3 ({explained_var[2]:.1%})", fontsize=10)
        ax[ax_id].tick_params(axis='both', labelsize=10)
        ax[ax_id].set_title(f"{rec.filename} ({rec.genotype})", color=sty.color_dict[rec.genotype][0], fontsize=10)
        rows.append({"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.threshold,
                     "PC1": explained_var[0], "PC2": explained_var[1]})
    for extra_ax in ax[len(recs):]:
        extra_ax.set_axis_off()
    fig.suptitle("PCA of neuronal activity during trials")
    fig.canvas.manager.set_window_title("PCA neuron act")
    plt.show()
    return pd.DataFrame(rows)


# Not used in the final paper
def compare_pca_trial_vs_concat(recs, n_type="EXC"):
    """
    Compare PCA projections of neuronal activity between individual trials and concatenated trials.

    For each recording, this function separates threshold-level trials into hit and miss groups,
    extracts neuronal activity, and performs a PCA on each trial individually as well as on the
    concatenated trials. Neurons are plotted in a 2D PCA space, with their response pattern encoded
    by different markers: activated (1, "^"), inhibited (-1, "s"), or non-responsive (0, "o"). Colors
    are assigned uniquely to neurons to facilitate tracking across trials. This visualization highlights
    whether trial-by-trial PCA embeddings differ from the global structure observed when concatenating
    all trials of the same behavioral outcome.

    Parameters
    ----------
    recs : list
        List of recording objects, each containing:
        - `zscore_exc` / `zscore_inh` : activity matrices.
        - `matrices[n_type]["Responsivity"]` : responsivity matrix per trial.
        - `stim_ampl` : stimulation amplitudes.
        - `stim_time` : stimulation onset frames.
        - `stim_durations` : duration of each stimulation.
        - `session_threshold` : threshold amplitude for behavior.
        - `detected_stim` : trial-by-trial detection outcome.
        - `filename`, `genotype` : metadata for labeling plots.
    n_type : {"EXC", "INH"}, optional
        Neuron type to analyze (default is `"EXC"`).

    Returns
    -------
    None
        The function generates PCA plots for each recording and displays them.
    """
    cmap = plt.get_cmap("rainbow")
    marker_map = {-1: "s", 0: "o", 1: "^"}
    for rec in recs:
        activity = rec.zscore_exc if n_type == "EXC" else rec.zscore_inh
        resp_mat = rec.matrices[n_type]["Responsivity"]
        # Getting the activity for the threshold trials, splitting hit and miss
        hit_mask = (rec.stim_ampl == rec.session_threshold) & (rec.detected_stim == True)
        miss_mask = (rec.stim_ampl == rec.session_threshold) & (rec.detected_stim == False)
        hit_times = rec.stim_time[hit_mask]
        miss_times = rec.stim_time[miss_mask]
        hit_durations = rec.stim_durations[hit_mask]
        miss_durations = rec.stim_durations[miss_mask]
        hit_resp = resp_mat[:, hit_mask]
        miss_resp = resp_mat[:, miss_mask]
        fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(24, 16), constrained_layout=True)
        ax = axs.flatten()
        counts = 0
        neuron_colors = cmap(np.linspace(0, 1, activity.shape[0]))
        for behavior, times, durations, resp in zip(["Hit", "Miss"], [hit_times, miss_times], [hit_durations, miss_durations], [hit_resp, miss_resp]):
            type_activity = np.empty((activity.shape[0], 0))
            for trial_id in range(len(times)):
                start = int(times[trial_id])
                end = int(start + durations[trial_id])
                trial_activity = activity[:, start:end]
                trial_resp = resp[:, trial_id]
                # Concatenating the trial activity
                type_activity = np.hstack((type_activity, trial_activity))
                # Performing a PCA
                pca = PCA(n_components=2, svd_solver="auto", whiten=False)
                X_pca = pca.fit_transform(trial_activity)
                explained_var = pca.explained_variance_ratio_
                # Plotting the different neurons, with a specific marker depending on the response pattern
                for resp_val, marker in marker_map.items():
                    mask = (trial_resp == resp_val)
                    if np.any(mask):
                        ax[counts + trial_id].scatter(X_pca[mask, 0], X_pca[mask, 1], c=neuron_colors[mask], marker=marker, edgecolor="dimgrey", s=20, alpha=0.75)
                # ax[counts + trial_id].scatter(X_pca[:, 0], X_pca[:, 1], color=neuron_colors, alpha=0.7, s=5)
                ax[counts + trial_id].set_xlabel(f"PC1 ({explained_var[0]:.1%})", fontsize=10)
                ax[counts + trial_id].set_ylabel(f"PC2 ({explained_var[1]:.1%})", fontsize=10)
                ax[counts + trial_id].tick_params(axis='both', labelsize=10)
                ax[counts + trial_id].set_title(f"{behavior} n°{trial_id}", color=sty.color_dict[rec.genotype][0 if behavior == "Hit" else 1], fontsize=10)
            # PCA on the concatenated trials
            pca = PCA(n_components=2, svd_solver="auto", whiten=False)
            X_pca_concat = pca.fit_transform(type_activity)
            explained_var = pca.explained_variance_ratio_
            # Storing the values in a DataFrame and plotting them
            ax[counts + trial_id + 1].scatter(X_pca_concat[:, 0], X_pca_concat[:, 1], color=neuron_colors, alpha=0.7, s=20)
            ax[counts + trial_id + 1].set_xlabel(f"PC1 ({explained_var[0]:.1%})", fontsize=10)
            ax[counts + trial_id + 1].set_ylabel(f"PC2 ({explained_var[1]:.1%})", fontsize=10)
            ax[counts + trial_id + 1].tick_params(axis='both', labelsize=10)
            ax[counts + trial_id + 1].set_title(f"{behavior} (concatenated)", color=sty.color_dict[rec.genotype][0 if behavior == "Hit" else 1], fontsize=10)
            counts += trial_id + 2
        # Setting the unused axes off
        for ax_id in ax[counts:]:
            ax_id.set_axis_off()
        fig.suptitle(f"PCA of neuronal activity during threshold trials for {rec.filename} ({rec.genotype})", fontsize=12)
        fig.canvas.manager.set_window_title(f"PCA_{rec.filename}")
        plt.show()


# Not used in the final paper
def hit_tuned_neurons(recs, normalize=True):
    """
    Compare the proportion of neurons significantly tuned to hit detection across genotypes.

    For each recording, this function computes the fraction of excitatory and inhibitory neurons
    that are reliably activated or inhibited during hit trials. If `normalize=True`, the number
    of tuned neurons is expressed relative to the pool of neurons that were responsive during
    detected trials, which accounts for differences in overall responsivity. If `normalize=False`,
    tuning proportions are calculated relative to the total number of neurons of each type,
    which highlights global group-level differences but ignores the variability in the number
    of recruited neurons across recordings. The results are summarized in a DataFrame and
    illustrated with boxplots comparing WT, KO-Hypo, and KO genotypes.

    Parameters
    ----------
    recs : dict
        Dictionary of recording objects, each containing:
        - `matrices["EXC"/"INH"]["Responsivity"]` : responsivity matrix per trial.
        - `detected_stim` : Boolean mask for detected trials.
        - `hit_tuned_exc` / `hit_tuned_inh` : tuning classification per neuron
          (-1 = inhibited, 0 = non-tuned, 1 = activated).
        - `zscore_exc` / `zscore_inh` : activity matrices (to get neuron counts).
        - `genotype`, `filename` : metadata for grouping and labeling.
    normalize : bool, optional
        Whether to normalize the counts of hit-tuned neurons by the number of responsive
        neurons in detected trials (`True`) or by the total number of neurons of each type (`False`).
        Default is `True`.

    Returns
    -------
    pd.DataFrame
        Summary of tuning proportions per recording, including:
        - `"exc_activated"`, `"exc_inhibited"`, `"inh_activated"`, `"inh_inhibited"`
          : normalized proportions.
        - `"_perc"` versions of the above, computed relative to the total neuron count.
        - `"Genotype"`, `"ID"` : recording metadata.

    """
    rows = []
    for rec in recs.values():
        if normalize:
            nb_resp_exc = ((rec.matrices["EXC"]["Responsivity"][:, rec.detected_stim] != 0).any(axis=1)).sum()
            nb_resp_inh = ((rec.matrices["INH"]["Responsivity"][:, rec.detected_stim] != 0).any(axis=1)).sum()
        else:
            nb_resp_exc = 1
            nb_resp_inh = 1
        exc_activated_rec = rec.hit_tuned_exc.tolist().count(1)/nb_resp_exc
        exc_inhibited_rec = rec.hit_tuned_exc.tolist().count(-1)/nb_resp_exc
        inh_activated_rec = rec.hit_tuned_inh.tolist().count(1)/nb_resp_inh
        inh_inhibited_rec = rec.hit_tuned_inh.tolist().count(-1)/nb_resp_inh
        nb_exc = rec.zscore_exc.shape[0]
        nb_inh = rec.zscore_inh.shape[0]
        exc_activated_tot = rec.hit_tuned_exc.tolist().count(1)/nb_exc
        exc_inhibited_tot = rec.hit_tuned_exc.tolist().count(-1)/nb_exc
        inh_activated_tot = rec.hit_tuned_inh.tolist().count(1)/nb_inh
        inh_inhibited_tot = rec.hit_tuned_inh.tolist().count(-1)/nb_inh
        print(f"EXC: {rec.hit_tuned_exc.tolist().count(1)} act {rec.hit_tuned_exc.tolist().count(-1)} inh / {nb_resp_exc} / {nb_exc}")
        print(f"INH: {rec.hit_tuned_inh.tolist().count(1)} act {rec.hit_tuned_inh.tolist().count(-1)} inh / {nb_resp_inh} / {nb_inh}")
        rows.append({"Genotype": rec.genotype, "ID": rec.filename,
                     "exc_activated": exc_activated_rec, "exc_inhibited": exc_inhibited_rec,
                     "inh_activated": inh_activated_rec, "inh_inhibited": inh_inhibited_rec,
                     "exc_activated_perc": exc_activated_tot, "exc_inhibited_perc": exc_inhibited_tot,
                     "inh_activated_perc": inh_activated_tot, "inh_inhibited_perc": inh_inhibited_tot})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(24, 32), constrained_layout=True)
    for col_id, cluster in enumerate(["exc_activated", "exc_inhibited", "inh_activated", "inh_inhibited"]):
        wt = data[data["Genotype"] == "WT"][cluster].values
        hypo = data[data["Genotype"] == "KO-Hypo"][cluster].values
        ko = data[data["Genotype"] == "KO"][cluster].values
        wt_perc = data[data["Genotype"] == "WT"][f"{cluster}_perc"].values
        hypo_perc = data[data["Genotype"] == "KO-Hypo"][f"{cluster}_perc"].values
        ko_perc = data[data["Genotype"] == "KO"][f"{cluster}_perc"].values

        ppt.boxplot(ax[0, col_id], wt, hypo, paired=False, ylabel=f"n {cluster}", title=f"WT/KO-Hypo",
                    ylim=[], colors=[sty.wt_color, sty.hypo_color])
        ppt.boxplot(ax[1, col_id], wt_perc, hypo_perc, paired=False, ylabel=f"% {cluster}", title=f"WT/KO-Hypo",
                    ylim=[], colors=[sty.wt_color, sty.hypo_color])
        ppt.boxplot(ax[2, col_id], wt, ko, paired=False, ylabel=f"n {cluster}", title=f"WT/KO",
                    ylim=[], colors=[sty.wt_color, sty.ko_color])
        ppt.boxplot(ax[3, col_id], wt_perc, ko_perc, paired=False, ylabel=f"% {cluster}", title=f"WT/KO",
                    ylim=[], colors=[sty.wt_color, sty.ko_color])
    fig.suptitle(f"Comparison between genotypes of the number of Hit tuned neurons\n[Normalization by recruited = {normalize}]", fontsize=12)
    fig.canvas.manager.set_window_title(f"Hit tuned neurons_norm={normalize}")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/Hit_tuned_neurons_{normalize}.pdf")
    plt.show()
    return data


# Not used in the final paper
def plot_hit_amp_tuned(recs):
    """
    Visualize and summarize hit-tuned and amplitude-tuned neurons for each recording.

    For each mouse, this function plots side-by-side heatmaps showing the tuning
    classification of excitatory (EXC) and inhibitory (INH) neurons with respect to
    hit detection and amplitude discrimination. Neurons are color-coded according
    to their tuning category (-1 = inhibited, 0 = non-tuned, 1 = activated). The
    function also returns a DataFrame summarizing the raw tuning arrays for each
    recording.

    Parameters
    ----------
    recs : list
        List of recording objects, each containing:
        - `hit_tuned_exc`, `amp_tuned_exc` : arrays of tuning classification for EXC neurons.
        - `hit_tuned_inh`, `amp_tuned_inh` : arrays of tuning classification for INH neurons.
        - `genotype`, `filename` : metadata for labeling and grouping.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per recording, containing:
        - `"Hit tuned EXC"`, `"Amp tuned EXC"`,
        - `"Hit tuned INH"`, `"Amp tuned INH"`,
        along with `"Genotype"` and `"ID"`.
    """
    rows = []
    fig, ax = plt.subplots(nrows=2, ncols=22, figsize=(22, 12), gridspec_kw={'height_ratios': [3, 1]},
                           constrained_layout=True)
    for col, rec in enumerate(recs):
        rows.append({"Genotype": rec.genotype, "ID": rec.filename,
                     "Hit tuned EXC": rec.hit_tuned_exc, "Amp tuned EXC": rec.amp_tuned_exc,
                     "Hit tuned INH": rec.hit_tuned_inh, "Amp tuned INH": rec.amp_tuned_inh})
        im_exc = ax[0, col].imshow(np.vstack([rec.hit_tuned_exc, rec.amp_tuned_exc]).T, cmap="inferno", aspect='auto',
                                   interpolation='nearest', vmin=-1, vmax=1)
        ax[0, col].set_title(f"{rec.filename}\nEXC", fontsize=10, fontweight="bold", color=sty.color_dict[rec.genotype][0])
        ax[0, col].set_yticks([])
        ax[0, col].set_xticks([0, 1])
        ax[0, col].set_xticklabels(["Hit", "Amp"], fontsize=8, rotation=90)
        im_inh = ax[1, col].imshow(np.vstack([rec.hit_tuned_inh, rec.amp_tuned_inh]).T, cmap="inferno", aspect='auto',
                                   interpolation='nearest', vmin=-1, vmax=1)
        ax[1, col].set_title(f"{rec.filename}\nINH", fontsize=10, fontweight="bold", color=sty.color_dict[rec.genotype][0])
        ax[1, col].set_yticks([])
        ax[1, col].set_xticks([0, 1])
        ax[1, col].set_xticklabels(["Hit", "Amp"], fontsize=8, rotation=90)
    tuned_df = pd.DataFrame(rows)
    cbar = fig.colorbar(im_exc, ax=ax[1, 21], ticks=[-1, 0, 1], orientation='vertical')
    fig.suptitle("Hit & amplitude tuned neurons", fontsize=12)
    plt.show()
    return tuned_df


# Not used in the final paper
def neurons_hit_consistency(recs):
    """
    Evaluate the consistency of hit-tuned neurons across detected trials.

    For each recording and neuron type (EXC, INH), the function calculates how
    consistently a neuron responds in the same direction (activated or inhibited)
    across all detected stimuli. Neurons labeled as hit-tuned are categorized as
    "ON" (activated) or "OFF" (inhibited), and their consistency is expressed as
    the proportion of trials in which their responsivity matches their tuning
    label (ON = +1, OFF = -1). Only neurons labeled as hit-tuned are included in
    the analysis. Results are aggregated per recording and genotype, and a figure
    (2×2 grid of boxplots) is produced to compare WT and KO-Hypo groups for each
    neuron type and tuning category.

    Parameters
    ----------
    recs : dict
        Dictionary of recording objects, where each object must provide:
        - `matrices[n_type]["Responsivity"]` : trial-by-trial responsivity matrix
          (neurons × trials) with values {-1, 0, 1}.
        - `detected_stim` : indices of detected trials.
        - `hit_tuned_exc`, `hit_tuned_inh` : arrays labeling neurons as
          activated (1), inhibited (-1), or non-tuned (0).
        - `genotype`, `filename` : metadata for grouping and labeling.

    Returns
    -------
    pd.DataFrame
        Aggregated consistency values with columns:
        - `"Genotype"`, `"ID"`, `"Type"` (EXC or INH),
        - `"Label"` (ON or OFF),
        - `"Consistency"` (mean proportion across neurons).
    """
    rows = []
    for rec in recs.values():
        for n_type in ["EXC", "INH"]:
            # resp_mat = rec.matrices[n_type]["Responsivity"][:, rec.detected_stim]
            resp_mat = rec.matrices[n_type]["Responsivity"][:, rec.detected_stim]
            label_list = rec.hit_tuned_exc if n_type == "EXC" else rec.hit_tuned_inh
            for n_id, label in enumerate(label_list):
                if label == 1:
                    prop = np.count_nonzero(resp_mat[n_id] == 1) / resp_mat.shape[1]
                    rows.append({"Genotype": rec.genotype, "ID": rec.filename, "Type": n_type, "Neuron": n_id,
                                 "Label": "ON", "Consistency": prop})
                elif label == -1:
                    prop = np.count_nonzero(resp_mat[n_id] == -1) / resp_mat.shape[1]
                    rows.append({"Genotype": rec.genotype, "ID": rec.filename, "Type": n_type, "Neuron": n_id,
                                 "Label": "OFF", "Consistency": prop})
                else:
                    continue
    data = pd.DataFrame(rows)
    # data["Label"] = data["Label"].map({-1: "inhibited", 0: "none", 1: "activated"}).astype("category")
    gp_data = data.groupby(["Genotype", "ID", "Type", "Label"], as_index=False).mean().drop(columns=["Neuron"])
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 16), constrained_layout=True)
    for row_id, n_type in enumerate(["EXC", "INH"]):
        for col_id, label in enumerate(["OFF", "ON"]):
            ppt.boxplot(ax[row_id, col_id], gp_data[(gp_data["Type"] == n_type) & (gp_data["Label"] == label) & (gp_data["Genotype"] == "WT")]["Consistency"],
                        gp_data[(gp_data["Type"] == n_type) & (gp_data["Label"] == label) & (gp_data["Genotype"] == "KO-Hypo")]["Consistency"],
                        paired=False, ylabel=f"Consistency", title=f"{n_type} - {label}", ylim=[], colors=[sty.wt_color, sty.hypo_color])
    fig.suptitle(f"Comparison of the consistency of hit tuned neuron response between WT and KO-Hypo", fontsize=12)
    fig.canvas.manager.set_window_title("Hit tuned consistency")
    plt.show()
    return gp_data



# endregion ============================================================================================================

if __name__ == '__main__':
    BMS_analysis = False
    ### Initialisation of recs instances ###
    if BMS_analysis:
        directory = "C:/Users/cvandromme/Desktop/Tactile_detection/Data_DMSO_BMS/"
        roi_path = "C:/Users/cvandromme/Desktop/Tactile_detection/Fmko_bms&dmso_info.xlsx"
    else:
        directory = "C:/Users/cvandromme/Desktop/Tactile_detection/Data/"
        roi_path = "C:/Users/cvandromme/Desktop/Tactile_detection/FmKO_ROIs&inhibitory.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/"
    roi_info = pd.read_excel(roi_path)
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]
    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path, cache=True, correction=False)
        return rec
    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    if BMS_analysis:
        recs = {f"{ar.get().filename}-{ar.get().genotype.split("-")[1]}": ar.get() for ar in async_results}
    else:
        recs = {ar.get().filename: ar.get() for ar in async_results}

    # ====== Building a summary table ======
    summary = []
    for rec in recs.values():
        summary.append({"Genotype": rec.genotype, "ID": rec.filename, "x0_psy": rec.x0_psy, "Session Threshold": rec.session_threshold, "Global Threshold": rec.threshold,
                        "n_trials": len(rec.detected_stim), "n_EXC": len(rec.zscore_exc), "n_INH": len(rec.zscore_inh)})
    summary = pd.DataFrame(summary)

    save_fig = False

    # ====== Response features ======
    for rec in recs.values():
        rec.peak_delay_amp()
        # rec.auc()
        # rec.hit_tuned()
        # rec.amp_tuned()
    full_data = get_features(recs.values(), amp_delay=True, auc=False)
        # data = full_data[~full_data["ID"].isin([5893, 7539, 7554])] # Trying to exclude KO but not enough mice
        # data = full_data[~full_data["ID"].isin([6611])]
    #   --- Within ---
    # compare_sub_supra_within(data, behavior_filter=None, genotype="KO", comparison="all_supra")
    # for filter in [None, True, False]:
    #     for gen in ["KO", "KO-Hypo", "WT"]:
    #         for comp in ["sub", "all_sub", "supra", "all_supra"]:
    #             compare_sub_supra_within(data, behavior_filter=filter, genotype=gen, comparison=comp)
    #   --- Between ---
    # wt, hypo = compare_sub_supra_between(data, behavior_filter=None, gp1="WT-DMSO", gp2="WT-BMS", gp1_amps="real_mean_genotype", gp2_amps="gp1_threshold", colors=[sty.wt_color, sty.wt_bms_color])
    wt, hypo = compare_sub_supra_between(full_data, behavior_filter=True, gp1="WT-BMS", gp2="KO-BMS", gp1_amps="all", gp2_amps="all", colors=[sty.wt_color, sty.hypo_color])
    #   --- Between (Deltas) ---
    # sub_supra_delta_df = compare_sub_supra_deltas(data, behavior_filter=None, gp1="WT", gp2="KO-Hypo")
    # sub_supra_delta_df_wt, sub_supra_delta_df_hypo = compare_sub_supra_deltas(data, behavior_filter=None, gp1="WT",
    #                                                                           gp2="KO-Hypo", delta="sub")
    btw_wt, btw_hypo = compare_det_undet_between(full_data, gp1="WT", gp2="KO-Hypo", amplitude="all", behavior="miss")

    # --- Hit vs. Miss ---
    det, undet = compare_det_undet(full_data, genotype="KO-Hypo", amplitude="all") # /!\ full_data for all amp and data for threshold analysis

    # mean_det = np.mean(det.drop(columns="Genotype"), axis=0)
    # mean_undet = np.mean(undet.drop(columns="Genotype"), axis=0)

    results = plot_neuron_perc_amp(recs.values(), pattern="recruited", detected_trials=True, undetected_trials=True,
                                   nogo_norm=False, ylim=[0, 60], transformation="yeojohnson", normality=[False, True],
                                   homogeneity=[False, False], colors=[sty.hypo_bms_color, sty.hypo_color, sty.wt_bms_color, sty.wt_color])
    results = plot_neuron_perc_amp(recs.values(), pattern="recruited", detected_trials=True, undetected_trials=False,
                                   nogo_norm=False, ylim=[0, 60], transformation="yeojohnson", normality=[False, True],
                                   homogeneity=[False, True], colors=[sty.ko_color, sty.hypo_color, sty.wt_color])

    nogo_df = nogo_fa_cr(recs.values(), condition=None)
    delta_df = delta_hit_miss_comp(full_data, threshold_only=False, wt_threshold=False, condition="BMS")
    # delta_df = delta_hit_miss_comp(data, threshold_only=False, wt_threshold=False, condition="BMS") #/!\ full_data for all amp and data for threshold analysis


    # ====== Responsivity ======
    # recs_without_ko = {k: v for k, v in recs.items() if v.genotype != "KO"}
    # neurons = nb_neurons(recs.values())
    #
    # results = plot_neuron_perc_amp(recs.values(), pattern="activated", detected_trials=True, undetected_trials=True, ylim=[0, 30],
    #                                transformation="yeojohnson", normality=[False, False], homogeneity=[False, False], qq_show=False,
    #                                colors=[sty.ko_color, sty.hypo_color, sty.wt_color])
    # To save the results from ampcurv:
    # test_exc = results['test_EXC']
    # post_exc = results["post_EXC"]
    # post_exc_wt = post_exc["WT"]
    # post_exc_hypo = post_exc["KO-Hypo"]
    # post_exc_ko = post_exc["KO"]
    # post_exc_btw = post_exc["between"]
    # data_exc = results["data_EXC"]
    # data_inh = results["data_INH"]
    # test_inh = results['test_INH']
    # post_inh = results["post_INH"]
    # post_inh_btw = post_inh["between"]
    # post_inh_wt = post_inh["WT"]
    # post_inh_hypo = post_inh["KO-Hypo"]
    # post_inh_ko = post_inh["KO"]

    # concat_act = get_concat_act(recs[4445], n_type="EXC", zscore=True)
    # neuron_pca_df = pca_neurons(recs.values(), pre_stim=False, min_trials=5)

    # rows = []
    # for rec in recs.values():
    #     rows.append({"Genotype": rec.genotype, "ID": rec.filename, "EXC_classif": rec.hit_tuned_exc, "INH_classif": rec.hit_tuned_inh,})
    # result = pd.DataFrame(rows)
    #
    # hit_tuned_df = hit_tuned_neurons(recs, normalize=True)
    # consistency_df = neurons_hit_consistency(recs)
    # fig, ax = plt.subplots(figsize=(8, 2), constrained_layout=True)
    # im = ax.imshow(np.vstack([recs[7553].hit_tuned_exc, recs[7553].amp_tuned_exc]), cmap="inferno", aspect='auto', interpolation='nearest')
    # cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], orientation='horizontal')
    # plt.show()

    # tuned_df = plot_hit_amp_tuned(recs.values())
    # var_df = plot_response_variance(full_data, variable="act_EXC_perc")

    # rows = []
    # for rec in recs.values():
    #     perc_non_recr = get_perc_non_recruited_neurons_trials(rec, amplitude_filter="threshold", n_type="EXC")
    #     rows.append({"ID": rec.filename, "Genotype": rec.genotype, "Threshold": rec.session_threshold, "Perc_non_recr": perc_non_recr})
    # non_recr_df = pd.DataFrame(rows)

    # rows = []
    # for rec in recs.values():
    #     hit = ((rec.stim_ampl == rec.session_threshold) & (rec.detected_stim == True)).sum()
    #     miss = ((rec.stim_ampl == rec.session_threshold) & (rec.detected_stim == False)).sum()
    #     rows.append({"ID": rec.filename, "Genotype": rec.genotype, "Threshold": rec.session_threshold, "Hit": hit, "Miss": miss})
    # nb_threshold_trials_df = pd.DataFrame(rows)
    #
    # compare_pca_trial_vs_concat([recs[4445]], n_type="EXC")