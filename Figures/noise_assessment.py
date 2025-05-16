# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool

from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import percephone.core.recording as pc
import percephone.plts.stats as ppt
from Figures.stimulus_encoding import get_features
from percephone.plts.utils import stat_boxplot


# endregion
# region ======================================== TBT variability ======================================================

def get_mean_trial_activity_df(recs, zscore=True):
    prestim_frames = 15
    rows = []
    for rec in recs:
        exc_activity = rec.zscore_exc if zscore else rec.df_f_exc
        inh_activity = rec.zscore_inh if zscore else rec.df_f_inh
        behavior_vector = rec.detected_stim
        amplitude_vector = rec.stim_ampl
        stim_time_vector = rec.stim_time
        stim_duration_vector = rec.stim_durations
        for neuron_type, activity in zip(["EXC", "INH"], [exc_activity, inh_activity]):
            resp_mat = rec.matrices[neuron_type]["Responsivity"]
            for n_id, neuron_activity in enumerate(activity):
                for trial_id in range(len(behavior_vector)):
                    stim_start = stim_time_vector[trial_id]
                    stim_duration = int(stim_duration_vector[trial_id])
                    trial_activity = neuron_activity[stim_start: stim_start + stim_duration]
                    trial_mean_activity = np.mean(trial_activity)
                    trial_std_activity = np.std(trial_activity)
                    prestim_activity = neuron_activity[stim_start - prestim_frames: stim_start]
                    prestim_mean_activity = np.mean(prestim_activity)
                    prestim_std_activity = np.std(prestim_activity)
                    row = {"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.session_threshold,
                           "Trial": trial_id, "Amplitude": amplitude_vector[trial_id],
                           "Behavior": behavior_vector[trial_id], "Neuron": f"{neuron_type}_{n_id}", "Resp": resp_mat[n_id, trial_id],
                           "Stim_mean": trial_mean_activity, "Stim_std": trial_std_activity,
                           "Prestim_mean": prestim_mean_activity, "Prestim_std": prestim_std_activity}
                    rows.append(row)
    activity_df = pd.DataFrame(rows)
    activity_df["Diff_mean"] = activity_df["Stim_mean"] - activity_df["Prestim_mean"]
    activity_df["Abs_Diff_mean"] = abs(activity_df["Stim_mean"]) - abs(activity_df["Prestim_mean"])
    activity_df["Diff_std"] = activity_df["Stim_std"] - activity_df["Prestim_std"]
    activity_df["Ratio_mean"] = activity_df["Stim_mean"] / activity_df["Prestim_mean"]
    activity_df["Ratio_std"] = activity_df["Stim_std"] / activity_df["Prestim_std"]
    activity_df["Abs_Ratio_std"] = abs(activity_df["Stim_std"]) / abs(activity_df["Prestim_std"])
    return activity_df


def pca(mean_activity_df):
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(20, 12), constrained_layout=True)
    ax = axs.flatten()
    rows = []
    for ax_id, rec_id in enumerate(mean_activity_df["ID"].unique()):
        rec_data = mean_activity_df[mean_activity_df["ID"] == rec_id].copy()
        data = rec_data.pivot(index=["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Behavior"], columns="Neuron",
                              values="Stim_mean").reset_index()
        genotype = data["Genotype"].values[0]
        mouse_id = data["ID"].values[0]
        threshold = data["Threshold"].values[0]
        y = data["Behavior"]
        X = data.drop(columns=["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Behavior"])
        # Performing a PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        # Storing the values in a DataFrame and plotting them
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Behavior"] = y.values
        for behavior_id, behavior_label in enumerate(sorted(pca_df["Behavior"].unique(), reverse=True)):
            subset = pca_df[pca_df["Behavior"] == behavior_label]
            ax[ax_id].scatter(subset["PC1"], subset["PC2"], c=color_dict[genotype][behavior_id], label=behavior_label,
                              alpha=0.7, s=5)
        ax[ax_id].set_xlabel(f"PC1 ({explained_var[0]:.1%})", fontsize=10)
        ax[ax_id].set_ylabel(f"PC2 ({explained_var[1]:.1%})", fontsize=10)
        ax[ax_id].tick_params(axis='both', labelsize=10)
        ax[ax_id].set_title(f"{mouse_id} ({genotype})", color=color_dict[genotype][0], fontsize=10)
        rows.append({"Genotype": genotype, "ID": mouse_id, "Threshold": threshold,
                     "PC1": explained_var[0], "PC2": explained_var[1]})
    fig.suptitle("PCA of zean Z-score Across Trials", fontsize=20)
    fig.canvas.manager.set_window_title("PCA")
    plt.show()
    return pd.DataFrame(rows)


def compare_tbt_var_per_amp(recruitment_df):
    """
    For each recording, compute the standard deviation
    Parameters
    ----------
    df

    Returns
    -------

    """
    grouped = recruitment_df.groupby(["Genotype", "ID", "threshold", "behavior", "amplitude"], as_index=False).std()
    hit = grouped[grouped["behavior"] == True]
    miss = grouped[grouped["behavior"] == False]
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), constrained_layout=True)
    ppt.curveplot(ax[0], hit[hit["Genotype"] != "KO"], between="Genotype", within="amplitude", variable="act_EXC_perc",
                  title="Variation of std of the % of activated EXC neurons across amplitudes for hit trials",
                  ylabel=None, xlabel=None, ylim=None, colors=[ppt.hypo_color, ppt.wt_color],
                  id_display=True, legend_display=True,
                  qq_show=True, transformation=None, consider_normality=False, consider_homogeneity=False)
    ppt.curveplot(ax[1], miss[miss["Genotype"] != "KO"], between="Genotype", within="amplitude", variable="act_EXC_perc",
                  title="Variation of std of the % of activated EXC neurons across amplitudes for miss trials",
                  ylabel=None, xlabel=None, ylim=None, colors=[ppt.hypo_color, ppt.wt_color],
                  id_display=True, legend_display=True,
                  qq_show=True, transformation=None, consider_normality=False, consider_homogeneity=False)
    plt.show()
    return hit


# endregion ============================================================================================================
# region ======================================== Pre-stimulus =========================================================

def filter_stim_recruited_neurons(activity_df):
    exc_neurons_df = activity_df[activity_df["Neuron"].str.startswith("EXC")]
    active_neurons = exc_neurons_df[(exc_neurons_df["Behavior"] == True) & (exc_neurons_df["Resp"] == 1)]
    valid_pairs = active_neurons[["ID", "Neuron"]].drop_duplicates()
    filtered_df = activity_df.merge(valid_pairs, on=["ID", "Neuron"])
    return filtered_df

def prestim_comp(activity_df):
    """
    Compares the raw pre-stimulus activity between WT and KO groups, averaging all trials and then by behavioral label.
    Is the global pre-stimulus higher or more variable in KO ?
    Parameters
    ----------
    activity_df

    Returns
    -------

    """
    gp_mean = activity_df.drop(columns=["Trial", "Amplitude", "Behavior", "Neuron", "Resp"]).groupby(["Genotype", "ID"], as_index=False).mean()
    gp_split = activity_df.drop(columns=["Trial", "Amplitude", "Neuron", "Resp"]).groupby(["Genotype", "ID", "Behavior"], as_index=False).mean()
    gp_hit = gp_split[gp_split["Behavior"] == True]
    gp_miss = gp_split[gp_split["Behavior"] == False]
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    for col, (condition, data) in enumerate(zip(["Global", "Hit", "Miss"], [gp_mean, gp_hit, gp_miss])):
        wt = data[data["Genotype"] == "WT"]
        hypo = data[data["Genotype"] == "KO-Hypo"]
        ppt.boxplot(ax[0, col], wt["Prestim_mean"].values, hypo["Prestim_mean"].values, ylabel="Prestim_mean", paired=False,
                    title=condition, ylim=[],
                    colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[1, col], wt["Prestim_std"].values, hypo["Prestim_std"].values, ylabel="Prestim_std", paired=False,
                    title=condition, ylim=[],
                    colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
    fig.suptitle("Pre-stimulus mean and std comparison between genotypes", fontsize=12)
    fig.canvas.manager.set_window_title("Pre-stimulus comparison")
    plt.show()
    return gp_mean, gp_hit, gp_miss


def snr_comp(activity_df):
    """
    Compares the difference between the stimulus and its pre-stimulus. Is the SNR higher during hit compared to miss
    trials ?
    Parameters
    ----------
    activity_df

    Returns
    -------

    """
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    data = activity_df[activity_df["Amplitude"] != 0]
    gp_data = data.drop(columns=["Trial", "Amplitude", "Neuron", "Resp"]).groupby(["Genotype", "ID", "Behavior"], as_index=False).mean()
    hit_data = gp_data[gp_data["Behavior"] == True]
    miss_data = gp_data[gp_data["Behavior"] == False]
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    for col, genotype in enumerate(gp_data["Genotype"].unique()):
        hit = hit_data[hit_data["Genotype"] == genotype]
        miss = miss_data[miss_data["Genotype"] == genotype]
        ppt.boxplot(ax[0, col], hit["Diff_mean"].values, miss["Diff_mean"].values, ylabel="Mean stim - mean prestim",
                    paired=True, title=genotype, ylim=[], colors=color_dict[genotype], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[1, col], hit["Ratio_std"].values, miss["Ratio_std"].values, ylabel="Std stim / std prestim",
                    paired=True, title=genotype, ylim=[], colors=color_dict[genotype], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of the SNR between hit and miss trials", fontsize=12)
    fig.canvas.manager.set_window_title("SNR comparison")
    plt.show()
    return gp_data

def compare_to_wt_threshold(activity_df):
    """
    Compares the SNR, of KO-Hypo mice to the one of WT mice at an amplitude where we have a difference between stim and
    pre-stim in WT but not in Hypo (6). And then compares the signal (stim) and the noise (pre-stim). Why don't KO-Hypo
    mice detects lower amplitude (lower signal or higher noise ?)

    Parameters
    ----------
    activity_df

    Returns
    -------

    """
    data = activity_df[activity_df["Amplitude"] == 4]
    gp_data = data.drop(columns=["Trial", "Amplitude", "Neuron", "Resp", "Behavior"]).groupby(["Genotype", "ID"], as_index=False).mean()
    wt = gp_data[gp_data["Genotype"] == "WT"]
    hypo = gp_data[gp_data["Genotype"] == "KO-Hypo"]
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), constrained_layout=True)
    ax = axs.flatten()
    for col, param in enumerate(["Abs_Diff_mean", "Abs_Ratio_std", "Stim_mean", "Prestim_mean", "Stim_std", "Prestim_std"]):
        ppt.boxplot(ax[col], wt[param].values, hypo[param].values, ylabel="DFF",
                    paired=False, title=param, ylim=[], colors=[ppt.wt_color, ppt.hypo_color], det_marker=False,
                    force_markers_identity=False)
    ppt.boxplot(ax[6], wt["Prestim_mean"].values, wt["Stim_mean"].values, ylabel="DFF",
                paired=True, title="Pre-stim/stim WT", ylim=[], colors=[ppt.wt_color, ppt.wt_color], det_marker=False,
                force_markers_identity=False)
    ppt.boxplot(ax[7], hypo["Prestim_mean"].values, hypo["Stim_mean"].values, ylabel="DFF",
                paired=True, title="Pre-stim/stim KO-Hypo", ylim=[], colors=[ppt.hypo_color, ppt.hypo_color], det_marker=False,
                force_markers_identity=False)

    fig.suptitle("Comparison of signal, noise and SNR between genotypes at average WT threshold", fontsize=12)
    fig.canvas.manager.set_window_title("Signal, noise & SNR comparison")
    plt.show()
    return wt, hypo

def prestim_influence_neuron_activation(activity_df):
    """
    Comparison of the pre-stimulus activity of stimulus activated neurons between trials when they are activated or not
    activated. Comparison of the mean value of prestimulus across all trials where the neurons were activated to the mean
    value of pre-stimulus across all trials where the neuron was not activated.
    Parameters
    ----------
    activity_df

    Returns
    -------

    """
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    data = activity_df[activity_df["Amplitude"] != 0]
    gp_neuron = data.drop(columns=["Trial", "Amplitude", "Behavior", "Threshold"]).groupby(["Genotype", "ID", "Neuron", "Resp"], as_index=False).mean()
    gp_mouse = gp_neuron.drop(columns=["Neuron"]).groupby(["Genotype", "ID", "Resp"], as_index=False).mean()
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 24), constrained_layout=True)
    for col, genotype in enumerate(gp_mouse["Genotype"].unique()):
        act_trials = gp_mouse[(gp_mouse["Genotype"] == genotype) & (gp_mouse["Resp"] == 1)]
        non_trials = gp_mouse[(gp_mouse["Genotype"] == genotype) & (gp_mouse["Resp"] == 0)]
        ppt.boxplot(ax[0, col], act_trials["Prestim_mean"].values, non_trials["Prestim_mean"].values, ylabel="Mean DFF",
                    paired=True, title="Prestim_mean (act/non_act)", ylim=[], colors=color_dict[genotype],
                    det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[1, col], act_trials["Stim_mean"].values, non_trials["Stim_mean"].values,
                    ylabel="Mean DFF",
                    paired=True, title="Stim_mean (act/non_act)", ylim=[], colors=color_dict[genotype],
                    det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[2, col], act_trials["Abs_Diff_mean"].values, non_trials["Abs_Diff_mean"].values, ylabel="Mean DFF",
                    paired=True, title="Abs_Diff_mean (act/non_act)", ylim=[], colors=color_dict[genotype],
                    det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of mean pre-stimulus activity of stimulus activated neurons between trials when they were activated or not", fontsize=12)
    fig.canvas.manager.set_window_title("Pre-stim comp (act vs non_act)")
    plt.savefig("Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/Pre-stim_comp_(act_vs_non_act).pdf")
    plt.show()
    return gp_mouse


def prestim_activated_neurons(activity_df):
    """
    Plot, for each neuron, the significance of the difference in prestim between the trials where the neuron is
    activated and the trials where the neuron is not responsive.

    Parameters
    ----------
    df

    Returns
    -------

    """
    neurons_sets = []
    rows = []
    for rec_id in activity_df["ID"].unique():
        rec_data = activity_df[(activity_df["ID"] == rec_id) & (activity_df["Amplitude"] == activity_df["Threshold"])].copy()
        # Keeping only the EXC neurons
        rec_data = rec_data[rec_data["Neuron"].str.startswith("EXC")]
        # Selecting the neurons that are activated at least once during detected trials
        act_hit_neurons = set(rec_data[(rec_data["Behavior"] == True) & (rec_data["Resp"] == 1)]["Neuron"].values)
        neurons_sets.append({"Genotype": rec_data["Genotype"].values[0], "ID": rec_id, "Kept neurons": act_hit_neurons})
        neuron_data = rec_data[rec_data["Neuron"].isin(act_hit_neurons)].copy()
        # Comparing the pre-stim of neurons when they are activated compared to when they are not
        for neuron_id in neuron_data["Neuron"].unique():
            activated = neuron_data[(neuron_data["Neuron"] == neuron_id) & (neuron_data["Resp"] == 1)].copy()
            non_activated = neuron_data[(neuron_data["Neuron"] == neuron_id) & (neuron_data["Resp"] != 1)].copy()
            if len(activated) > 2 and len(non_activated) > 2:
                raw_mean = stat_boxplot(activated["Prestim_mean"], non_activated["Prestim_mean"], "Prestim_mean", title="", paired=False, verbose=False)
                raw_std = stat_boxplot(activated["Prestim_std"], non_activated["Prestim_std"], "Prestim_std", title="", paired=False, verbose=False)
                diff_mean = stat_boxplot(activated["Diff_mean"], non_activated["Diff_mean"], "Diff_mean", title="", paired=False, verbose=False)
                diff_std = stat_boxplot(activated["Diff_std"], non_activated["Diff_std"], "Diff_std", title="", paired=False, verbose=False)
            else:
                raw_mean = np.nan
                raw_std = np.nan
                diff_mean = np.nan
                diff_std = np.nan
            rows.append({"Genotype": rec_data["Genotype"].values[0], "ID": rec_id, "Neuron": neuron_id,
                         "Raw_Mean": raw_mean, "Raw_Std": raw_std, "Diff_Mean": diff_mean, "Diff_Std": diff_std})
    results = pd.DataFrame(rows)
    # Plotting the results
    param = "Diff_Std"
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(20, 12), constrained_layout=True)
    ax = axs.flatten()
    for ax_id, rec_id in enumerate(results["ID"].unique()):
        data = results[results["ID"] == rec_id].copy()
        colors = ['green' if val <= 0.05 else ("orange" if val <= 0.1 else 'red') for val in data[param].values]
        ax[ax_id].bar(range(len(data)), data[param].values, color= colors, alpha=0.7, width=0.5)
        ax[ax_id].set_title(f"{rec_id} - {data["Genotype"].values[0]}", fontsize=12)
        ax[ax_id].axhline(y=0.05, color='gray', linestyle='--', lw=0.5)
        ax[ax_id].tick_params(axis='both', labelsize=10)
    fig.suptitle(f"Significance of the difference in pre-stim activity ({param}) for each neuron (activated vs. non activated)", fontsize=15)
    plt.show()
    return pd.DataFrame(rows)


def prestim_act_vector(activity_df, metric=None, hit_activated_only=False):
    """
    Compute the cosine similarity between each pair of trial, then see if there is a difference between within-condition
    similarity and cross-condition similarity. Working with threshold trials (and neurons that are activated at least
    once during detected trials).
    Parameters
    ----------
    activity_df

    Returns
    -------

    """
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    if metric is None:
        metric = activity_df.columns[-1]
    rows = []
    for rec_id in activity_df["ID"].unique():
        # rec_data = activity_df[(activity_df["ID"] == rec_id) & (activity_df["Amplitude"] == activity_df["Threshold"])].copy()
        rec_data = activity_df[(activity_df["ID"] == rec_id) & (activity_df["Amplitude"] == 4)].copy()
        # Keeping only the EXC neurons
        rec_data = rec_data[rec_data["Neuron"].str.startswith("EXC")]
        if hit_activated_only:
            # Selecting the neurons that are activated at least once during detected trials
            act_hit_neurons = set(rec_data[(rec_data["Behavior"] == True) & (rec_data["Resp"] == 1)]["Neuron"].values)
            rec_data[metric] = rec_data[metric].where(rec_data["Neuron"].isin(act_hit_neurons), np.nan)
            rec_data = rec_data.dropna(subset=[metric])
        # Building a data frame with detected trials and one with non-detected trials
        long_hit_data = rec_data[rec_data["Behavior"] == True].copy()
        long_miss_data = rec_data[rec_data["Behavior"] == False].copy()
        hit_data = long_hit_data.pivot(index=["Neuron"], columns=["Trial"], values=metric)
        miss_data = long_miss_data.pivot(index=["Neuron"], columns=["Trial"], values=metric)
        # Concatenate columns with names indicating their original dataframe
        def concat_cols(df, name):
            df_copy = df.copy()
            df_copy.columns = [f"{name}_{col}" for col in df_copy.columns]
            return df_copy
        # Concatenate dataframes
        concat_hit = concat_cols(hit_data, 'Hit')
        concat_miss = concat_cols(miss_data, 'Miss')
        combined_df = pd.concat([concat_hit, concat_miss], axis=1)
        # Compute cosine similarity between columns
        cos_sim_matrix = pd.DataFrame(cosine_similarity(combined_df.T), index=combined_df.columns, columns=combined_df.columns)
        # Transform to long format (pairwise)
        similarity_df = cos_sim_matrix.stack().reset_index()
        similarity_df.columns = ['Column_1', 'Column_2', 'Cosine_Similarity']
        # Add table names
        similarity_df['Table_1'] = similarity_df['Column_1'].apply(lambda x: x.split('_')[0])
        similarity_df['Table_2'] = similarity_df['Column_2'].apply(lambda x: x.split('_')[0])
        # Filtering out self-similarity and duplicate pairs
        similarity_df = similarity_df[similarity_df['Column_1'] != similarity_df['Column_2']]
        similarity_df = similarity_df.reset_index(drop=True)
        similarity_df["abs_cos_sim"] = abs(similarity_df["Cosine_Similarity"])
        similarity_df = similarity_df.drop(columns=['Cosine_Similarity', "Column_1", "Column_2"])
        grouped = similarity_df.groupby(['Table_1', 'Table_2'], as_index=False).mean()
        hit_sim = grouped[(grouped["Table_1"] == "Hit") & (grouped["Table_2"] == "Hit")]["abs_cos_sim"].values[0] if len(grouped[(grouped["Table_1"] == "Hit") & (grouped["Table_2"] == "Hit")]["abs_cos_sim"].values) > 0 else np.nan
        miss_sim = grouped[(grouped["Table_1"] == "Miss") & (grouped["Table_2"] == "Miss")]["abs_cos_sim"].values[0] if len(grouped[(grouped["Table_1"] == "Miss") & (grouped["Table_2"] == "Miss")]["abs_cos_sim"].values) > 0 else np.nan
        btw_sim = grouped[grouped["Table_1"] != grouped["Table_2"]]["abs_cos_sim"].values[0]
        rows.append({"Genotype": rec_data["Genotype"].values[0], "ID": rec_id, "Hit_sim": hit_sim,
                     "Miss_sim": miss_sim, "Between_sim": btw_sim})
    results = pd.DataFrame(rows)
    # Plotting the difference
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 12), constrained_layout=True)
    for row, genotype in enumerate(results["Genotype"].unique()):
        colors = color_dict[genotype]
        data = results[results["Genotype"] == genotype].copy()
        ppt.boxplot(ax[row, 0], data["Hit_sim"].values, data["Miss_sim"].values, ylabel="Mean_abs_cos_sim", paired=True, title=f"{genotype} - Hit/Miss", ylim=[],
                    colors=colors, det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 1], data["Hit_sim"].values, data["Between_sim"].values, ylabel="Mean_abs_cos_sim", paired=True, title=f"{genotype} - Hit/btw", ylim=[],
                    colors=[colors[0], "purple"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 2], data["Miss_sim"].values, data["Between_sim"].values, ylabel="Mean_abs_cos_sim", paired=True, title=f"{genotype} - Miss/btw", ylim=[],
                    colors=[colors[1], "purple"], det_marker=False, force_markers_identity=False)
    wt = results[results["Genotype"] == "WT"].copy()
    hypo = results[results["Genotype"] == "KO-Hypo"].copy()
    colors2 = {"Hit_sim": [ppt.wt_color, ppt.hypo_color], "Miss_sim": [ppt.wt_light_color, ppt.hypo_light_color],
               "Between_sim": ["purple", "purple"]}
    for row2, comp in enumerate(["Hit_sim", "Miss_sim", "Between_sim"]):
        ppt.boxplot(ax[row2, 3], wt[comp].values, hypo[comp].values, ylabel="Mean_abs_cos_sim", paired=False, title=f"WT/KO-Hypo - {comp}", ylim=[],
                    colors=colors2[comp], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of mean cosine similarity across pairs of trials within and between condition, threshold trials"
                 f"\n {metric} [hit_activated_only={hit_activated_only}]", fontsize=12)
    fig.canvas.manager.set_window_title(f"Trials_pairs_{metric}_cos_sim_{hit_activated_only}")
    plt.show()
    return results

# endregion ============================================================================================================
# region ======================================== E/I Ratio ============================================================

def get_ei_ratio_df(recs):
    """
    Returns a Dataframe withl each row being the E/I ratio for a specific trial for a specific animal. E/I ratio is
    defined as the percentage of activated ExC neurons over the percentage of activated INH neurons.
    TODO: find a better definition of E/I ratio to avoid infinity values

    Parameters
    ----------
    recs

    Returns
    -------

    """
    rows = []
    for rec in recs:
        behavior_vector = rec.detected_stim
        amplitude_vector = rec.stim_ampl
        act_EXC_vector = rec.get_perc_resp(pattern=1, n_type="EXC")
        act_INH_vector = rec.get_perc_resp(pattern=1, n_type="INH")
        # === Computing E/I ratio
        # 1) Basic E/I ratio
        ei_ratio_vector = np.divide(act_EXC_vector, act_INH_vector)
        # 2) E/I ratio + cste
        epsilon = 0.0001
        cst_ei_ratio_vector = np.divide(act_EXC_vector + epsilon, act_INH_vector + epsilon)
        # 3) Normalized E/I ratio
        norm_act_EXC_vector = (act_EXC_vector - np.mean(act_EXC_vector)) / np.std(act_EXC_vector)
        norm_act_INH_vector = (act_INH_vector - np.mean(act_INH_vector)) / np.std(act_INH_vector)
        norm_ei_ratio_vector = np.divide(norm_act_EXC_vector, norm_act_INH_vector)
        # 4) Log E/I Ratio + cste
        log_cst_ei_ratio_vector = np.log(cst_ei_ratio_vector)
        # 5) Log normalized E/I Ratio
        log_norm_ei_ratio_vector = np.log(norm_ei_ratio_vector)
        # 6) Normalized difference
        norm_dif_ei_vector = np.divide(act_EXC_vector - act_INH_vector, act_EXC_vector + act_INH_vector)
        # 6) Difference
        dif_ei_vector = act_EXC_vector - act_INH_vector
        for trial_id in range(len(behavior_vector)):
            rows.append({"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.session_threshold,
                         "Trial": trial_id, "Amplitude": amplitude_vector[trial_id],
                         "Behavior": behavior_vector[trial_id],
                         "EI_ratio": ei_ratio_vector[trial_id],
                         "EI_ratio_cste": cst_ei_ratio_vector[trial_id],
                         "EI_ratio_norm": norm_ei_ratio_vector[trial_id],
                         "EI_ratio_log_cste": log_cst_ei_ratio_vector[trial_id],
                         "EI_ratio_log_norm": log_norm_ei_ratio_vector[trial_id],
                         "EI_ratio_norm_dif": norm_dif_ei_vector[trial_id],
                         "EI_ratio_dif": dif_ei_vector[trial_id]})
    return pd.DataFrame(rows)


def correlate_behavior(data, column=None):
    if column is None:
        column = data.columns[-1]
    rows = []
    for rec_id in data["ID"].unique():
        rec_data = data[data["ID"] == rec_id]
        r, pval = ss.pointbiserialr(rec_data[column], rec_data["Behavior"])
        rows.append({"ID": rec_data["ID"].values[0], "Genotype": rec_data["Genotype"].values[0],
                     "Threshold": rec_data["Threshold"].values[0], "R2": r**2, "pval": pval})
    return pd.DataFrame(rows)


def compare_ei_ratio(ei_df, gp1="WT", gp2="KO-Hypo", column=None):
    """
    Plot the comparison of the E/I ratio between detected and non detected trials and between genotypes
    Parameters
    ----------
    data
    column

    Returns
    -------

    """
    colors_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                   "KO": [ppt.ko_color, ppt.ko_light_color]}
    if column is None:
        column = ei_df.columns[-1]
    filtered_data = ei_df[ei_df["Amplitude"] == ei_df["Threshold"]].drop(columns=["Trial", "Amplitude"])
    data = filtered_data.groupby(["Genotype", "ID", "Threshold", "Behavior"], as_index=False).mean()
    # Plotting the data
    gp1_det = data[(data["Genotype"] == gp1) & (data["Behavior"] == True)][column].values
    gp1_undet = data[(data["Genotype"] == gp1) & (data["Behavior"] == False)][column].values
    gp2_det = data[(data["Genotype"] == gp2) & (data["Behavior"] == True)][column].values
    gp2_undet = data[(data["Genotype"] == gp2) & (data["Behavior"] == False)][column].values
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), constrained_layout=True)
    ppt.boxplot(ax[0, 0], gp1_det, gp1_undet, ylabel="E/I ratio", paired=True, title=f"{gp1}", ylim=[], colors=colors_dict[gp1], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[0, 1], gp2_det, gp2_undet, ylabel="E/I ratio", paired=True, title=f"{gp2}", ylim=[], colors=colors_dict[gp2], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 0], gp1_det, gp2_det, ylabel="E/I ratio", paired=False, title="Detected Trials", ylim=[], colors=[colors_dict[gp1][0], colors_dict[gp2][0]], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 1], gp1_undet, gp2_undet, ylabel="E/I ratio", paired=False, title="Non-Detected Trials", ylim=[], colors=[colors_dict[gp1][1], colors_dict[gp2][1]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"E/I ratio comparison ({gp1} & {gp2})\n [method = {column}]", fontsize=16)
    fig.canvas.manager.set_window_title(f"EI_comparison_{gp1}_{gp2})_{column}")
    plt.show()
    return data

def population_EI_ratio(recs, pyr_inhibition=False):
    """ Computes the averaged population E/I ratio in hit and miss trials for both genotypes and compare them"""
    rows = []
    for rec in recs:
        n_exc = rec.zscore_exc.shape[0]
        n_inh = rec.zscore_inh.shape[0]
        threshold_stim_vector = rec.stim_ampl == rec.session_threshold
        if pyr_inhibition:
            I_type = "EXC"
            I_activity = -1
            I_nb = n_exc
            I_label = "% inhibited EXC"
        else:
            I_type = "INH"
            I_activity = 1
            I_nb = n_inh
            I_label = "% activated INH"
        hit_exc = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, (rec.detected_stim & threshold_stim_vector)] == 1, axis=0)) / n_exc
        miss_exc = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, (~rec.detected_stim & threshold_stim_vector)] == 1, axis=0)) / n_exc
        hit_inh = np.mean(np.count_nonzero(rec.matrices[I_type]["Responsivity"][:, (rec.detected_stim & threshold_stim_vector)] == I_activity, axis=0)) / I_nb
        miss_inh = np.mean(np.count_nonzero(rec.matrices[I_type]["Responsivity"][:, (~rec.detected_stim & threshold_stim_vector)] == I_activity, axis=0)) / I_nb
        ei_hit = hit_exc / hit_inh
        ei_miss = miss_exc / miss_inh
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "EI_hit": ei_hit, "EI_miss": ei_miss})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24, 8), constrained_layout=True)
    wt_hit = data[data["Genotype"] == "WT"]["EI_hit"]
    wt_hit_un = wt_hit[np.isfinite(wt_hit)].values
    ko_hit = data[data["Genotype"] == "KO-Hypo"]["EI_hit"]
    ko_hit_un = ko_hit[np.isfinite(ko_hit)].values
    wt_miss = data[data["Genotype"] == "WT"]["EI_miss"]
    wt_miss_un = wt_miss[np.isfinite(wt_miss)].values
    ko_miss = data[data["Genotype"] == "KO-Hypo"]["EI_miss"]
    ko_miss_un = ko_miss[np.isfinite(ko_miss)].values
    wt_hit_paired = wt_hit[(np.isfinite(wt_hit) & np.isfinite(wt_miss))].values
    wt_miss_paired = wt_miss[(np.isfinite(wt_hit) & np.isfinite(wt_miss))].values
    ko_hit_paired = ko_hit[(np.isfinite(ko_hit) & np.isfinite(ko_miss))].values
    ko_miss_paired = ko_miss[(np.isfinite(ko_hit) & np.isfinite(ko_miss))].values
    ppt.boxplot(ax[0], wt_hit_un, ko_hit_un, paired=False, ylabel="E:I ratio", ylim=[], title="Hit trials", colors=[ppt.wt_color, ppt.hypo_color], det_marker=False)
    ppt.boxplot(ax[1], wt_miss_un, ko_miss_un, paired=False, ylabel="E:I ratio", ylim=[], title="Miss trials", colors=[ppt.wt_light_color, ppt.hypo_light_color], det_marker=False)
    ppt.boxplot(ax[2], wt_hit_paired, wt_miss_paired, paired=True, ylabel="E:I ratio", ylim=[], title="WT", colors=[ppt.wt_color, ppt.wt_light_color], det_marker=True)
    ppt.boxplot(ax[3], ko_hit_paired, ko_miss_paired, paired=True, ylabel="E:I ratio", ylim=[], title="KO-Hypo", colors=[ppt.hypo_color, ppt.hypo_light_color], det_marker=True)
    fig.suptitle(f"E:I ratio comparison between genotypes\n defined as % activated EXC/{I_label}", fontsize=12)
    fig.canvas.manager.set_window_title(f"EI_comparison")
    plt.show()
    return data

def correlation_gaba_act_pyr_inh(recs):
    """Correlates the number of activated GABAergic neurons with the number of inhibited Pyramidal neurons"""
    color_dict = {"WT": ppt.wt_color, "KO-Hypo": ppt.hypo_color, "KO": ppt.ko_color}
    rows = []
    for rec in recs:
        n_exc = rec.zscore_exc.shape[0]
        n_inh = rec.zscore_inh.shape[0]
        threshold_stim_vector = rec.stim_ampl == rec.session_threshold
        pyr_inh = (np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, threshold_stim_vector] == -1, axis=0)) / n_exc) * 100
        gaba_act = (np.mean(np.count_nonzero(rec.matrices["INH"]["Responsivity"][:, threshold_stim_vector] == 1, axis=0)) / n_inh) * 100
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "pyr_inh": pyr_inh, "gaba_act": gaba_act})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)
    for col, geno in enumerate(data["Genotype"].unique()):
        x = data[data["Genotype"] == geno]["gaba_act"]
        y = data[data["Genotype"] == geno]["pyr_inh"]
        results = dict(linregress(x, y)._asdict())
        r2 = results["rvalue"] ** 2
        line = results["slope"] * x + results["intercept"]
        # Plot the data points and regression line
        ax[col].plot(x, line, color=color_dict[geno], lw=2)
        ax[col].scatter(x, y, color=color_dict[geno], alpha=0.7, s=10, marker="+")
        ax[col].text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}", transform=ax[col].transAxes, fontsize=8,
                verticalalignment="top", color="black")
        ax[col].set_title(geno, color=color_dict[geno])
        ax[col].set_xlabel("% activated GABAergic neurons", fontsize=8)
        ax[col].set_ylabel("% inhibited Pyramidal neurons", fontsize=8)
    fig.suptitle("Correlation of the mean percentage of activated GABAergic interneurons with the number of inhibited Pyramidal neurons for threshold stimuli", fontsize=12)
    fig.canvas.manager.set_window_title(f"Correlation_nb_neurons_inhibition")
    plt.show()
    return data


# endregion ============================================================================================================
# region ========================================== Noise ==============================================================

def baseline_and_SNR(recruitment_df):
    """
    Compares the baseline activity and the SNR between WT and KO-Hypo

    Parameters
    ----------
    activity_df

    Returns
    -------

    """
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    grouped = recruitment_df.groupby(["Genotype", "ID", "threshold", "behavior", "amplitude"], as_index=False).mean()
    bsl = grouped[grouped["amplitude"] == 0]
    hit = grouped[(grouped["amplitude"] == grouped["threshold"]) & (grouped["behavior"] == True)]
    miss = grouped[(grouped["amplitude"] == grouped["threshold"]) & (grouped["behavior"] == False)]
    # Baseline
    fig, ax = plt.subplots(nrows=4, ncols=8, figsize=(25, 14), constrained_layout=True)
    for col_id, col in enumerate(["act_EXC_perc", "inh_EXC_perc", "act_INH_perc", "inh_INH_perc"]):
        for row_id, genotype in enumerate(bsl["Genotype"].unique()):
            ppt.boxplot(ax[row_id, 2 * col_id], bsl[bsl["Genotype"] == genotype][col].values, hit[hit["Genotype"] == genotype][col].values,
                        ylabel=col, paired=True, title=f"{genotype} Bsl/Hit", ylim=[],
                        colors=["purple", color_dict[genotype][0]], det_marker=False, force_markers_identity=False)
            ppt.boxplot(ax[row_id, 2 * col_id + 1], bsl[bsl["Genotype"] == genotype][col].values, miss[miss["Genotype"] == genotype][col].values,
                        ylabel=col, paired=True, title=f"{genotype} Bsl/Miss", ylim=[],
                        colors=["purple", color_dict[genotype][1]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[3, 2 * col_id], bsl[bsl["Genotype"] == "WT"][col].values, bsl[bsl["Genotype"] == "KO-Hypo"][col].values,
                    ylabel=col, paired=False, title="WT/KO-Hypo Bsl", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[3, 2 * col_id + 1], bsl[bsl["Genotype"] == "WT"][col].values, bsl[bsl["Genotype"] == "KO"][col].values,
                    ylabel=col, paired=False, title="WT/KO Bsl", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO"][0]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of percentage of recruited neurons during hit and miss trials to baseline (no-go trials)", fontsize=12)
    fig.canvas.manager.set_window_title(f"Baseline")
    # SNR
    fig_snr, ax_snr = plt.subplots(nrows=5, ncols=6, figsize=(25, 14), constrained_layout=True)
    for df in [hit, miss, bsl]:
        df["recr_EXC_perc"] = df["act_EXC_perc"] + df["inh_EXC_perc"]
        df["recr_INH_perc"] = df["act_INH_perc"] + df["inh_INH_perc"]
    key_cols = ["Genotype", "ID"]
    bsl_snr_cols = bsl.columns[-6:]  # last 6 columns assumed to be SNR-related
    hit_snr = hit[key_cols + list(bsl_snr_cols)].merge(bsl[key_cols + list(bsl_snr_cols)], on=key_cols, suffixes=("_hit", "_bsl"))
    miss_snr = miss[key_cols + list(bsl_snr_cols)].merge(bsl[key_cols + list(bsl_snr_cols)], on=key_cols, suffixes=("_miss", "_bsl"))
    for col in bsl_snr_cols:
        hit_snr[col] = hit_snr[f"{col}_hit"] / hit_snr[f"{col}_bsl"]
        miss_snr[col] = miss_snr[f"{col}_miss"] / miss_snr[f"{col}_bsl"]
    hit_snr = hit_snr[key_cols + list(bsl_snr_cols)]
    miss_snr = miss_snr[key_cols + list(bsl_snr_cols)]
    for col_id, col in enumerate(bsl_snr_cols):
        for row_id, genotype in enumerate(bsl["Genotype"].unique()):
            ppt.boxplot(ax_snr[row_id, col_id], hit_snr[hit_snr["Genotype"] == genotype][col].values, miss_snr[miss_snr["Genotype"] == genotype][col].values,
                        ylabel=col, paired=True, title=f"Hit/Miss {genotype}", ylim=[],
                        colors=color_dict[genotype], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_snr[3, col_id],
                    hit_snr[hit_snr["Genotype"] == "WT"][col].values,
                    hit_snr[hit_snr["Genotype"] == "KO-Hypo"][col].values,
                    ylabel=col, paired=True, title="Hit SNR WT/KO-Hypo", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_snr[4, col_id],
                    miss_snr[miss_snr["Genotype"] == "WT"][col].values,
                    miss_snr[miss_snr["Genotype"] == "KO-Hypo"][col].values,
                    ylabel=col, paired=True, title="Miss SNR WT/KO-Hypo", ylim=[],
                    colors=[color_dict["WT"][1], color_dict["KO-Hypo"][1]], det_marker=False, force_markers_identity=False)
    fig_snr.suptitle(f"Comparison of the SNR between hit and miss trials and between genotypes", fontsize=12)
    fig_snr.canvas.manager.set_window_title("SNR")
    plt.show()
    return bsl, hit, miss, hit_snr, miss_snr

# endregion ============================================================================================================


if __name__ == '__main__':
    BMS_analysis = True
    # region ====== Initialisation of recs instances ======
    if BMS_analysis:
        directory = "C:/Users/cvandromme/Desktop/Tactile_detection/Data_DMSO_BMS/"
        roi_path = "C:/Users/cvandromme/Desktop/Tactile_detection/Fmko_bms&dmso_info.xlsx"
    else:
        directory = "C:/Users/cvandromme/Desktop/Tactile_detection/Data/"
        roi_path = "C:/Users/cvandromme/Desktop/Tactile_detection/FmKO_ROIs&inhibitory.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/noise_assessment/"
    roi_info = pd.read_excel(roi_path)
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]
    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec
    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    if BMS_analysis:
        recs = {f"{ar.get().filename}-{ar.get().genotype.split("-")[1]}": ar.get() for ar in async_results}
    else:
        recs = {ar.get().filename: ar.get() for ar in async_results}
    # endregion
    # Dropping 5886 from the noise assessment analysis because its computed threshold is 3 (10% hit rate for 2µm and 90% for 4µm)
    if not BMS_analysis:
        excluded_rec = recs.pop(5886)
    # region ====== Comparison of threshold to session threshold ======
    rows = []
    for rec in recs.values():
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.threshold, "session_threshold": rec.session_threshold, "session_x0": rec.x0_psy})
    session_threshold = pd.DataFrame(rows)

    from percephone.utils.math_formulas import sigmoid_fit
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(20, 12), constrained_layout=True)
    axs = ax.flatten()
    for i, rec in enumerate(recs.values()):
        axs[i].set_title(f"{rec.filename} {rec.genotype}- {rec.threshold}/{rec.session_threshold}({rec.x0_psy:.2f})", fontsize=12)
        axs[i].set_ylim(0, 1)
        axs[i].scatter(np.arange(start=2, stop=13, step=2), rec.hit_rates[1:], s=5)
        if rec.filename == 7554 and rec.genotype == "KO-DMSO":
            x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates, p0=[4.0, 1.0])
        else:
            x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates)
        axs[i].plot(x, y, color='red', lw=2, alpha=0.75)
    plt.show()
    # endregion

    # region ====== TBT variability ======
    # recruitement_df = get_features(recs.values(), amp_delay=False)
    # activity_long_df = get_mean_trial_activity_df(recs.values(), zscore=True)
    # activity_long_dff = get_mean_trial_activity_df(recs.values(), zscore=False)
    # prestim_df = prestim_activated_neurons(filtered_activity_df)
    # prestim_vector_df = prestim_act_vector(activity_long_df, metric="Stim_mean", hit_activated_only=False)
    # prestim_vector_dff = prestim_act_vector(activity_long_dff, metric="Diff_mean", hit_activated_only=True)
    # tbt_recr_var_df = compare_tbt_var_per_amp(recruitement_df)
    # pca_df = pca(activity_long_df)
    # endregion

    # region ====== Pre-stimulus ======
    # filtered_activity_df = filter_stim_recruited_neurons(activity_long_dff[activity_long_dff["ID"] != 4456])
    # filtered_activity_df = filter_stim_recruited_neurons(activity_long_dff)
    # grouped_comp, gp_hit, gp_miss = prestim_comp(filtered_activity_df)
    # snr_data = snr_comp(filtered_activity_df)
    # snr_wt, snr_hypo = compare_to_wt_threshold(filtered_activity_df)
    # prestim_act_df = prestim_influence_neuron_activation(filtered_activity_df)
    # endregion

    # region ====== E/I Ratio ======
    # ei_df = get_ei_ratio_df(recs.values())
    # ei_behavior_df = correlate_behavior(ei_df, column="EI_ratio_cste")
    # ei_comp_df = compare_ei_ratio(ei_df, column="EI_ratio_norm")

    # ei_data = population_EI_ratio(recs.values(), pyr_inhibition=True)
    # inh_corr_data = correlation_gaba_act_pyr_inh(recs.values())
    # endregion

    # region ====== Baseline ======
    # baseline_df, hit_df, miss_df, hit_snr, miss_snr = baseline_and_SNR(recruitement_df)
    # endregion
    # rows = []
    # for rec in recs.values():
    #     hit_thres = (rec.detected_stim == True) & (rec.stim_ampl == 4)
    #     miss_thres = (rec.detected_stim == False) & (rec.stim_ampl == 4)
    #     rows.append({"Genotype": rec.genotype, "ID": rec.filename, "Hit": hit_thres.sum(), "Miss": miss_thres.sum()})
    # test = pd.DataFrame(rows)
