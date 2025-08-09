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
import statsmodels.formula.api as smf

import percephone.core.recording as pc
import percephone.plts.stats as ppt
from Figures.response_decoding import get_activity_by_frame_df
from Figures.stimulus_encoding import get_features
from percephone.plts.utils import stat_boxplot


# endregion
# region ======================================== TBT variability ======================================================

color_dict = {
    # WT
    "WT": [ppt.wt_color, ppt.wt_light_color],
    "WT-DMSO": [ppt.wt_color, ppt.wt_light_color],
    "WT-BMS": [ppt.wt_bms_color, ppt.wt_bms_light_color],
    # KO-Hypo
    "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
    "KO-DMSO": [ppt.hypo_color, ppt.hypo_light_color],
    "KO-BMS": [ppt.hypo_bms_color, ppt.hypo_bms_light_color],
    # KO
    "KO": [ppt.ko_color, ppt.ko_light_color]}

def get_mean_trial_activity_df(recs, zscore=True, BMS=False):
    prestim_frames = 15
    rows = []
    for rec in recs:
        if BMS:
            condition = rec.genotype.split("-")[-1]
            rec_id = f"{rec.filename}-{condition}"
        else:
            rec_id = rec.filename
        exc_activity = rec.zscore_exc if zscore else rec.df_f_exc
        inh_activity = rec.zscore_inh if zscore else rec.df_f_inh
        behavior_vector = rec.detected_stim
        amplitude_vector = rec.stim_ampl
        stim_time_vector = rec.stim_time
        licks_vector = rec.lick_time
        stim_duration_vector = rec.stim_durations
        fa_vector = []
        # Differentiating the False Alarm and Correct Rejection
        for trial_id, (trial_amp, time) in enumerate(zip(amplitude_vector, stim_time_vector)):
            if trial_amp == 0:
                # Defining if the no-go is a FA or CR
                time_diff_vector = licks_vector - ([time] * len(licks_vector))
                is_fa = np.any((time_diff_vector >= 0) & (time_diff_vector < 75))
            else:
                is_fa = np.nan
            fa_vector.append(is_fa)
        for neuron_type, activity in zip(["EXC", "INH"], [exc_activity, inh_activity]):
            resp_mat = rec.matrices[neuron_type]["Responsivity"]
            AUC = rec.matrices[neuron_type]["AUC"]
            cum_AUC = rec.matrices[neuron_type]["cum_AUC"]
            cum_AUC_pre = rec.matrices[neuron_type]["cum_AUC_pre"]
            cum_AUC_fixpre = rec.matrices[neuron_type]["cum_AUC_fixpre"]
            pos_AUC_fixpre = rec.matrices[neuron_type]["pos_AUC_fixpre"]
            neg_AUC_fixpre = rec.matrices[neuron_type]["neg_AUC_fixpre"]
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
                    row = {"Genotype": rec.genotype, "ID": rec_id, "Threshold": rec.session_threshold,
                           "Trial": trial_id, "Amplitude": amplitude_vector[trial_id],
                           "Behavior": behavior_vector[trial_id],
                           "FA": fa_vector[trial_id], "Neuron": f"{neuron_type}_{n_id}",
                           "Resp": resp_mat[n_id, trial_id],
                           "Stim_mean": trial_mean_activity, "Stim_std": trial_std_activity,
                           "Prestim_mean": prestim_mean_activity, "Prestim_std": prestim_std_activity,
                           "AUC": AUC[n_id, trial_id], "cum_AUC": cum_AUC[n_id, trial_id],
                           "cum_AUC_pre": cum_AUC_pre[n_id, trial_id], "cum_AUC_fixpre": cum_AUC_fixpre[n_id, trial_id],
                           "pos_AUC_fixpre": pos_AUC_fixpre[n_id, trial_id], "neg_AUC_fixpre": neg_AUC_fixpre[n_id, trial_id]}
                    rows.append(row)
    activity_df = pd.DataFrame(rows)
    activity_df["Diff_mean"] = activity_df["Stim_mean"] - activity_df["Prestim_mean"]
    activity_df["Abs_Diff_mean"] = abs(activity_df["Stim_mean"] - activity_df["Prestim_mean"])
    activity_df["Diff_std"] = activity_df["Stim_std"] - activity_df["Prestim_std"]
    activity_df["Ratio_mean"] = activity_df["Stim_mean"] / activity_df["Prestim_mean"]
    activity_df["Ratio_std"] = activity_df["Stim_std"] / activity_df["Prestim_std"]
    activity_df["Abs_Ratio_std"] = abs(activity_df["Stim_std"]) / abs(activity_df["Prestim_std"])
    return activity_df


def pca(mean_activity_df, split="Behavior"):
    """Perfroms a PCA trying to cluster trials using the mean activity of each neuron during each trials. Each point is
    a trial, each component a combination of neurons"""
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(20, 14), constrained_layout=True)
    ax = axs.flatten()
    rows = []
    for ax_id, rec_id in enumerate(mean_activity_df["ID"].unique()):
        rec_data = mean_activity_df[mean_activity_df["ID"] == rec_id].copy()
        rec_data["No-go"] = np.where(rec_data.Amplitude == 0, "No-Go", "Go")
        rec_data["Hit_FA"] = np.where(rec_data.Behavior, "Hit", np.where(rec_data.FA == True, "FA", "NA"))
        rec_data["Miss_CR"] = np.where(rec_data.FA == False, "CR", np.where(~rec_data.Behavior, "Miss", "NA"))
        if split in ["Hit_FA", "Miss_CR"]:
            rec_data = rec_data[rec_data[split] != "NA"]
        genotype = rec_data["Genotype"].values[0]
        mouse_id = rec_data["ID"].values[0]
        threshold = rec_data["Threshold"].values[0]
        data = rec_data.pivot(index=["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Behavior", "No-go", "Hit_FA", "Miss_CR"],
                              columns="Neuron", values="Stim_mean").reset_index()
        y = data[split]
        X = data.drop(columns=["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Behavior", "No-go", "Hit_FA", "Miss_CR"])
        # Performing a PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        # Storing the values in a DataFrame and plotting them
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df[split] = y.values
        for behavior_id, behavior_label in enumerate(sorted(pca_df[split].unique(), reverse=True)):
            subset = pca_df[pca_df[split] == behavior_label]
            ax[ax_id].scatter(subset["PC1"], subset["PC2"], c=color_dict[genotype][behavior_id], label=behavior_label,
                              alpha=0.7, s=5)
        ax[ax_id].set_xlabel(f"PC1 ({explained_var[0]:.1%})", fontsize=10)
        ax[ax_id].set_ylabel(f"PC2 ({explained_var[1]:.1%})", fontsize=10)
        ax[ax_id].tick_params(axis='both', labelsize=10)
        ax[ax_id].set_title(f"{mouse_id} ({genotype})", color=color_dict[genotype][0], fontsize=10)
        ax[ax_id].legend(fontsize=8)
        rows.append({"Genotype": genotype, "ID": mouse_id, "Threshold": threshold,
                     "PC1": explained_var[0], "PC2": explained_var[1]})
    fig.suptitle(f"PCA of zean Z-score Across Trials\n{split}", fontsize=20)
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
    hit_test, hit_posthoc = ppt.curveplot(ax[0], hit[hit["Genotype"] != "KO"], between="Genotype", within="amplitude",
                                          variable="act_EXC_perc",
                                          title="Variation of std of the % of activated EXC neurons across amplitudes for hit trials",
                                          ylabel=None, xlabel=None, ylim=None, colors=[ppt.hypo_color, ppt.wt_color],
                                          id_display=True, legend_display=True,
                                          qq_show=True, transformation="yeojohnson", consider_normality=False,
                                          consider_homogeneity=False)
    miss_test, miss_posthoc = ppt.curveplot(ax[1], miss[miss["Genotype"] != "KO"], between="Genotype",
                                            within="amplitude",
                                            variable="act_EXC_perc",
                                            title="Variation of std of the % of activated EXC neurons across amplitudes for miss trials",
                                            ylabel=None, xlabel=None, ylim=None, colors=[ppt.hypo_color, ppt.wt_color],
                                            id_display=True, legend_display=True,
                                            qq_show=True, transformation=None, consider_normality=False,
                                            consider_homogeneity=False)
    plt.show()
    return hit_test, hit_posthoc, miss_test, miss_posthoc


def compare_pop_tbt_var(features_df):
    """Compare the standard deviation of the number of recruited neurons during threshold trials between WT and KO-Hypo
    mice. This is used as a readout for the population tbt var"""
    data = features_df[features_df.amplitude == features_df.threshold].copy()
    data = data.drop(columns=["bounded_x0", "amplitude", "threshold"])
    variable_col = ["act_EXC_perc", "inh_EXC_perc", "act_INH_perc", "inh_INH_perc"]
    gp_data = (data.groupby(["ID", "Genotype"]).agg(**{f"{col}_std": (col, "std") for col in variable_col},
                                                n_trials=("behavior", "size")).reset_index())
    for col in variable_col:
        model = smf.ols(f"{col}_std ~ C(Genotype, Treatment(reference='WT')) + n_trials", gp_data)
        res = model.fit()
        print(f"====== {col} ======")
        print(res.summary())
    return data

# pop_tbt_var_df = compare_pop_tbt_var(recruitment_df)

def filter_reliable(mean_activity_df, recs, pattern="resp", get_non_reliable=False):
    """Filter the mean activity DataFrame to keep only the reliably responding neurons"""
    df = mean_activity_df.copy()
    for rec in recs:
        rec.reliable_responders()
        # Retrieving the lis corresponding to the pattern
        pattern_dict = {"resp": [rec.reliable_exc, rec.reliable_inh],
                        "act": [rec.reliable_act_exc, rec.reliable_act_inh],
                        "inh": [rec.reliable_inh_exc, rec.reliable_inh_inh]}
        # Creating a combined list of reliable responders neurons
        rec.reliable_responders()
        reliable_exc = [f"EXC_{i}" for i in pattern_dict[pattern][0]]
        reliable_inh = [f"INH_{i}" for i in pattern_dict[pattern][1]]
        reliable = reliable_exc + reliable_inh
        # Filtering out the non-reliable neurons
        if get_non_reliable:
            df = df[~((df["ID"] == rec.filename) & (df["Neuron"].isin(reliable)))].reset_index(drop=True)
        else:
            df = df[~((df["ID"] == rec.filename) & ~(df["Neuron"].isin(reliable)))].reset_index(drop=True)
    return df


def compare_nb_reliable_responders(recs):
    rows = []
    for rec in recs:
        rec.reliable_responders()
        rows.append({"ID": rec.filename, "Genotype": rec.genotype,
                     "Nb_reliably_act": len(rec.reliable_act_exc), "Nb_reliably_inh": len(rec.reliable_inh_exc)})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), constrained_layout=True)
    for col, pattern in enumerate(["Nb_reliably_act", "Nb_reliably_inh"]):
        ppt.boxplot(ax[col], data[data["Genotype"] == "WT"][pattern].values,
                    data[data["Genotype"] == "KO-Hypo"][pattern].values,
                    ylabel="Nb reliable responders", paired=False, title=pattern, ylim=[0, 80],
                    colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
    fig.suptitle("Comparison of the number of EXC reliable responders between genotypes", fontsize=14)
    fig.canvas.manager.set_window_title("Nb_reliable_comp")
    plt.show()
    return data


def compare_threshold_trials_cosine_similarity(mean_activity_df, include_gaba=False, title_precision="",
                                               amps={"WT": "threshold", "KO-Hypo": "threshold"}, centroid=False,
                                               min_nb_trials=2):
    """Compares the mean cosine similarity between each pairs of threshold trials between genotypes, considering each
    behavioral label independently. Mean cosine similarity across trials represents how reliably the same population of
    neurons participates (and in which direction) across detected (or all) trials. Are threshold trials more
    similar in WT compared to KO-Hypo ?"""
    if "KO" not in amps:
        amps.update({"KO": "threshold"})
    rows = []
    # Defining the figures for the matrices plotting
    names = ["Resp_All", "Resp_Hit", "Resp_Miss", "Resp_Nogo",  #"Resp_CR", "Resp_FA",
             "Stim_mean_All", "Stim_mean_Hit", "Stim_mean_Miss", "Stim_mean_Nogo"]  #, "Stim_mean_CR", "Stim_mean_FA"]
    figs = {}
    axes = {}
    for name in names:
        fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(24, 16), constrained_layout=True)
        figs[name] = fig
        axes[name] = axs.flatten()
    for i, rec_id in enumerate(mean_activity_df.ID.unique()):
        # Retrieving the data for the recording
        rec_data = mean_activity_df[mean_activity_df["ID"] == rec_id]
        rec_12_data = rec_data[rec_data["Amplitude"] == 12].sort_values(by=["Trial", "Neuron"]).copy()
        genotype = rec_data.Genotype.values[0]
        threshold = rec_data.Threshold.values[0]
        amplitude = amps[genotype]
        # Selecting the trials corresponding to the desired amplitude
        if amplitude == "threshold":
            rec_data = rec_data[rec_data["Amplitude"] == rec_data["Threshold"]].sort_values(
                by=["Trial", "Neuron"]).copy()
        elif amplitude == "sub_threshold":
            rec_data = rec_data[rec_data["Amplitude"] == (rec_data["Threshold"] - 2)].sort_values(
                by=["Trial", "Neuron"]).copy()
        elif amplitude == "supra_threshold":
            # Skipping the recordings with a threshold of 12 in this case
            if threshold == 12:
                print(f"{rec_id}({genotype}) → exclusion (threshold == 12)")
                continue
            else:
                rec_data = rec_data[rec_data["Amplitude"] == (rec_data["Threshold"] + 2)].sort_values(
                    by=["Trial", "Neuron"]).copy()
        elif amplitude == "wt_threshold":
            rec_data = rec_data[rec_data["Amplitude"] == 4].sort_values(by=["Trial", "Neuron"]).copy()
        elif isinstance(amplitude, list):
            rec_data = rec_data[rec_data["Amplitude"].isin(amplitude)].sort_values(by=["Trial", "Neuron"]).copy()
        else:
            raise ValueError("Amplitude must be either 'threshold', 'sub_threshold', 'wt_threshold', or a list")
        # Selecting the no-go trials
        rec_no_go_data = mean_activity_df[
            (mean_activity_df["ID"] == rec_id) & (mean_activity_df["Amplitude"] == 0)].sort_values(
            by=["Trial", "Neuron"]).copy()
        if not include_gaba:
            rec_data = rec_data[rec_data["Neuron"].str.startswith("EXC")]
            rec_no_go_data = rec_no_go_data[rec_no_go_data["Neuron"].str.startswith("EXC")]
            rec_12_data = rec_12_data[rec_12_data["Neuron"].str.startswith("EXC")]
        # Splitting hit and miss data
        miss_data = rec_data[rec_data["Behavior"] == False]
        hit_data = rec_data[rec_data["Behavior"] == True]
        miss_12_data = rec_12_data[rec_12_data["Behavior"] == False]
        hit_12_data = rec_12_data[rec_12_data["Behavior"] == True]
        n_hit = len(hit_data.Trial.unique())
        n_miss = len(miss_data.Trial.unique())
        hit_rate = n_hit / (n_hit + n_miss)
        nogo_data = rec_no_go_data
        row = {"ID": rec_id, "Genotype": genotype, "Hit Rate": hit_rate}
        # cr_data = rec_no_go_data[rec_no_go_data["FA"] == False]
        # fa_data = rec_no_go_data[rec_no_go_data["FA"] == True]
        # Computing cosine similarity (resp and zscore) between pairs of trials of the same behavioral label
        for behavior_label, behavior_data in zip(["All", "Miss", "Hit", "Nogo", "12_Miss", "12_Hit"],  #, "CR", "FA"],
                                                 [rec_data, miss_data, hit_data, nogo_data, miss_12_data,
                                                  hit_12_data]):  #, cr_data, fa_data]):
            n_trials = len(behavior_data.Trial.unique())
            if n_trials >= min_nb_trials:
                for metric in ["Resp", "Stim_mean", "Prestim_mean"]:
                    matrix = np.array(behavior_data.pivot(index="Trial", columns="Neuron", values=metric))

                    # ========== Cosine similarity computation and grouping ==========
                    if centroid:
                        # Testing another global similarity metric: mean distance of each trial to the centroid trial
                        centroid = matrix.mean(axis=0).reshape(1, -1)
                        mean_centroid_cos_sim = cosine_similarity(matrix, centroid).mean()
                        global_cos_metric = mean_centroid_cos_sim
                        centroid_str = "(centroid) "
                    else:
                        # Computing the cosine similarity matrix between pairs of trials
                        cos_sim_mat = cosine_similarity(matrix)
                        mean_cos_sim = cos_sim_mat[~np.eye(n_trials, dtype=bool)].mean()
                        global_cos_metric = mean_cos_sim
                        centroid_str = ""
                    # Computing the number of comparison of pairs of trials to normalize with it
                    n_comp = (n_trials ** 2 - n_trials) / 2

                    row[f"{metric}_{behavior_label}"] = global_cos_metric
                    if metric != "Prestim_mean":
                        if behavior_label not in ["12_Miss", "12_Hit"]:
                            # Retrieving the axes corresponding to the metrics to plot and plotting the rec matrix
                            ax = axes[f"{metric}_{behavior_label}"]
                            ax[i].imshow(cos_sim_mat, cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
                            ax[i].set_xlabel("Trial i", fontsize=10)
                            ax[i].set_ylabel("Trial j", fontsize=10)
                            ax[i].set_title(f"{int(rec_id)} - {genotype}({threshold})[{global_cos_metric:.2f}]",
                                            fontsize=12)
            else:
                if behavior_label not in ["12_Miss", "12_Hit"]:
                    for metric in ["Resp", "Stim_mean"]:
                        ax = axes[f"{metric}_{behavior_label}"]
                        ax[i].axis('off')
                        ax[i].set_title(f"{int(rec_id)} - {genotype}({threshold})", fontsize=12)
                        ax[i].text(0.5, 0.5, f"Not enough trials ({n_trials})", ha='center', va='center', fontsize=10)
                    print(
                        f"{rec_id}({genotype}) → Not enough {behavior_label} trials ({n_trials}) to compute cosine similarity")
                continue
        rows.append(row)
    data = pd.DataFrame(rows)
    # data = data.dropna(axis=0, how='any')
    # Plotting the figures with the individual matrices
    n_used = len(mean_activity_df.ID.unique())
    for fig_name, fig in figs.items():
        # Setting off the unused axes
        for extra_ax in axes[fig_name][n_used:]:
            extra_ax.set_axis_off()
        # # Adding a color bar
        # axs_flat = axes[fig_name]
        # for ax in axs_flat[:n_used]:
        #     mappable = ax.images[0]
        # fig.colorbar(mappable, ax=axs_flat[:n_used], orientation='vertical', fraction=0.02,pad=0.01)
        fig.suptitle(f"Cosine similarity {centroid_str}of {fig_name} between pairs of trials"
                     f"\nWT: {amps["WT"]} \nKO-Hypo: {amps["KO-Hypo"]} \nKO: {amps["KO"]}", fontsize=14)
        fig.canvas.manager.set_window_title(f"Cos_sim_mat_{centroid_str}{fig_name}")
    data["Resp_Delta"] = data["Resp_Hit"] - data["Resp_Miss"]
    data["Stim_mean_Delta"] = data["Stim_mean_Hit"] - data["Stim_mean_Miss"]
    # Plotting the comparisons
    fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(36, 32), constrained_layout=True)
    colors = {"Miss": [ppt.wt_light_color, ppt.hypo_light_color], "Hit": [ppt.wt_color, ppt.hypo_color],
              "Delta": [ppt.wt_color, ppt.hypo_color], "Nogo": [ppt.wt_light_color, ppt.hypo_light_color],
              "WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color]}
    for row, metric in enumerate(["Resp", "Stim_mean"]):
        # Plotting the difference between genotypes
        for col, behavior in enumerate(["Miss", "Hit", "Delta", "Nogo"]):
            param = f"{metric}_{behavior}"
            ppt.boxplot(ax[row, col], data[data["Genotype"] == "WT"][param].values,
                        data[data["Genotype"] == "KO-Hypo"][param].values,
                        ylabel="Mean cosine similarity", paired=False, title=param, ylim=[], colors=colors[behavior],
                        det_marker=False,
                        force_markers_identity=False)
        # Plotting the difference within genotypes
        for col, genotype in enumerate(["WT", "KO-Hypo"]):
            gp_data = data[data["Genotype"] == genotype]
            ppt.boxplot(ax[row, 4 + col], gp_data[f"{metric}_Hit"].values, gp_data[f"{metric}_Miss"].values,
                        ylabel="Mean cosine similarity", paired=True, title=f"Hit/Miss ({genotype})", ylim=[],
                        colors=colors[genotype],
                        det_marker=True, force_markers_identity=False)
            ppt.boxplot(ax[2 + row, 2 * col], gp_data[f"{metric}_Hit"].values, gp_data[f"{metric}_Nogo"].values,
                        ylabel="Mean cosine similarity", paired=True, title=f"{metric}→ Hit/Nogo ({genotype})", ylim=[],
                        colors=colors[genotype],
                        det_marker=True, force_markers_identity=False)
            ppt.boxplot(ax[2 + row, 2 * col + 1], gp_data[f"{metric}_Miss"].values, gp_data[f"{metric}_Nogo"].values,
                        ylabel="Mean cosine similarity", paired=True, title=f"{metric}→ Miss/Nogo ({genotype})",
                        ylim=[], colors=colors[genotype],
                        det_marker=True, force_markers_identity=False)
            ppt.boxplot(ax[2 + row, 4 + col], gp_data[f"{metric}_Hit"].values, gp_data[f"{metric}_12_Hit"].values,
                        ylabel="Mean cosine similarity", paired=True, title=f"{metric}→ Hit/Hit(12µm) ({genotype})",
                        ylim=[], colors=colors[genotype],
                        det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of the mean cosine similarity between pairs of trials {centroid_str}"
                 f"\nWT: {amps["WT"]} \nKO-Hypo: {amps["KO-Hypo"]} \nKO: {amps["KO"]}"
                 f"\n[GABAergic interneurons included = {include_gaba}]"
                 f"\n{title_precision}", fontsize=14)
    fig.canvas.manager.set_window_title(f"Cosine_sim_comp_gaba={include_gaba}_{title_precision}")
    # Plotting the correlation
    fig_corr, ax_corr = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), constrained_layout=True)
    colors_corr = {"WT": ppt.wt_color, "KO-Hypo": ppt.hypo_color}
    for i, x_metric in enumerate(["Prestim_mean_Hit", "Hit Rate"]):
        for y_legend, geno in zip([0.95, 0.85], ["WT", "KO-Hypo"]):
            x = data[data["Genotype"] == geno][x_metric].values
            y = data[data["Genotype"] == geno]["Stim_mean_Hit"].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            results = dict(linregress(x, y)._asdict())
            r2 = results["rvalue"] ** 2
            line = results["slope"] * x + results["intercept"]
            # Plot the data points and regression line
            ax_corr[i].scatter(x, y, color=colors_corr[geno], alpha=0.7, s=10, marker="+")
            ax_corr[i].plot(x, line, color=colors_corr[geno], lw=2)
            ax_corr[i].text(0.05, y_legend, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}",
                            transform=ax_corr[i].transAxes,
                            fontsize=8, verticalalignment="top", color=colors_corr[geno])
        ax_corr[i].set_xlabel(x_metric, fontsize=12)
        ax_corr[i].set_ylabel("Stim cosine stim", fontsize=12)
    fig_corr.suptitle(f"Cosine similarity between pre-stimulus and stimulus for hit trials {centroid_str}"
                      f"\n[GABAergic interneurons included = {include_gaba}]"
                      f"\n{title_precision}", fontsize=14)
    fig_corr.canvas.manager.set_window_title(f"Corr_cosine_sim{centroid_str}_gaba={include_gaba}_{title_precision}")
    plt.show()
    return data


def compare_global_cosine_sim(mean_activity_df, include_gaba=False, centroid=False):
    names = ["Resp_All", "Resp_Hit", "Resp_Miss", "Stim_mean_All", "Stim_mean_Hit", "Stim_mean_Miss"]
    figs = {}
    axes = {}
    for name in names:
        fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(24, 16), constrained_layout=True)
        figs[name] = fig
        axes[name] = axs.flatten()
    for i, rec_id in enumerate(mean_activity_df.ID.unique()):
        # Retrieving the data for the recording
        rec_data = mean_activity_df[mean_activity_df["ID"] == rec_id]
        if not include_gaba:
            rec_data = rec_data[rec_data["Neuron"].str.startswith("EXC")]
        genotype = rec_data.Genotype.values[0]
        threshold = rec_data.Threshold.values[0]
        # Splitting hit and miss data
        miss_data = rec_data[rec_data["Behavior"] == False]
        hit_data = rec_data[rec_data["Behavior"] == True]
        row = {"ID": rec_id, "Genotype": genotype}
        for behavior_label, behavior_data in zip(["All", "Miss", "Hit"], [rec_data, miss_data, hit_data]):
            n_trials = len(behavior_data.Trial.unique())
            for metric in ["Resp", "Stim_mean", "Prestim_mean"]:
                matrix = np.array(behavior_data.pivot(index="Trial", columns="Neuron", values=metric))

                # ========== Cosine similarity computation and grouping ==========
                global_cos_metric = compute_mean_cos_sim(matrix, centroid=centroid)

                row[f"{metric}_{behavior_label}"] = global_cos_metric
                if metric != "Prestim_mean":
                    # Retrieving the axes corresponding to the metrics to plot and plotting the rec matrix
                    ax = axes[f"{metric}_{behavior_label}"]
                    ax[i].imshow(cosine_similarity(matrix), cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
                    ax[i].set_xlabel("Trial i", fontsize=10)
                    ax[i].set_ylabel("Trial j", fontsize=10)
                    ax[i].set_title(f"{int(rec_id)} - {genotype}({threshold})[{global_cos_metric:.2f}]", fontsize=12)
        rows.append(row)
    data = pd.DataFrame(rows)
    # Setting off the unused axes
    n_used = len(mean_activity_df.ID.unique())
    for fig_name, fig in figs.items():
        for extra_ax in axes[fig_name][n_used:]:
            extra_ax.set_axis_off()
        fig.suptitle(f"Cosine similarity of {fig_name} between pairs of trials (all amplitudes)\n[centroid={centroid}]",
                     fontsize=14)
        fig.canvas.manager.set_window_title(f"Cos_sim_mat_{centroid}{fig_name}")
    # Plotting the comparisons
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 16), constrained_layout=True)
    colors = {"Miss": [ppt.wt_light_color, ppt.hypo_light_color], "Hit": [ppt.wt_color, ppt.hypo_color],
              "All": [ppt.wt_color, ppt.hypo_color],
              "WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color]}
    for row, metric in enumerate(["Resp", "Stim_mean"]):
        # Plotting the difference between genotypes
        for col, behavior in enumerate(["Miss", "Hit", "All"]):
            param = f"{metric}_{behavior}"
            ppt.boxplot(ax[row, col], data[data["Genotype"] == "WT"][param].values,
                        data[data["Genotype"] == "KO-Hypo"][param].values,
                        ylabel="Mean cosine similarity", paired=False, title=param, ylim=[],
                        colors=colors[behavior], det_marker=False,
                        force_markers_identity=False)
        # Plotting the difference within genotypes
        for col, genotype in enumerate(["WT", "KO-Hypo"]):
            gp_data = data[data["Genotype"] == genotype]
            ppt.boxplot(ax[row, 3 + col], gp_data[f"{metric}_Hit"].values, gp_data[f"{metric}_Miss"].values,
                        ylabel="Mean cosine similarity", paired=True, title=f"Hit/Miss ({genotype})", ylim=[],
                        colors=colors[genotype],
                        det_marker=True, force_markers_identity=False)
    fig.suptitle(f"Comparison of the mean cosine similarity between pairs of trials (centroid={centroid})"
                 f"\n[GABAergic interneurons included = {include_gaba}]", fontsize=14)
    fig.canvas.manager.set_window_title(f"Cosine_sim_comp_gaba={include_gaba}")
    plt.show()
    return data


def compute_mean_cos_sim(matrix, centroid=False):
    if centroid:
        # Testing another global similarity metric: mean distance of each trial to the centroid trial
        centroid = matrix.mean(axis=0).reshape(1, -1)
        mean_centroid_cos_sim = cosine_similarity(matrix, centroid).mean()
        global_cos_metric = mean_centroid_cos_sim
    else:
        # Computing the cosine similarity matrix between pairs of trials
        n_trials = matrix.shape[0]
        cos_sim_mat = cosine_similarity(matrix)
        mean_cos_sim = cos_sim_mat[~np.eye(n_trials, dtype=bool)].mean()
        global_cos_metric = mean_cos_sim
    return global_cos_metric


def correlate_tbt_var_behavior(mean_activity_df, include_gaba=False):
    """Is more variability between the trials associated with more variability in the behavioral response ?
    Correlation for each mouse of the cosine similarity between trials of each amplitude with the consistency of
    behavioral response for this amplitude. Then global correlation for the genotype for each amplitude
    (relatively to the threshold)"""
    # Building a Dataframe with the cosine similarity between trials for each amplitude and each mouse
    rows = []
    for rec_id in mean_activity_df.ID.unique():
        rec_data = mean_activity_df[mean_activity_df["ID"] == rec_id]
        genotype = rec_data.Genotype.values[0]
        threshold = rec_data.Threshold.values[0]
        # Filtering out the GABAergic interneurons according to the optional parameter
        if not include_gaba:
            rec_data = rec_data[rec_data["Neuron"].str.startswith("EXC")]
        for amp in range(2, 13, 2):
            row = {"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "Amplitude": amp}
            amp_data = rec_data[rec_data["Amplitude"] == amp].sort_values(by=["Trial", "Neuron"]).copy()
            n_trials = len(amp_data.Trial.unique())
            # Computing the similarity between trials at the neuronal level
            for metric in ["Resp", "Stim_mean"]:
                neural_matrix = np.array(amp_data.pivot(index="Trial", columns="Neuron", values=metric))
                # Computing the cosine similarity matrix between pairs of trials
                cos_sim_mat = cosine_similarity(neural_matrix)
                mean_sim = cos_sim_mat[~np.eye(n_trials, dtype=bool)].mean()
                row[metric] = mean_sim
            # Computing the similarity between trials at the behavioral level
            behavior_vector = (amp_data.groupby("Trial", as_index=False).first())["Behavior"].values
            behavior_mat = np.where(np.equal.outer(behavior_vector, behavior_vector), 1, -1)
            mean_behavior_sim = behavior_mat[~np.eye(n_trials, dtype=bool)].mean()
            row["Behavior"] = mean_behavior_sim
            rows.append(row)
    data = pd.DataFrame(rows)
    # Plotting each mouse's correlation
    color_dict = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color}
    cmap = plt.get_cmap("plasma")
    mouse_resp_fig, r_axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 20), constrained_layout=True)
    mouse_mean_fig, m_axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 20), constrained_layout=True)
    for fig, axes, metric in zip([mouse_resp_fig, mouse_mean_fig], [r_axes, m_axes], ["Resp", "Stim_mean"]):
        ax = axes.flatten()
        for ax_id, rec_id in enumerate(data.ID.unique()):
            rec_data = data[data["ID"] == rec_id]
            geno = rec_data["Genotype"].values[0]
            thre = rec_data["Threshold"].values[0]
            x = rec_data[metric]
            y = rec_data["Behavior"]
            results = dict(linregress(x, y)._asdict())
            r2 = results["rvalue"] ** 2
            line = results["slope"] * x + results["intercept"]
            # Plot the data points and regression line
            ax[ax_id].plot(x, line, color=color_dict[geno], lw=2)
            ax[ax_id].scatter(x, y, color=cmap(np.linspace(0, 1, len(x))), alpha=0.7, s=10, marker="+")
            ax[ax_id].text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}",
                           transform=ax[ax_id].transAxes, fontsize=8,
                           verticalalignment="top", color="black")
            ax[ax_id].set_title(f"{int(rec_id)}[{geno}] - {thre}", color=color_dict[geno])
            ax[ax_id].set_xlabel(metric, fontsize=10)
            ax[ax_id].set_ylabel("Behavioral similarity of trials", fontsize=10)
            ax[ax_id].set_xlim([-1, 1.05])
            ax[ax_id].set_ylim([-1, 1.05])
            ax[ax_id].spines['top'].set_visible(False)
            ax[ax_id].spines['right'].set_visible(False)
            ax[ax_id].axvline(x=0, linestyle=":", linewidth=0.5, color="gray")
            ax[ax_id].axhline(y=0, linestyle=":", linewidth=0.5, color="gray")
        fig.suptitle(f"Correlation per mouse of {metric} cosine similarity between trials of each amplitude "
                     f"and corresponding similarity of behavioral outcome", fontsize=12)
        fig.canvas.manager.set_window_title(f"Corr_{metric}_cosim_behavior")
    # Plotting the mean genotype correlation between neural tbt var and behavioral similarity per amplitude
    grouped_data = data.groupby(["Genotype", "Amplitude"], as_index=False).mean()
    genotype_resp_fig, gr_axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), constrained_layout=True)
    genotype_mean_fig, gm_axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), constrained_layout=True)
    for fig, axes, metric in zip([genotype_resp_fig, genotype_mean_fig], [gr_axes, gm_axes], ["Resp", "Stim_mean"]):
        ax = axes.flatten()
        for ax_id, geno in enumerate(grouped_data.Genotype.unique()):
            geno_data = grouped_data[grouped_data["Genotype"] == geno]
            x = geno_data[metric]
            y = geno_data["Behavior"]
            results = dict(linregress(x, y)._asdict())
            r2 = results["rvalue"] ** 2
            line = results["slope"] * x + results["intercept"]
            # Plot the data points and regression line
            ax[ax_id].plot(x, line, color=color_dict[geno], lw=2)
            ax[ax_id].scatter(x, y, color=cmap(np.linspace(0, 1, len(x))), alpha=0.7, s=10, marker="+")
            ax[ax_id].text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}",
                           transform=ax[ax_id].transAxes, fontsize=8,
                           verticalalignment="top", color="black")
            ax[ax_id].set_title(geno, color=color_dict[geno])
            ax[ax_id].set_xlabel(metric, fontsize=10)
            ax[ax_id].set_ylabel("Behavioral similarity of trials", fontsize=10)
            ax[ax_id].set_xlim([-1, 1.05])
            ax[ax_id].set_ylim([-1, 1.05])
            ax[ax_id].spines['top'].set_visible(False)
            ax[ax_id].spines['right'].set_visible(False)
            ax[ax_id].axvline(x=0, linestyle=":", linewidth=0.5, color="gray")
            ax[ax_id].axhline(y=0, linestyle=":", linewidth=0.5, color="gray")
        fig.suptitle(f"Correlation per mouse of mean {metric} cosine similarity between trials per amplitude "
                     f"and corresponding mean similarity of behavioral outcome", fontsize=12)
        fig.canvas.manager.set_window_title(f"Corr_mean_{metric}_cosim_behavior")
    # Plotting each genotype correlation for different amplitude
    plt.show()
    return data


def compare_nb_trials_wt_ko(mean_activity_df):
    """Compares the number of threshold trials for both behavioral outcome between 2 genotypes"""
    data = mean_activity_df[mean_activity_df["Amplitude"] == mean_activity_df["Threshold"]].groupby(
        ["Genotype", "ID", "Behavior", "Neuron"], as_index=False).size()
    data = data.drop(columns="Neuron").groupby(["Genotype", "ID", "Behavior"], as_index=False).mean()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 16), constrained_layout=True)
    colors = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color]}
    for col, label in enumerate([True, False]):
        ppt.boxplot(ax[0, col], data[(data["Genotype"] == "WT") & (data["Behavior"] == label)]["size"].values,
                    data[(data["Genotype"] == "KO-Hypo") & (data["Behavior"] == label)]["size"].values,
                    ylabel="Nb of trials", paired=False, title=f"Det = {label}", ylim=[],
                    colors=[ppt.wt_color, ppt.hypo_color],
                    det_marker=False, force_markers_identity=False)
    for col, genotype in enumerate(["WT", "KO-Hypo"]):
        ppt.boxplot(ax[1, col], data[(data["Genotype"] == genotype) & (data["Behavior"] == True)]["size"].values,
                    data[(data["Genotype"] == genotype) & (data["Behavior"] == False)]["size"].values,
                    ylabel="Nb of trials", paired=True, title=genotype, ylim=[], colors=colors[genotype],
                    det_marker=True, force_markers_identity=False)
    fig.suptitle("Comparison of the number of threshold trials between genotype according to behavior outcome",
                 fontsize=12)
    fig.canvas.manager.set_window_title("Comp_nb_threshold_trials_genotypes")
    plt.show()
    return data


def plot_venn_upset(recs, amp="threshold", behavior_filter=None, pattern="responsive", n_type="EXC"):
    pattern_dict = {"activated": [1], "inhibited": [-1], "responsive": [-1, 1]}
    for rec in recs:
        if amp == "threshold":
            amp_list = [rec.session_threshold]
        elif isinstance(amp, list):
            amp_list = amp
        if behavior_filter is not None:
            trial_mask = (rec.stim_ampl.isin(amp_list) & (rec.detected_stim == behavior_filter))
        else:
            trial_mask = rec.stim_ampl.isin(amp_list)
        # Filtering the repsonsivity
        resp = rec.matrices[n_type]["Responsivity"][:, trial_mask]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15), constrained_layout=True)


def compute_neuronal_sensitivity(data_df, amplitude="All", n_type="all", filter_out_nr=False, gp1="WT", gp2="KO-Hypo"):
    """"""
    data = data_df.copy()
    # For each recording
    rows = []
    for rec_id in data["ID"].unique():
        rec_full_data = data[data["ID"] == rec_id]
        genotype = rec_full_data["Genotype"].values[0]
        threshold = rec_full_data["Threshold"].values[0]
        rec_data = rec_full_data[rec_full_data["Amplitude"] != 0].drop(columns=["FA"])
        # Filtering out non-responsive neurons
        if filter_out_nr:
            rec_data = rec_data.groupby("Neuron").filter(lambda grp: (grp["Resp"] != 0).any()).reset_index(drop=True)
        rec_data = rec_data.drop(columns=["Resp"])
        rec_full_data = rec_full_data.drop(columns=["Resp"])
        # Taking all amplitudes together
        all_hit_data = rec_data[rec_data["Behavior"] == True].drop(columns=["ID", "Genotype", "Threshold", "Behavior"]).copy()
        all_miss_data = rec_data[rec_data["Behavior"] == False].drop(columns=["ID", "Genotype", "Threshold", "Behavior"]).copy()
        grouped_hit_data = all_hit_data.groupby(["Neuron"], as_index=True).mean().drop(columns=["Trial", "Amplitude"])
        grouped_miss_data = all_miss_data.groupby(["Neuron"], as_index=True).mean().drop(columns=["Trial", "Amplitude"])
        # Sorting and counting the neurons
        sorted_idx = sorted(grouped_hit_data.index,key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
        grouped_hit_data = grouped_hit_data.loc[sorted_idx]
        grouped_miss_data = grouped_miss_data.loc[sorted_idx]
        n_exc = sum(1 for nid in sorted_idx if nid.split("_")[0] == "EXC")
        n_inh = sum(1 for nid in sorted_idx if nid.split("_")[0] == "INH")
        # Computing global sensitivity
        grouped_sensitivity_data = grouped_hit_data.subtract(grouped_miss_data)["Stim_mean"].values if ((len(grouped_hit_data) > 0) and (len(grouped_miss_data) > 0)) else np.nan
        rows.append({"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "n_EXC": n_exc, "n_INH": n_inh,
                     "Amplitude": "All", "Sensitivity": grouped_sensitivity_data})
        # Computing the sensitivity for no-go trials (activity FA - CR)
        fa_data = rec_full_data[(rec_full_data["Amplitude"] == 0) & (rec_full_data["FA"] == True)].drop(columns=["ID", "Genotype", "Threshold", "Behavior", "Amplitude"]).copy()
        cr_data = rec_full_data[(rec_full_data["Amplitude"] == 0) & (rec_full_data["FA"] == False)].drop(columns=["ID", "Genotype", "Threshold", "Behavior", "Amplitude"]).copy()
        grouped_fa_data = fa_data.groupby(["Neuron"], as_index=True).mean().drop(columns=["Trial"])
        grouped_cr_data = cr_data.groupby(["Neuron"], as_index=True).mean().drop(columns=["Trial"])
        grouped_fa_data = grouped_fa_data.reindex(sorted_idx)
        grouped_cr_data = grouped_cr_data.reindex(sorted_idx)
        nogo_sensitivity_data = grouped_fa_data.subtract(grouped_cr_data)["Stim_mean"].values if ((len(grouped_fa_data) > 0) and (len(grouped_cr_data) > 0)) else np.nan
        rows.append({"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "n_EXC": n_exc, "n_INH": n_inh,
                     "Amplitude": 0, "Sensitivity": nogo_sensitivity_data})
        # Each amplitude individually
        for amp in sorted(rec_data["Amplitude"].unique()):
            amp_hit_data = all_hit_data[all_hit_data["Amplitude"] == amp].groupby(["Neuron"], as_index=True).mean().drop(columns=["Trial", "Amplitude"])
            amp_miss_data = all_miss_data[all_miss_data["Amplitude"] == amp].groupby(["Neuron"], as_index=True).mean().drop(columns=["Trial", "Amplitude"])
            amp_hit_data = amp_hit_data.reindex(sorted_idx)
            amp_miss_data = amp_miss_data.reindex(sorted_idx)
            sensitivity_data = amp_hit_data.subtract(amp_miss_data)["Stim_mean"].values if ((len(amp_hit_data) > 0) and (len(amp_miss_data) > 0)) else np.nan
            rows.append({"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "n_EXC": n_exc, "n_INH": n_inh,
                         "Amplitude": amp, "Sensitivity": sensitivity_data})
    data = pd.DataFrame(rows)
    # Splitting the sensitivity of EXC and INH neurons
    data["Sensitivity_exc"] = data.apply(lambda row: row["Sensitivity"][: row["n_EXC"]], axis=1)
    data["Sensitivity_inh"] = data.apply(lambda row: row["Sensitivity"][row["n_EXC"]:], axis=1)
    # Computing the mean sensitivity per mouse/amplitude combination
    data["Mean_sensitivity"] = data["Sensitivity"].apply(lambda lst: np.mean(lst))
    data["Mean_sensitivity_exc"] = data["Sensitivity_exc"].apply(lambda lst: np.mean(lst))
    data["Mean_sensitivity_inh"] = data["Sensitivity_inh"].apply(lambda lst: np.mean(lst))
    # Filtering the sensitivity vector to keep only the d'>1
    data["Sensitive_neurons"] = data["Sensitivity"].apply(lambda lst: [v for v in lst if v > 1])
    data["Sensitive_neurons_exc"] = data["Sensitivity_exc"].apply(lambda lst: [v for v in lst if v > 1])
    data["Sensitive_neurons_inh"] = data["Sensitivity_inh"].apply(lambda lst: [v for v in lst if v > 1])
    # Computing the mean sensitivity of sensitive neurons
    data["Mean_sens_neurons"] = data["Sensitive_neurons"].apply(lambda lst: np.mean(lst))
    data["Mean_sens_neurons_exc"] = data["Sensitive_neurons_exc"].apply(lambda lst: np.mean(lst))
    data["Mean_sens_neurons_inh"] = data["Sensitive_neurons_inh"].apply(lambda lst: np.mean(lst))
    # Computing the number of sensitive neurons
    data["Nb_Sens_neurons"] = data["Sensitivity"].apply(lambda lst: np.nan if any(pd.isna(v) for v in lst) else sum(v > 1 for v in lst))
    data["Nb_Sens_neurons_exc"] = data["Sensitivity_exc"].apply(lambda lst: np.nan if any(pd.isna(v) for v in lst) else sum(v > 1 for v in lst))
    data["Nb_Sens_neurons_inh"] = data["Sensitivity_inh"].apply(lambda lst: np.nan if any(pd.isna(v) for v in lst) else sum(v > 1 for v in lst))
    # Computing the percentage of sensitive neurons
    data["Perc_Sens_neurons"] = data["Nb_Sens_neurons"] / (data["n_EXC"] + data["n_INH"]) * 100
    data["Perc_Sens_neurons_exc"] = data["Nb_Sens_neurons_exc"] / data["n_EXC"] * 100
    data["Perc_Sens_neurons_inh"] = data["Nb_Sens_neurons_inh"] / data["n_INH"] * 100
    # Defining the column to take into consideration according to the neuron type chosen
    if n_type == "all":
        sens_col = "Sensitivity"
        mean_sens_col = "Mean_sensitivity"
        nb_sens_col = "Nb_Sens_neurons"
        perc_sens_col = "Perc_Sens_neurons"
        d_sens_col = "Sensitive_neurons"
        d_mean_sens_col = "Mean_sens_neurons"
    elif n_type == "EXC":
        sens_col = "Sensitivity_exc"
        mean_sens_col = "Mean_sensitivity_exc"
        nb_sens_col = "Nb_Sens_neurons_exc"
        perc_sens_col = "Perc_Sens_neurons_exc"
        d_sens_col = "Sensitive_neurons_exc"
        d_mean_sens_col = "Mean_sens_neurons_exc"
    elif n_type == "INH":
        sens_col = "Sensitivity_inh"
        mean_sens_col = "Mean_sensitivity_inh"
        nb_sens_col = "Nb_Sens_neurons_inh"
        perc_sens_col = "Perc_Sens_neurons_inh"
        d_sens_col = "Sensitive_neurons_inh"
        d_mean_sens_col = "Mean_sens_neurons_inh"
    # Plotting the stem plots of neuronal sensitivity for each mouse
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(20, 15), constrained_layout=True)
    ax = axs.flatten()
    for ax_id, rec_id in enumerate(data["ID"].unique()):
        if amplitude == "All":
            vector = data[(data["ID"] == rec_id) & (data["Amplitude"] == "All")]["Sensitivity"].values[0]
        elif amplitude == "Threshold":
            vector = data[(data["ID"] == rec_id) & (data["Amplitude"] == data["Threshold"])]["Sensitivity"].values[0]
        elif isinstance(amplitude, int):
            vector = data[(data["ID"] == rec_id) & (data["Amplitude"] == amplitude)]["Sensitivity"].values[0]
        geno = data[data["ID"] == rec_id]["Genotype"].values[0]
        thre = data[data["ID"] == rec_id]["Threshold"].values[0]
        # Splitting the EXC and INH neurons
        x = np.arange(len(vector))
        split_idx = data[data["ID"] == rec_id]["n_EXC"].values[0]
        x_exc = x[:split_idx]
        y_exc = vector[:split_idx]
        x_inh = x[split_idx:]
        y_inh = vector[split_idx:]
        # 1) Plot all EXC stems in default color (blue)
        markerline_exc, stemlines_exc, baseline_exc = ax[ax_id].stem(x_exc, y_exc)
        markerline_exc.set_markersize(3)
        stemlines_exc.set_linewidth(1)
        baseline_exc.set_linewidth(2)
        # 2) Over‐plot the INH stems in orange
        if len(x_inh) > 0:
            markerline_inh, stemlines_inh, baseline_inh = ax[ax_id].stem(x_inh, y_inh, linefmt="orange", markerfmt="orange")
            markerline_inh.set_markersize(3)
            stemlines_inh.set_linewidth(1)
            baseline_inh.set_linewidth(2)
        ax[ax_id].set_xlabel("Neuron ID", fontsize=10)
        ax[ax_id].set_ylabel("Sensitivity d'", fontsize=10)
        ax[ax_id].set_title(f"{rec_id} - {geno} ({thre})", fontsize=12)
        ax[ax_id].tick_params(axis="both", labelsize=8)
        ax[ax_id].axhline(y=1, lw=1, ls=":", color="green")
    fig.suptitle(f"Sensitivity of neurons for {amplitude} amplitudes", fontsize=14)
    fig.canvas.manager.set_window_title(f"Sensitivity_{amplitude}")
    # Plotting the curveplot of mean d' per mouse as a function of the amplitude
    curve_plot_data = data[~data["Amplitude"].isin(["All", 0])].copy()
    curve_fig, curve_ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 16), constrained_layout=True)
    sorted_groups = sorted([gp1, gp2])
    exc_test, exc_posthoc = ppt.curveplot(curve_ax[0], curve_plot_data[curve_plot_data["Genotype"].isin([gp1, gp2])], between="Genotype", within="Amplitude",
                                          variable="Mean_sensitivity_exc", data_points=True,
                                          title=f"Variation of the mean sensitivity per animal for EXC neurons across amplitudes",
                                          ylabel=None, xlabel=None, ylim=[-0.75, 1], colors=[color_dict[sorted_groups[0]][0], color_dict[sorted_groups[1]][0]],
                                          id_display=False, legend_display=True,
                                          qq_show=True, transformation=None, consider_normality=True,
                                          consider_homogeneity=True)
    inh_test, inh_posthoc = ppt.curveplot(curve_ax[1], curve_plot_data[curve_plot_data["Genotype"].isin([gp1, gp2])], between="Genotype", within="Amplitude",
                                          variable="Mean_sensitivity_inh", data_points=True,
                                          title=f"Variation of the mean sensitivity per animal for INH neurons across amplitudes",
                                          ylabel=None, xlabel=None, ylim=[-0.75, 1], colors=[color_dict[sorted_groups[0]][0], color_dict[sorted_groups[1]][0]],
                                          id_display=False, legend_display=True,
                                          qq_show=True, transformation=None, consider_normality=True,
                                          consider_homogeneity=True)
    curve_fig.canvas.manager.set_window_title(f"Curv_Sensitivity")
    # Plotting the number of sensitive neurons (d' > 1) between both genotypes
    fig_stats, axs_stats = plt.subplots(nrows=2, ncols=5, figsize=(30, 16), constrained_layout=True)
    ax_stats = axs_stats.flatten()
    # Plotting the percentage of sensitive neurons (d' > 1) between both genotypes
    fig_perc, axs_perc = plt.subplots(nrows=2, ncols=5, figsize=(30, 16), constrained_layout=True)
    ax_perc = axs_perc.flatten()
    # Plotting the mean sensitivity per animal between both genotypes
    fig_mean, axs_mean = plt.subplots(nrows=2, ncols=5, figsize=(30, 16), constrained_layout=True)
    ax_mean = axs_mean.flatten()
    # Plotting the mean sensitivity of sensitive neurons per animal between both genotypes
    fig_d_mean, axs_d_mean = plt.subplots(nrows=2, ncols=5, figsize=(30, 16), constrained_layout=True)
    ax_d_mean = axs_d_mean.flatten()
    for i, amp in enumerate(data["Amplitude"].unique()):
        # Number
        ppt.boxplot(ax_stats[i], data[(data["Genotype"] == gp1) & (data["Amplitude"] == amp)][nb_sens_col].values,
                    data[(data["Genotype"] == gp2) & (data["Amplitude"] == amp)][nb_sens_col].values,
                    ylabel="Nb neurons with d'>1", paired=False, title=f"Amplitude = {amp}", ylim=[], colors=[color_dict[gp1][0], color_dict[gp2][0]],
                    det_marker=False, force_markers_identity=False)
        # Percentage
        ppt.boxplot(ax_perc[i], data[(data["Genotype"] == gp1) & (data["Amplitude"] == amp)][perc_sens_col].values,
                    data[(data["Genotype"] == gp2) & (data["Amplitude"] == amp)][perc_sens_col].values,
                    ylabel="% neurons with d'>1", paired=False, title=f"Amplitude = {amp}", ylim=[0, 60], colors=[color_dict[gp1][0], color_dict[gp2][0]],
                    det_marker=False, force_markers_identity=False)
        # Mean
        ppt.boxplot(ax_mean[i], data[(data["Genotype"] == gp1) & (data["Amplitude"] == amp)][mean_sens_col].values,
                    data[(data["Genotype"] == gp2) & (data["Amplitude"] == amp)][mean_sens_col].values,
                    ylabel="Mean d'", paired=False, title=f"Amplitude = {amp}", ylim=[-0.7, 0.7],
                    colors=[color_dict[gp1][0], color_dict[gp2][0]],
                    det_marker=False, force_markers_identity=False)
        # Mean - sensitive neurons
        ppt.boxplot(ax_d_mean[i], data[(data["Genotype"] == gp1) & (data["Amplitude"] == amp)][d_mean_sens_col].values,
                    data[(data["Genotype"] == gp2) & (data["Amplitude"] == amp)][d_mean_sens_col].values,
                    ylabel="Mean d' of sensitive neurons", paired=False, title=f"Amplitude = {amp}", ylim=[],
                    colors=[color_dict[gp1][0], color_dict[gp2][0]],
                    det_marker=False, force_markers_identity=False)
        # Global sensitivity
        # amp_data = data[data["Amplitude"] == amp].copy()
        # amp_long = amp_data.explode(d_sens_col).rename(columns={d_sens_col: "neuron_sens"})
        # amp_long["neuron_sens"] = pd.to_numeric(amp_long["neuron_sens"])
        # amp_long["Genotype"] = pd.Categorical(amp_long["Genotype"], categories=["WT", "KO", "KO-Hypo"])
        # amp_long = amp_long.dropna(subset=["neuron_sens"])
        # model = smf.mixedlm("neuron_sens ~ Genotype", amp_long, groups=amp_long["ID"])
        # result = model.fit()
        # print(f"=== === === Amplitude: {amp} === === ===")
        # print(result.summary())
    # Number - threshold
    ppt.boxplot(ax_stats[8], data[(data["Genotype"] == gp1) & (data["Amplitude"] == data["Threshold"])][nb_sens_col].values,
                data[(data["Genotype"] == gp2) & (data["Amplitude"] == data["Threshold"])][nb_sens_col].values,
                ylabel="Nb neurons with d'>1", paired=False, title=f"Amplitude = Threshold", ylim=[],
                colors=[color_dict[gp1][0], color_dict[gp2][0]],
                det_marker=False, force_markers_identity=False)
    # Mean - threshold
    ppt.boxplot(ax_mean[8],
                data[(data["Genotype"] == gp1) & (data["Amplitude"] == data["Threshold"])][mean_sens_col].values,
                data[(data["Genotype"] == gp2) & (data["Amplitude"] == data["Threshold"])][mean_sens_col].values,
                ylabel="Mean d'", paired=False, title=f"Amplitude = Threshold", ylim=[],
                colors=[color_dict[gp1][0], color_dict[gp2][0]],
                det_marker=False, force_markers_identity=False)
    # Mean sensitive neurons - threshold
    ppt.boxplot(ax_d_mean[8],
                data[(data["Genotype"] == gp1) & (data["Amplitude"] == data["Threshold"])][d_mean_sens_col].values,
                data[(data["Genotype"] == gp2) & (data["Amplitude"] == data["Threshold"])][d_mean_sens_col].values,
                ylabel="Mean d'", paired=False, title=f"Amplitude = Threshold", ylim=[],
                colors=[color_dict[gp1][0], color_dict[gp2][0]],
                det_marker=False, force_markers_identity=False)
    # # Global sensitivity - threshold
    # thre_data = data[data["Amplitude"] == data["Threshold"]].copy()
    # thre_long = thre_data.explode(sens_col).rename(columns={sens_col: "neuron_sens"})
    # thre_long["neuron_sens"] = pd.to_numeric(thre_long["neuron_sens"])
    # thre_long["Genotype"] = pd.Categorical(thre_long["Genotype"], categories=["WT", "KO", "KO-Hypo"])
    # thre_long = thre_long.dropna(subset=["neuron_sens"])
    # model_thre = smf.mixedlm("neuron_sens ~ Genotype", thre_long, groups=thre_long["ID"])
    # result_thre = model_thre.fit()
    # print(f"=== === === Amplitude: Threshold === === ===")
    # print(result_thre.summary())
    # # Global sensitivity - Including the amplitude in the model
    # global_data = data.copy()
    # global_data["Neuron_ID"] = global_data[sens_col].apply(lambda lst: list(range(len(lst))))
    # global_data = global_data.explode(["Neuron_ID", sens_col]).rename(columns={sens_col: "neuron_sens", "Neuron_ID": "Neuron"})
    # global_data["neuron_sens"] = pd.to_numeric(global_data["neuron_sens"], errors="coerce")
    # global_data["Genotype"] = pd.Categorical(global_data["Genotype"], categories=["WT", "KO", "KO-Hypo"])
    # global_data["Neuron"] = global_data["Neuron"].astype("category")
    # global_data = global_data.dropna(subset=["neuron_sens"])
    # # model_global = smf.mixedlm("neuron_sens ~ Genotype + Amplitude", global_data, groups=global_data["ID"], vc_formula={"Neuron": "0 + C(Neuron)"})
    # model_global = smf.mixedlm("neuron_sens ~ Genotype + Amplitude", global_data, groups=global_data["ID"])
    # result_global = model_global.fit()
    # print(f"=== === === MixedLM including amplitude === === ===")
    # print(result_global.summary())

    fig_stats.suptitle(f"Comparison between genotypes of the number of sensitive {n_type} neurons for different amplitudes\n filter_out_NR={filter_out_nr}", fontsize=14)
    fig_stats.canvas.manager.set_window_title(f"Sensitivity_comp")
    fig_perc.suptitle(f"Comparison between genotypes of the percentage of sensitive {n_type} neurons for different amplitudes\n filter_out_NR={filter_out_nr}", fontsize=14)
    fig_perc.canvas.manager.set_window_title(f"Sensitivity_perc_comp")
    fig_mean.suptitle(f"Comparison mean {n_type} sensitivity per mouse for different amplitudes", fontsize=14)
    fig_mean.canvas.manager.set_window_title(f"Mean_sens_comp")
    fig_d_mean.suptitle(f"Comparison mean {n_type} sensitivity of sensitive neurons per mouse for different amplitudes", fontsize=14)
    fig_d_mean.canvas.manager.set_window_title(f"Mean_sens_d1_comp")
    plt.show()
    return data


def ntn_cosine_similarity(mean_activity_df, amplitude="threshold", nogo=False, metric="Resp", filter_out_nr=False, n_type="all", gp1="WT", gp2="KO-Hypo"):
    """Computing and plotting each vector of neuronal activity across trials cosine similarity."""
    filtered_data = mean_activity_df[mean_activity_df["Genotype"].isin([gp1, gp2])]
    nogo_data = filtered_data[filtered_data.Amplitude == 0].copy()
    # Filtering out the no-gos according to the specified parameter
    if not nogo:
        data = filtered_data[filtered_data.Amplitude != 0].copy()
    else:
        data = filtered_data.copy()
    # Filtering the neuron type
    if n_type in ["EXC", "INH"]:
        data = data[data["Neuron"].str.startswith(n_type)]
        nogo_data = nogo_data[nogo_data["Neuron"].str.startswith(n_type)]
    # Selection of trials, amplitude
    if amplitude == "threshold":
        data = data[data.Amplitude == data.Threshold].copy()
    elif amplitude == "all":
        data = data[data.Amplitude != 0].copy()
    elif isinstance(amplitude, list):
        data = data[data.Amplitude.isin(amplitude)].copy()
    rows = []
    hit = data[data.Behavior == True].copy()
    miss = data[data.Behavior == False].copy()
    all = data.copy()
    for data, label in zip([nogo_data, hit, miss, all], ["Nogo", "Hit", "Miss", "All"]):
        # For each recording, computing the cosine similarity between its neurons
        fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(24, 16), constrained_layout=True)
        ax = axs.flatten()
        for i, rec_id in enumerate(data.ID.unique()):
            rec_data_long = data[data.ID == rec_id].copy()
            genotype = rec_data_long.Genotype.values[0]
            threshold = rec_data_long.Threshold.values[0]
            rec_data = rec_data_long.pivot(index="Neuron", columns="Trial", values=metric)
            # Filtering out the neurons that are never responsive
            if filter_out_nr:
                rec_data = rec_data[~(rec_data == 0).all(axis=1)]
            rec_data = rec_data.loc[sorted(rec_data.index, key=lambda lab: (lab.split('_')[0], int(lab.split('_')[1])))]
            # computing cosine similarity between pairs of neurons
            matrix = np.array(rec_data)
            n_neurons = matrix.shape[0]
            n_trials = matrix.shape[1]
            if n_neurons >= 1:
                cos_sim_mat = cosine_similarity(matrix)
                abs_cos_sim = np.abs(cos_sim_mat)
                mean_cos_sim = abs_cos_sim[~np.eye(n_neurons, dtype=bool)].mean()
                # Plotting the cosine similarity matrix
                ax[i].imshow(abs_cos_sim, cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
                ax[i].set_xlabel("Neuron i", fontsize=10)
                ax[i].set_ylabel("Neuron j", fontsize=10)
                ax[i].set_title(f"{rec_id} - {genotype}({threshold})[{mean_cos_sim:.2f}]", fontsize=12)
            else:
                mean_cos_sim = np.nan
            rows.append({"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "Behavior": label,
                         "n_neurons": n_neurons, "n_trials": n_trials, "mean_cos_sim": mean_cos_sim})
        fig.suptitle(f"Neuron to neuron {metric} cosine similarity between trials\n"
                     f"Amp: {amplitude} - Trials: {label} - No-Go: {nogo} - Neurons: {n_type}", fontsize=12)
        fig.canvas.manager.set_window_title(f"ntn_cosim_{amplitude}_{label}_{nogo}_{n_type}_{gp1}_{gp2}")
    mean_cos_sim_data = pd.DataFrame(rows)

    # Computing the SNR of neuronal correlation: ntn_corr all trials/ntn no-gos
    snr_df = mean_cos_sim_data.pivot(index=["ID", "Genotype"], columns="Behavior", values="mean_cos_sim").reset_index()
    snr_df["SNR"] = snr_df["All"] / snr_df["Nogo"]

    # # WT
    # model_wt = smf.mixedlm("mean_cos_sim ~ C(Behavior, Treatment(reference='Miss')) + n_neurons + n_trials",
    #                     mean_cos_sim_data[mean_cos_sim_data.Genotype == "WT"],
    #                     groups=mean_cos_sim_data[mean_cos_sim_data.Genotype == "WT"]["ID"])
    # result_wt = model_wt.fit(reml=False)
    # print(f"====== WT ======")
    # print(result_wt.summary())
    # # KO-Hypo
    # model_hypo = smf.mixedlm("mean_cos_sim ~ C(Behavior, Treatment(reference='Miss')) + n_neurons + n_trials",
    #                     mean_cos_sim_data[mean_cos_sim_data.Genotype == "KO-Hypo"],
    #                     groups=mean_cos_sim_data[mean_cos_sim_data.Genotype == "KO-Hypo"]["ID"])
    # result_hypo = model_hypo.fit(reml=False)
    # print(f"====== KO-Hypo ======")
    # print(result_hypo.summary())
    # # Hit
    # model_hit = smf.mixedlm("mean_cos_sim ~ C(Genotype, Treatment(reference='WT')) + n_neurons + n_trials",
    #                     mean_cos_sim_data[mean_cos_sim_data.Behavior == "Hit"],
    #                     groups=mean_cos_sim_data[mean_cos_sim_data.Behavior == "Hit"]["ID"])
    # result_hit = model_hit.fit(reml=False)
    # print(f"====== Hit ======")
    # print(result_hit.summary())
    # # Miss
    # model_miss = smf.mixedlm("mean_cos_sim ~ C(Genotype, Treatment(reference='WT')) + n_neurons + n_trials",
    #                     mean_cos_sim_data[mean_cos_sim_data.Behavior == "Miss"],
    #                     groups=mean_cos_sim_data[mean_cos_sim_data.Behavior == "Miss"]["ID"])
    # result_miss = model_miss.fit(reml=False)
    # print(f"====== Miss ======")
    # print(result_miss.summary())
    # # No-Go
    # model_miss = smf.mixedlm("mean_cos_sim ~ C(Genotype, Treatment(reference='WT')) + n_neurons + n_trials",
    #                     mean_cos_sim_data[mean_cos_sim_data.Behavior == "Nogo"],
    #                     groups=mean_cos_sim_data[mean_cos_sim_data.Behavior == "Nogo"]["ID"])
    # result_miss = model_miss.fit(reml=False)
    # print(f"====== No-Go ======")
    # print(result_miss.summary())
    # # All
    # model_all = smf.mixedlm("mean_cos_sim ~ C(Genotype, Treatment(reference='WT')) + n_neurons + n_trials",
    #                     mean_cos_sim_data[mean_cos_sim_data.Behavior == "All"],
    #                     groups=mean_cos_sim_data[mean_cos_sim_data.Behavior == "All"]["ID"])
    # result_all = model_all.fit(reml=False)
    # print(f"====== All ======")
    # print(result_all.summary())

    def resid_correct(group):
        # m = smf.ols('mean_cos_sim ~ n_neurons + n_trials', data=group).fit()
        m = smf.ols('mean_cos_sim ~ n_neurons', data=group).fit()
        return m.resid

    mean_cos_sim_data["cos_resid"] = (mean_cos_sim_data.groupby('Behavior').apply(resid_correct).droplevel(0))
    # Plotting the correlation of the number of neurons with the cosine similarity to control for the absence of bias
    corr_data_hit = mean_cos_sim_data[mean_cos_sim_data.Behavior == "Hit"].copy()
    corr_data_miss = mean_cos_sim_data[mean_cos_sim_data.Behavior == "Miss"].copy()
    corr_data_all = mean_cos_sim_data[mean_cos_sim_data.Behavior == "All"].copy()
    corr_data_nogo = mean_cos_sim_data[mean_cos_sim_data.Behavior == "Nogo"].copy()
    # Plotting the correlation of the mean cosine similarity with the nb of neurons and nb of trials
    for nb_variable in ["n_neurons", "n_trials"]:
        fig_corr, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), constrained_layout=True)
        for i, (label, corr_data) in enumerate(zip(["All", "Hit", "Miss", "Nogo"],
                                                [corr_data_all, corr_data_hit, corr_data_miss, corr_data_nogo])):
            # --- Correlation global
            y_col = corr_data["mean_cos_sim"]
            x_col = corr_data[nb_variable]
            results = dict(linregress(x_col, y_col)._asdict())
            r2 = results["rvalue"] ** 2
            line = results["slope"] * x_col + results["intercept"]
            # --- Correlation gp1
            y_col_gp1 = corr_data[corr_data.Genotype == gp1]["mean_cos_sim"]
            x_col_gp1 = corr_data[corr_data.Genotype == gp1][nb_variable]
            results_gp1 = dict(linregress(x_col_gp1, y_col_gp1)._asdict())
            r2_gp1 = results_gp1["rvalue"] ** 2
            line_gp1 = results_gp1["slope"] * x_col_gp1 + results_gp1["intercept"]
            # --- Correlation gp2
            y_col_gp2 = corr_data[corr_data.Genotype == gp2]["mean_cos_sim"]
            x_col_gp2 = corr_data[corr_data.Genotype == gp2][nb_variable]
            results_gp2 = dict(linregress(x_col_gp2, y_col_gp2)._asdict())
            r2_gp2 = results_gp2["rvalue"] ** 2
            line_gp2 = results_gp2["slope"] * x_col_gp2 + results_gp2["intercept"]
            # --- Plot the data points and regression lines
            ax[i].plot(x_col, line, color="black", lw=2)
            ax[i].plot(x_col_gp1, line_gp1, color=color_dict[gp1][0], lw=2)
            ax[i].plot(x_col_gp2, line_gp2, color=color_dict[gp2][0], lw=2)
            for g in sorted(corr_data["Genotype"].unique()):
                group = corr_data[corr_data["Genotype"] == g]
                sc = ax[i].scatter(group[nb_variable], group["mean_cos_sim"], color=color_dict[g][0], alpha=0.7, s=10, marker="+")
            # --- Annotate the plot with R² and p-value
            ax[i].text(0.05, 0.95, f"$r^2={r2:.3f}$ p-val={results["pvalue"]:.3f}", transform=ax[i].transAxes, fontsize=8, verticalalignment="top", color="black")
            ax[i].text(0.05, 0.90, f"$r^2={r2_gp1:.3f}$ p-val={results_gp1["pvalue"]:.3f}", transform=ax[i].transAxes, fontsize=8, verticalalignment="top", color=color_dict[gp1][0])
            ax[i].text(0.05, 0.85, f"$r^2={r2_gp2:.3f}$ p-val={results_gp2["pvalue"]:.3f}", transform=ax[i].transAxes, fontsize=8, verticalalignment="top", color=color_dict[gp2][0])
            ax[i].set_title(f"{label} trials", fontsize=10)
            ax[i].set_xlabel(nb_variable, fontsize=10)
            ax[i].set_ylabel("mean_cos_sim", fontsize=10)
            ax[i].set_ylim(ymin=0, ymax=1)
            ax[i].tick_params(axis='both', which='major', labelsize=5)
        fig_corr.suptitle(f"Correlation of the {nb_variable} with the mean cosine similarity", fontsize=10)
        fig_corr.canvas.manager.set_window_title(f"Corr_{nb_variable}_mean_cos_sim_{amplitude}_{metric}_{filter_out_nr}_{n_type}_{gp1}_{gp2}")
    # Plotting the comparison between behavior labels and genotypes
    fig_comp, ax_comp = plt.subplots(nrows=2, ncols=4, figsize=(24, 16), constrained_layout=True)
    variable_col = "mean_cos_sim"
    # Within comparisons
    ppt.boxplot(ax_comp[0, 0], mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp1) & (mean_cos_sim_data.Behavior == "Hit")][variable_col].values,
                mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp1) & (mean_cos_sim_data.Behavior == "Miss")][variable_col].values,
                ylabel=variable_col, paired=True, title=f"{gp1} - Hit/Miss", ylim=[0, 1.1],
                colors=color_dict[gp1], det_marker=True, force_markers_identity=False)
    # ppt.boxplot(ax_comp[1, 0], mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp2) & (mean_cos_sim_data.Behavior == "Hit")][variable_col].values,
    #             mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp2) & (mean_cos_sim_data.Behavior == "Miss")][variable_col].values,
    #             ylabel=variable_col, paired=True, title=f"{gp2} - Hit/Miss", ylim=[0, 1.1],
    #             colors=[ppt.hypo_color, ppt.hypo_light_color], det_marker=True, force_markers_identity=False)
    # Between Comparisons
    ppt.boxplot(ax_comp[0, 1], mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp1) & (mean_cos_sim_data.Behavior == "Nogo")][variable_col].values,
                mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp2) & (mean_cos_sim_data.Behavior == "Nogo")][variable_col].values,
                ylabel=variable_col, paired=False, title=f"{gp1}/{gp2} (Nogo)", ylim=[0, 1.1],
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ppt.boxplot(ax_comp[1, 1], mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp1) & (mean_cos_sim_data.Behavior == "All")][variable_col].values,
                mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp2) & (mean_cos_sim_data.Behavior == "All")][variable_col].values,
                ylabel=variable_col, paired=False, title=f"{gp1}/{gp2} (All)", ylim=[0, 1.1],
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ppt.boxplot(ax_comp[0, 2], mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp1) & (mean_cos_sim_data.Behavior == "Hit")][variable_col].values,
                mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp2) & (mean_cos_sim_data.Behavior == "Hit")][variable_col].values,
                ylabel=variable_col, paired=False, title=f"{gp1}/{gp2} (Hit)", ylim=[0, 1.1],
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ppt.boxplot(ax_comp[1, 2], mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp1) & (mean_cos_sim_data.Behavior == "Miss")][variable_col].values,
                mean_cos_sim_data[(mean_cos_sim_data.Genotype == gp2) & (mean_cos_sim_data.Behavior == "Miss")][variable_col].values,
                ylabel=variable_col, paired=False, title=f"{gp1}/{gp2} (Miss)", ylim=[0, 1.1],
                colors=[color_dict[gp1][1], color_dict[gp2][1]], det_marker=False, force_markers_identity=False)
    # SNR
    snr_gp1 = snr_df[snr_df.Genotype == gp1]["SNR"].values
    snr_gp2 = snr_df[snr_df.Genotype == gp2]["SNR"].values
    ppt.boxplot(ax_comp[0, 3], snr_gp1[np.isfinite(snr_gp1)], snr_gp2[np.isfinite(snr_gp2)],
                ylabel="SNR", paired=False, title=f"{gp1}/{gp2} (SNR)", ylim=[0, 5],
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ax_comp[1, 3].set_axis_off()
    fig_comp.suptitle(f"Comparison of the {variable_col} between pairs of neurons between {gp1} and {gp2}\n"
                      f"Amp: {amplitude} - No-Go: {nogo} - Neurons: {n_type} - NR filter: {filter_out_nr}", fontsize=12)
    title = f"ntn_cosim_comp_{amplitude}_{nogo}_{filter_out_nr}_{n_type}_{gp1}_{gp2}"
    fig_comp.canvas.manager.set_window_title(title)
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/9/{title}.pdf", format="pdf")
    plt.show()
    return mean_cos_sim_data, snr_df

def ntn_cosim_per_amp(mean_activity_df, metric="Resp", filter_out_nr=False):
    """Computing and plotting the neuronal cosine similarity for each amplitude"""
    all_data = mean_activity_df.copy()
    rows = []
    for amplitude in range(2, 14, 2):
        amp_data = all_data[all_data.Amplitude == amplitude].copy()
        hit = amp_data[amp_data.Behavior == True].copy()
        miss = amp_data[amp_data.Behavior == False].copy()
        for data, label in zip([hit, miss, amp_data], ["Hit", "Miss", "All"]):
            for i, rec_id in enumerate(data.ID.unique()):
                rec_data_long = data[data.ID == rec_id].copy()
                genotype = rec_data_long.Genotype.values[0]
                threshold = rec_data_long.Threshold.values[0]
                rec_data = rec_data_long.pivot(index="Neuron", columns="Trial", values=metric)
                # Filtering out the neurons that are never responsive
                if filter_out_nr:
                    rec_data = rec_data[~(rec_data == 0).all(axis=1)]
                rec_data = rec_data.loc[sorted(rec_data.index, key=lambda lab: (lab.split('_')[0], int(lab.split('_')[1])))]
                if rec_data.shape[0] < 2 or rec_data.shape[1] < 2:
                    print(f"Step 2 -> {rec_id}{genotype} has not data for {label} trials at {amplitude}µm ({rec_data.shape[0]} neurons / {rec_data.shape[1]} trial)")
                    continue
                # computing cosine similarity between pairs of neurons
                matrix = np.array(rec_data)
                n_neurons = matrix.shape[0]
                n_trials = matrix.shape[1]
                cos_sim_mat = cosine_similarity(matrix)
                abs_cos_sim = np.abs(cos_sim_mat)
                mean_cos_sim = abs_cos_sim[~np.eye(n_neurons, dtype=bool)].mean()
                rows.append({"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "Behavior": label, "Amplitude": amplitude,
                             "n_neurons": n_neurons, "n_trials": n_trials, "mean_cos_sim": mean_cos_sim})
    mean_cos_sim_amp_data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), constrained_layout=True)
    ppt.curveplot(ax[0], mean_cos_sim_amp_data[mean_cos_sim_amp_data["Behavior"] == "All"], between="Genotype", within="Amplitude",
                  variable="mean_cos_sim", data_points=True, title="All trials", ylabel=None, xlabel=None, ylim=None,
                  colors=[ppt.ko_color, ppt.hypo_color, ppt.wt_color], id_display=True,
                  legend_display=True, qq_show=True, transformation=None, consider_normality=True,
                  consider_homogeneity=False)
    ppt.curveplot(ax[1], mean_cos_sim_amp_data[mean_cos_sim_amp_data["Behavior"] == "Hit"], between="Genotype", within="Amplitude",
                  variable="mean_cos_sim", data_points=True, title="Hit trials", ylabel=None, xlabel=None, ylim=None,
                  colors=[ppt.ko_color, ppt.hypo_color, ppt.wt_color], id_display=True,
                  legend_display=True, qq_show=True, transformation=None, consider_normality=True,
                  consider_homogeneity=False)
    ppt.curveplot(ax[2], mean_cos_sim_amp_data[mean_cos_sim_amp_data["Behavior"] == "Miss"], between="Genotype", within="Amplitude",
                  variable="mean_cos_sim", data_points=True, title="Miss trials", ylabel=None, xlabel=None, ylim=None,
                  colors=[ppt.ko_color, ppt.hypo_color, ppt.wt_color], id_display=True,
                  legend_display=True, qq_show=True, transformation=None, consider_normality=True,
                  consider_homogeneity=False)
    fig.suptitle(f"Mean cosine similarity of {metric} across amplitudes\n Filter ou Non-Responsive neurons == {filter_out_nr}")
    fig.canvas.manager.set_window_title(f"Mean_cos_sim_amp_{metric}_{filter_out_nr}")
    plt.show()
    return mean_cos_sim_amp_data

# mean_cos_sim_amp_df = ntn_cosim_per_amp(activity_long_df, metric="Resp", filter_out_nr=True)


def temporal_correlation_neuron(frame_df):
    """Compute the correlation between the temporal dynamics of neurons within each trial"""
    data = frame_df.drop(columns=["resp"]).copy()
    rows = []
    for rec_id in data.ID.unique():
        rec_data = data[data.ID == rec_id]
        genotype = rec_data.Genotype.values[0]
        threshold = rec_data.Threshold.values[0]
        rec_rows = []
        for trial_id in rec_data.Trial.unique():
            trial_data = rec_data[rec_data.Trial == trial_id]
            duration = trial_data.Duration.values[0]
            amplitude = trial_data.Amplitude.values[0]
            behavior = trial_data.Behavior.values[0]
            n_neurons = len(trial_data)
            n_exc = (trial_data["n_type"] == "EXC").sum()
            trial_data = trial_data.drop(columns=["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Duration", "Behavior"]).copy()
            # Selecting the desired frames
            trial_data = trial_data.loc[:, 30:30+duration]
            # Computing the correlation between neurons
            corr_mat = np.corrcoef(trial_data)
            rec_rows.append({"Genotype": genotype, "ID": rec_id, "Threshold": threshold, "Amplitude": amplitude,
                         "Duration": duration, "Behavior": behavior, "n_exc": n_exc, "corr_mat": corr_mat})
        corr_data = pd.DataFrame(rec_rows)
        corr_data.sort_values(by=["Behavior", "Amplitude"], ascending=True, inplace=True)
        corr_data.reset_index(drop=True, inplace=True)
        # ====== Plotting each single correlation matrix of trials ======
        # ncols = int(np.ceil(np.sqrt(len(corr_data))))
        # nrows = int(np.ceil(len(corr_data) / ncols))
        # fig_all, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows), constrained_layout=True)
        # ax = axs.flatten()
        # for i, trial in corr_data.iterrows():
        #     ax[i].imshow(trial.corr_mat, cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
        #     ax[i].set_title(f"{trial.Amplitude}µm ({trial.Duration}s) - {trial.Behavior}", fontsize=8)
        #     ax[i].axhline(y=trial.n_exc - 0.5, ls="--", lw=0.5, color="black")
        #     ax[i].axvline(x=trial.n_exc - 0.5, ls="--", lw=0.5, color="black")
        #     ax[i].tick_params(axis="both", which="major", labelsize=6)
        # for ax in ax[len(corr_data):]:
        #     ax.axis("off")
        # fig_all.suptitle(f"Temporal dynamics correlation of neurons within trials\n {rec_id}({genotype}) [{threshold}]", fontsize=12)
        # fig_all.canvas.manager.set_window_title(f"temp_corr_{rec_id}")

        # ====== Plotting the mean matrix per behavioral outcome ======
        def mean_matrix(arrays):
            """given a sequence of (M×N) arrays, return their elementwise mean."""
            stack = np.stack(arrays, axis=0)
            return stack.mean(axis=0)
        # Computing the mean matrices for hit and miss trials, removing the no-go trials
        mean_mats = corr_data[corr_data.Amplitude != 0].groupby("Behavior")["corr_mat"].apply(lambda series_of_arrays: mean_matrix(series_of_arrays.values))
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), constrained_layout=True)
        for i, behavior_label in enumerate([True, False]):
            # ax[i].imshow(mean_mats[behavior_label], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
            mean_corr = mean_mats[behavior_label][~np.eye(n_neurons, dtype=bool)].mean()
            # ax[i].set_title(f"{behavior_label} [{mean_corr}]", fontsize=10)
            # ax[i].axhline(y=n_exc - 0.5, ls="--", lw=0.5, color="black")
            # ax[i].axvline(x=n_exc - 0.5, ls="--", lw=0.5, color="black")
            # ax[i].tick_params(axis="both", which="major", labelsize=6)
            rows.append({"Genotype": genotype, "ID": rec_id, "Behavior": behavior_label, "n_exc": n_exc, "Corr_mat": mean_mats[behavior_label], "Mean_corr": mean_corr})
        # fig.suptitle(f"Mean correlation matrices for hit and miss trials for {rec_id}({genotype})", fontsize=12)
    results_df = pd.DataFrame(rows)
    ncols = int(np.ceil(np.sqrt(len(results_df[results_df.Behavior == True]))))
    nrows = int(np.ceil(len(results_df[results_df.Behavior == True]) / ncols))
    fig_hit, axs_hit = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), constrained_layout=True)
    fig_miss, axs_miss = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), constrained_layout=True)
    ax_hit = axs_hit.flatten()
    ax_miss = axs_miss.flatten()
    for ax_id, rec_id in enumerate(results_df.ID.unique()):
        for behavior_ax, behavior_label in zip([ax_hit, ax_miss], [True, False]):
            current_row = results_df[(results_df.ID == rec_id) & (results_df.Behavior == behavior_label)]
            behavior_ax[ax_id].imshow(current_row["Corr_mat"].values[0], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
            behavior_ax[ax_id].set_title(f"{rec_id}({current_row["Genotype"].values[0]}) - {current_row["Mean_corr"].values[0]:.3f}", fontsize=8)
            behavior_ax[ax_id].axhline(y=current_row["n_exc"].values[0] - 0.5, ls="--", lw=0.5, color="black")
            behavior_ax[ax_id].axvline(x=current_row["n_exc"].values[0] - 0.5, ls="--", lw=0.5, color="black")
            behavior_ax[ax_id].tick_params(axis="both", which="major", labelsize=6)
    fig_hit.suptitle(f"Mean temporal correlation of neurons during hit trials", fontsize=12)
    fig_miss.suptitle(f"Mean temporal correlation of neurons during miss trials", fontsize=12)
    # Plotting the difference in mean correlation within and between genotypes
    fig_comp, ax_comp = plt.subplots(nrows=1, ncols=4, figsize=(24, 8), constrained_layout=True)
    ppt.boxplot(ax_comp[0], results_df[(results_df.Genotype == "WT") & (results_df.Behavior == True)]["Mean_corr"].values,
                results_df[(results_df.Genotype == "WT") & (results_df.Behavior == False)]["Mean_corr"].values,
                ylabel="Mean_corr", paired=True, title=f"Hit/Miss (WT)", ylim=[],
                colors=[ppt.wt_color, ppt.wt_light_color], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax_comp[1], results_df[(results_df.Genotype == "KO-Hypo") & (results_df.Behavior == True)]["Mean_corr"].values,
                results_df[(results_df.Genotype == "KO-Hypo") & (results_df.Behavior == False)]["Mean_corr"].values,
                ylabel="Mean_corr", paired=True, title=f"Hit/Miss (KO-Hypo)", ylim=[],
                colors=[ppt.hypo_color, ppt.hypo_light_color], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax_comp[2], results_df[(results_df.Genotype == "WT") & (results_df.Behavior == True)]["Mean_corr"].values,
                results_df[(results_df.Genotype == "KO-Hypo") & (results_df.Behavior == True)]["Mean_corr"].values,
                ylabel="Mean_corr", paired=False, title=f"WT/KO-Hypo (Hit)", ylim=[],
                colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax_comp[3], results_df[(results_df.Genotype == "WT") & (results_df.Behavior == False)]["Mean_corr"].values,
                results_df[(results_df.Genotype == "KO-Hypo") & (results_df.Behavior == False)]["Mean_corr"].values,
                ylabel="Mean_corr", paired=False, title=f"WT/KO-Hypo (Miss)", ylim=[],
                colors=[ppt.wt_light_color, ppt.hypo_light_color], det_marker=False, force_markers_identity=False)
    fig_comp.suptitle(f"Comparison of the mean temporal correlation between neurons during hit and miss trials within and between genotypes", fontsize=12)
    fig_comp.canvas.manager.set_window_title(f"Mean temporal correlation comp")
    plt.show()
    return mean_mats

# frame_data = get_activity_by_frame_df(recs.values(), zscore=True, BMS=False)
# temp_corr_df = temporal_correlation_neuron(frame_data[frame_data.ID.isin([4445]) & (frame_data.Amplitude == frame_data.Threshold)])
# temp_corr_df = temporal_correlation_neuron(frame_data)



# endregion ============================================================================================================
# region ======================================== Pre-stimulus =========================================================

def filter_stim_recruited_neurons(activity_df, det_filter=None, amplitude_filter=None, recruitment_filter="activated",
                                  n_type_filter=None, get_opposite=False):
    """Filter the activity dataframe to keep only the neurons that exhibits the specified pattern at least once during
    specified trials."""
    recruitment_dict = {"activated": [1], "inhibited": [-1], "recruited": [-1, 1], "non_resp": [0], "all": [-1, 0, 1]}
    # Filtering the neuron type
    df = activity_df.copy()
    if n_type_filter in ["EXC", "INH"]:
        df = df[df["Neuron"].str.startswith(n_type_filter)]
    # Filtering the amplitude
    if amplitude_filter == "threshold":
        df = df[df["Amplitude"] == df["Threshold"]]
    elif isinstance(amplitude_filter, list):
        df = df[df["Amplitude"].isin(amplitude_filter)]
    # Filtering the detection and recruitment
    if det_filter is not None:
        df = df[df["Behavior"] == det_filter]
    # Filtering the responsivity
    matching = df[df["Resp"].isin(recruitment_dict[recruitment_filter])][["ID", "Neuron"]].drop_duplicates()
    # Handling the case where we want to keep the neurons that never exhibits this pattern
    if get_opposite:
        # If we want the neurons that NEVER match, collect all neurons in the filtered df, drop those in matching
        all_pairs = df[["ID", "Neuron"]].drop_duplicates()
        # left-anti-join to remove matching
        opp = all_pairs.merge(matching, on=["ID", "Neuron"], how="left", indicator=True).query('_merge=="left_only"')[
            ["ID", "Neuron"]]
        valid_pairs = opp
    else:
        # The default: neurons that match at least once
        valid_pairs = matching
    # Returning the filtered dataframe with only valid neurons
    filtered_df = activity_df.merge(valid_pairs, on=["ID", "Neuron"], how="inner")
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
    gp_mean = activity_df.drop(columns=["Trial", "Amplitude", "Behavior", "Neuron", "Resp"]).groupby(["Genotype", "ID"],
                                                                                                     as_index=False).mean()
    gp_split = activity_df.drop(columns=["Trial", "Amplitude", "Neuron", "Resp"]).groupby(
        ["Genotype", "ID", "Behavior"], as_index=False).mean()
    gp_hit = gp_split[gp_split["Behavior"] == True]
    gp_miss = gp_split[gp_split["Behavior"] == False]
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    for col, (condition, data) in enumerate(zip(["Global", "Hit", "Miss"], [gp_mean, gp_hit, gp_miss])):
        wt = data[data["Genotype"] == "WT"]
        hypo = data[data["Genotype"] == "KO-Hypo"]
        ppt.boxplot(ax[0, col], wt["Prestim_mean"].values, hypo["Prestim_mean"].values, ylabel="Prestim_mean",
                    paired=False,
                    title=condition, ylim=[],
                    colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[1, col], wt["Prestim_std"].values, hypo["Prestim_std"].values, ylabel="Prestim_std",
                    paired=False,
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
    gp_data = data.drop(columns=["Trial", "Amplitude", "Neuron", "Resp"]).groupby(["Genotype", "ID", "Behavior"],
                                                                                  as_index=False).mean()
    hit_data = gp_data[gp_data["Behavior"] == True]
    miss_data = gp_data[gp_data["Behavior"] == False]
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    for col, genotype in enumerate(gp_data["Genotype"].unique()):
        hit = hit_data[hit_data["Genotype"] == genotype]
        miss = miss_data[miss_data["Genotype"] == genotype]
        ppt.boxplot(ax[0, col], hit["Diff_mean"].values, miss["Diff_mean"].values, ylabel="Mean stim - mean prestim",
                    paired=True, title=genotype, ylim=[], colors=color_dict[genotype], det_marker=False,
                    force_markers_identity=False)
        ppt.boxplot(ax[1, col], hit["Ratio_std"].values, miss["Ratio_std"].values, ylabel="Std stim / std prestim",
                    paired=True, title=genotype, ylim=[], colors=color_dict[genotype], det_marker=False,
                    force_markers_identity=False)
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
    gp_data = data.drop(columns=["Trial", "Amplitude", "Neuron", "Resp", "Behavior"]).groupby(["Genotype", "ID"],
                                                                                              as_index=False).mean()
    wt = gp_data[gp_data["Genotype"] == "WT"]
    hypo = gp_data[gp_data["Genotype"] == "KO-Hypo"]
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), constrained_layout=True)
    ax = axs.flatten()
    for col, param in enumerate(
            ["Abs_Diff_mean", "Abs_Ratio_std", "Stim_mean", "Prestim_mean", "Stim_std", "Prestim_std"]):
        ppt.boxplot(ax[col], wt[param].values, hypo[param].values, ylabel="DFF",
                    paired=False, title=param, ylim=[], colors=[ppt.wt_color, ppt.hypo_color], det_marker=False,
                    force_markers_identity=False)
    ppt.boxplot(ax[6], wt["Prestim_mean"].values, wt["Stim_mean"].values, ylabel="DFF",
                paired=True, title="Pre-stim/stim WT", ylim=[], colors=[ppt.wt_color, ppt.wt_color], det_marker=False,
                force_markers_identity=False)
    ppt.boxplot(ax[7], hypo["Prestim_mean"].values, hypo["Stim_mean"].values, ylabel="DFF",
                paired=True, title="Pre-stim/stim KO-Hypo", ylim=[], colors=[ppt.hypo_color, ppt.hypo_color],
                det_marker=False,
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
    gp_neuron = data.drop(columns=["Trial", "Amplitude", "Behavior", "Threshold"]).groupby(
        ["Genotype", "ID", "Neuron", "Resp"], as_index=False).mean()
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
        ppt.boxplot(ax[2, col], act_trials["Abs_Diff_mean"].values, non_trials["Abs_Diff_mean"].values,
                    ylabel="Mean DFF",
                    paired=True, title="Abs_Diff_mean (act/non_act)", ylim=[], colors=color_dict[genotype],
                    det_marker=False, force_markers_identity=False)
    fig.suptitle(
        f"Comparison of mean pre-stimulus activity of stimulus activated neurons between trials when they were activated or not",
        fontsize=12)
    fig.canvas.manager.set_window_title("Pre-stim comp (act vs non_act)")
    # plt.savefig(
    #     "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/Pre-stim_comp_(act_vs_non_act).pdf")
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
        rec_data = activity_df[
            (activity_df["ID"] == rec_id) & (activity_df["Amplitude"] == activity_df["Threshold"])].copy()
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
                raw_mean = stat_boxplot(activated["Prestim_mean"], non_activated["Prestim_mean"], "Prestim_mean",
                                        title="", paired=False, verbose=False)
                raw_std = stat_boxplot(activated["Prestim_std"], non_activated["Prestim_std"], "Prestim_std", title="",
                                       paired=False, verbose=False)
                diff_mean = stat_boxplot(activated["Diff_mean"], non_activated["Diff_mean"], "Diff_mean", title="",
                                         paired=False, verbose=False)
                diff_std = stat_boxplot(activated["Diff_std"], non_activated["Diff_std"], "Diff_std", title="",
                                        paired=False, verbose=False)
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
        ax[ax_id].bar(range(len(data)), data[param].values, color=colors, alpha=0.7, width=0.5)
        ax[ax_id].set_title(f"{rec_id} - {data["Genotype"].values[0]}", fontsize=12)
        ax[ax_id].axhline(y=0.05, color='gray', linestyle='--', lw=0.5)
        ax[ax_id].tick_params(axis='both', labelsize=10)
    fig.suptitle(
        f"Significance of the difference in pre-stim activity ({param}) for each neuron (activated vs. non activated)",
        fontsize=15)
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
        cos_sim_matrix = pd.DataFrame(cosine_similarity(combined_df.T), index=combined_df.columns,
                                      columns=combined_df.columns)
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
        hit_sim = grouped[(grouped["Table_1"] == "Hit") & (grouped["Table_2"] == "Hit")]["abs_cos_sim"].values[
            0] if len(grouped[(grouped["Table_1"] == "Hit") & (grouped["Table_2"] == "Hit")][
                          "abs_cos_sim"].values) > 0 else np.nan
        miss_sim = grouped[(grouped["Table_1"] == "Miss") & (grouped["Table_2"] == "Miss")]["abs_cos_sim"].values[
            0] if len(grouped[(grouped["Table_1"] == "Miss") & (grouped["Table_2"] == "Miss")][
                          "abs_cos_sim"].values) > 0 else np.nan
        btw_sim = grouped[grouped["Table_1"] != grouped["Table_2"]]["abs_cos_sim"].values[0]
        rows.append({"Genotype": rec_data["Genotype"].values[0], "ID": rec_id, "Hit_sim": hit_sim,
                     "Miss_sim": miss_sim, "Between_sim": btw_sim})
    results = pd.DataFrame(rows)
    # Plotting the difference
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 12), constrained_layout=True)
    for row, genotype in enumerate(results["Genotype"].unique()):
        colors = color_dict[genotype]
        data = results[results["Genotype"] == genotype].copy()
        ppt.boxplot(ax[row, 0], data["Hit_sim"].values, data["Miss_sim"].values, ylabel="Mean_abs_cos_sim", paired=True,
                    title=f"{genotype} - Hit/Miss", ylim=[],
                    colors=colors, det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 1], data["Hit_sim"].values, data["Between_sim"].values, ylabel="Mean_abs_cos_sim",
                    paired=True, title=f"{genotype} - Hit/btw", ylim=[],
                    colors=[colors[0], "purple"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 2], data["Miss_sim"].values, data["Between_sim"].values, ylabel="Mean_abs_cos_sim",
                    paired=True, title=f"{genotype} - Miss/btw", ylim=[],
                    colors=[colors[1], "purple"], det_marker=False, force_markers_identity=False)
    wt = results[results["Genotype"] == "WT"].copy()
    hypo = results[results["Genotype"] == "KO-Hypo"].copy()
    colors2 = {"Hit_sim": [ppt.wt_color, ppt.hypo_color], "Miss_sim": [ppt.wt_light_color, ppt.hypo_light_color],
               "Between_sim": ["purple", "purple"]}
    for row2, comp in enumerate(["Hit_sim", "Miss_sim", "Between_sim"]):
        ppt.boxplot(ax[row2, 3], wt[comp].values, hypo[comp].values, ylabel="Mean_abs_cos_sim", paired=False,
                    title=f"WT/KO-Hypo - {comp}", ylim=[],
                    colors=colors2[comp], det_marker=False, force_markers_identity=False)
    fig.suptitle(
        f"Comparison of mean cosine similarity across pairs of trials within and between condition, threshold trials"
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
                         "EI_ratio": ei_ratio_vector[trial_id],})
                         # "EI_ratio_cste": cst_ei_ratio_vector[trial_id],
                         # "EI_ratio_norm": norm_ei_ratio_vector[trial_id],
                         # "EI_ratio_log_cste": log_cst_ei_ratio_vector[trial_id],
                         # "EI_ratio_log_norm": log_norm_ei_ratio_vector[trial_id],
                         # "EI_ratio_norm_dif": norm_dif_ei_vector[trial_id],
                         # "EI_ratio_dif": dif_ei_vector[trial_id]})
    return pd.DataFrame(rows)


def plot_ei_variance(ei_df):
    data = ei_df[ei_df["Amplitude"] != 0]
    data = data[np.isfinite(data["EI_ratio"])]
    data_all = data.drop(columns=["Behavior", "Trial", "Amplitude"]).groupby(["Genotype", "ID"], as_index=False).std()
    data_split = data.drop(columns=["Trial", "Amplitude"]).groupby(["Genotype", "ID", "Behavior"], as_index=False).std()
    common_ids = set(data_split[data_split["Behavior"] == True]["ID"].values) & set(data_split[data_split["Behavior"] == False]["ID"].values)
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 8), constrained_layout=True)
    # All
    ppt.boxplot(ax[0], data_all[data_all["Genotype"] == "WT"]["EI_ratio"].values,
                data_all[data_all["Genotype"] == "KO-Hypo"]["EI_ratio"].values,
                paired=False, ylabel="E:I ratio std", ylim=[0, 5], title="All trials",
                colors=[ppt.wt_color, ppt.hypo_color], det_marker=True, force_markers_identity=True)
    # Hit vs Miss
    ppt.boxplot(ax[1], data_split[(data_split["Genotype"] == "WT") & (data_split["Behavior"] == True) & (data_split["ID"].isin(common_ids))]["EI_ratio"].values,
                data_split[(data_split["Genotype"] == "WT") & (data_split["Behavior"] == False) & (data_split["ID"].isin(common_ids))]["EI_ratio"].values,
                paired=True, ylabel="E:I ratio std", ylim=[0, 5], title="WT",
                colors=[ppt.wt_color, ppt.wt_light_color], det_marker=True, force_markers_identity=False)
    ppt.boxplot(ax[2], data_split[(data_split["Genotype"] == "KO-Hypo") & (data_split["Behavior"] == True) & (data_split["ID"].isin(common_ids))]["EI_ratio"].values,
                data_split[(data_split["Genotype"] == "KO-Hypo") & (data_split["Behavior"] == False) & (data_split["ID"].isin(common_ids))]["EI_ratio"].values,
                paired=True, ylabel="E:I ratio std", ylim=[0, 5], title="KO-Hypo",
                colors=[ppt.hypo_color, ppt.hypo_light_color], det_marker=True, force_markers_identity=False)
    # Hit
    ppt.boxplot(ax[3], data_split[(data_split["Genotype"] == "WT") & (data_split["Behavior"] == True)]["EI_ratio"].values,
                data_split[(data_split["Genotype"] == "KO-Hypo") & (data_split["Behavior"] == True)]["EI_ratio"].values,
                paired=False, ylabel="E:I ratio std", ylim=[0, 5], title="Hit",
                colors=[ppt.wt_color, ppt.hypo_color], det_marker=True, force_markers_identity=True)
    # Miss
    ppt.boxplot(ax[4], data_split[(data_split["Genotype"] == "WT") & (data_split["Behavior"] == False)]["EI_ratio"].values,
                data_split[(data_split["Genotype"] == "KO-Hypo") & (data_split["Behavior"] == False)]["EI_ratio"].values,
                paired=False, ylabel="E:I ratio std", ylim=[0, 5], title="Miss",
                colors=[ppt.wt_light_color, ppt.hypo_light_color], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"EI ratio variance across trials")
    plt.show()
    return common_ids


def correlate_behavior(data, column=None):
    if column is None:
        column = data.columns[-1]
    rows = []
    for rec_id in data["ID"].unique():
        rec_data = data[data["ID"] == rec_id]
        r, pval = ss.pointbiserialr(rec_data[column], rec_data["Behavior"])
        rows.append({"ID": rec_data["ID"].values[0], "Genotype": rec_data["Genotype"].values[0],
                     "Threshold": rec_data["Threshold"].values[0], "R2": r ** 2, "pval": pval})
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
    ppt.boxplot(ax[0, 0], gp1_det, gp1_undet, ylabel="E/I ratio", paired=True, title=f"{gp1}", ylim=[],
                colors=colors_dict[gp1], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[0, 1], gp2_det, gp2_undet, ylabel="E/I ratio", paired=True, title=f"{gp2}", ylim=[],
                colors=colors_dict[gp2], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 0], gp1_det, gp2_det, ylabel="E/I ratio", paired=False, title="Detected Trials", ylim=[],
                colors=[colors_dict[gp1][0], colors_dict[gp2][0]], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 1], gp1_undet, gp2_undet, ylabel="E/I ratio", paired=False, title="Non-Detected Trials", ylim=[],
                colors=[colors_dict[gp1][1], colors_dict[gp2][1]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"E/I ratio comparison ({gp1} & {gp2})\n [method = {column}]", fontsize=16)
    fig.canvas.manager.set_window_title(f"EI_comparison_{gp1}_{gp2})_{column}")
    plt.show()
    return data


def population_EI_ratio(recs, pyr_inhibition=False, amp="threshold", test_INH=False, gp1="WT", gp2="KO-Hypo"):
    """ Computes the averaged population E/I ratio in hit and miss trials for both genotypes and compare them"""
    rows = []
    for rec in recs:
        n_exc = rec.zscore_exc.shape[0]
        n_inh = rec.zscore_inh.shape[0]
        # Selection of the desired amplitude
        if amp == "threshold":
            amp_vector = rec.stim_ampl == rec.session_threshold
        elif amp == "wt_threshold":
            amp_vector = rec.stim_ampl == 4
        elif amp == "all":
            amp_vector = [True] * len(rec.stim_ampl)
        # Selection of the hit and miss trials of the specified amplitude
        hit_vector = rec.detected_stim & amp_vector
        miss_vector = ~rec.detected_stim & amp_vector
        # Selection of the method use to compute the E/I ratio
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
        all_exc = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, amp_vector] == 1, axis=0)) / n_exc
        hit_exc = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, hit_vector] == 1, axis=0)) / n_exc
        miss_exc = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, miss_vector] == 1, axis=0)) / n_exc
        all_inh = np.mean(np.count_nonzero(rec.matrices[I_type]["Responsivity"][:, amp_vector] == I_activity, axis=0)) / I_nb
        hit_inh = np.mean(np.count_nonzero(rec.matrices[I_type]["Responsivity"][:, hit_vector] == I_activity, axis=0)) / I_nb
        miss_inh = np.mean(np.count_nonzero(rec.matrices[I_type]["Responsivity"][:, miss_vector] == I_activity, axis=0)) / I_nb
        ei_all = all_exc / all_inh
        ei_hit = hit_exc / hit_inh
        ei_miss = miss_exc / miss_inh
        if test_INH:
            all_exc_INH = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, amp_vector] == -1, axis=0)) / n_exc
            hit_exc_INH = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, hit_vector] == -1, axis=0)) / n_exc
            miss_exc_INH = np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, miss_vector] == -1, axis=0)) / n_exc
            all_inh_INH = np.mean(np.count_nonzero(rec.matrices["INH"]["Responsivity"][:, amp_vector] == 1, axis=0)) / n_inh
            hit_inh_INH = np.mean(np.count_nonzero(rec.matrices["INH"]["Responsivity"][:, hit_vector] == 1, axis=0)) / n_inh
            miss_inh_INH = np.mean(np.count_nonzero(rec.matrices["INH"]["Responsivity"][:, miss_vector] == 1, axis=0)) / n_inh
            ei_all = all_exc_INH / all_inh_INH
            ei_hit = hit_exc_INH / hit_inh_INH
            ei_miss = miss_exc_INH / miss_inh_INH
            EI_label = "% inhibited EXC / % activated INH"
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "EI_all": ei_all, "EI_hit": ei_hit, "EI_miss": ei_miss})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 8), constrained_layout=True)
    gp1_all = data[data["Genotype"] == gp1]["EI_all"].values
    gp2_all = data[data["Genotype"] == gp2]["EI_all"].values
    gp1_hit = data[data["Genotype"] == gp1]["EI_hit"]
    gp1_hit_un = gp1_hit[np.isfinite(gp1_hit)].values
    gp2_hit = data[data["Genotype"] == gp2]["EI_hit"]
    gp2_hit_un = gp2_hit[np.isfinite(gp2_hit)].values
    gp1_miss = data[data["Genotype"] == gp1]["EI_miss"]
    gp1_miss_un = gp1_miss[np.isfinite(gp1_miss)].values
    gp2_miss = data[data["Genotype"] == gp2]["EI_miss"]
    gp2_miss_un = gp2_miss[np.isfinite(gp2_miss)].values
    gp1_hit_paired = gp1_hit[(np.isfinite(gp1_hit) & np.isfinite(gp1_miss))].values
    gp1_miss_paired = gp1_miss[(np.isfinite(gp1_hit) & np.isfinite(gp1_miss))].values
    gp2_hit_paired = gp2_hit[(np.isfinite(gp2_hit) & np.isfinite(gp2_miss))].values
    gp2_miss_paired = gp2_miss[(np.isfinite(gp2_hit) & np.isfinite(gp2_miss))].values
    ppt.boxplot(ax[0], gp1_all, gp2_all, paired=False, ylabel="E:I ratio", ylim=[0, 5], title="All trials",
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ppt.boxplot(ax[1], gp1_hit_un, gp2_hit_un, paired=False, ylabel="E:I ratio", ylim=[0, 5], title="Hit trials",
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ppt.boxplot(ax[2], gp1_miss_un, gp2_miss_un, paired=False, ylabel="E:I ratio", ylim=[0, 5], title="Miss trials",
                colors=[color_dict[gp1][1], color_dict[gp2][1]], det_marker=False)
    ppt.boxplot(ax[3], gp1_hit_paired, gp1_miss_paired, paired=True, ylabel="E:I ratio", ylim=[0, 5], title=gp1,
                colors=color_dict[gp1], det_marker=True)
    ppt.boxplot(ax[4], gp2_hit_paired, gp2_miss_paired, paired=True, ylabel="E:I ratio", ylim=[0, 5], title=gp2,
                colors=color_dict[gp2], det_marker=True)
    fig.suptitle(f"E:I ratio comparison between {gp1} and {gp2}\n defined as {"% activated EXC / " if not test_INH else EI_label}{I_label if not test_INH else ""}  \n amp = {amp}", fontsize=12)
    fig.canvas.manager.set_window_title(f"EI_comparison_{gp1}_{gp2}")
    plt.show()
    return data


def correlate_ntn_ei(ntn_df, ei_df, corr_threshold_ei=False):
    variable = "Threshold" if corr_threshold_ei else "mean_cos_sim"
    color_dict = {"WT": ppt.wt_color, "KO-Hypo": ppt.hypo_color, "KO": ppt.ko_color}
    ntn = ntn_df[ntn_df["Behavior"] == "All"][["ID", "Threshold", "mean_cos_sim"]]
    data = ei_df.merge(ntn, on="ID", how="left").drop(columns=["EI_hit", "EI_miss"])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), constrained_layout=True)
    for y_legend, geno in zip([0.95, 0.90, 0.85], ["WT", "KO-Hypo", "KO"]):
        x = data[data["Genotype"] == geno]["EI_all"].values
        y = data[data["Genotype"] == geno][variable].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        results = dict(linregress(x, y)._asdict())
        r2 = results["rvalue"] ** 2
        line = results["slope"] * x + results["intercept"]
        # Plot the data points and regression line
        ax.scatter(x, y, color=color_dict[geno], alpha=0.7, s=10, marker="+")
        ax.plot(x, line, color=color_dict[geno], lw=2)
        ax.text(0.05, y_legend, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}", transform=ax.transAxes,
                        fontsize=8, verticalalignment="top", color=color_dict[geno])
    # Computing and plotting the global correlation
    x = data["EI_all"].values
    y = data[variable].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    results = dict(linregress(x, y)._asdict())
    r2 = results["rvalue"] ** 2
    line = results["slope"] * x + results["intercept"]
    # Plot the data points and regression line
    ax.plot(x, line, color="black", lw=2)
    ax.text(0.05, 0.80, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}", transform=ax.transAxes,
            fontsize=8, verticalalignment="top", color="black")
    ax.set_xlabel("EI_all", fontsize=12)
    ax.set_ylabel(variable, fontsize=12)
    fig.suptitle(f"Correlation of {variable} and EI ratio for all trials", fontsize=14)
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
        pyr_inh = (np.mean(np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, threshold_stim_vector] == -1,
                                            axis=0)) / n_exc) * 100
        gaba_act = (np.mean(
            np.count_nonzero(rec.matrices["INH"]["Responsivity"][:, threshold_stim_vector] == 1, axis=0)) / n_inh) * 100
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
        ax[col].text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}", transform=ax[col].transAxes,
                     fontsize=8,
                     verticalalignment="top", color="black")
        ax[col].set_title(geno, color=color_dict[geno])
        ax[col].set_xlabel("% activated GABAergic neurons", fontsize=8)
        ax[col].set_ylabel("% inhibited Pyramidal neurons", fontsize=8)
    fig.suptitle(
        "Correlation of the mean percentage of activated GABAergic interneurons with the number of inhibited Pyramidal neurons for threshold stimuli",
        fontsize=12)
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
            ppt.boxplot(ax[row_id, 2 * col_id], bsl[bsl["Genotype"] == genotype][col].values,
                        hit[hit["Genotype"] == genotype][col].values,
                        ylabel=col, paired=True, title=f"{genotype} Bsl/Hit", ylim=[],
                        colors=["purple", color_dict[genotype][0]], det_marker=False, force_markers_identity=False)
            ppt.boxplot(ax[row_id, 2 * col_id + 1], bsl[bsl["Genotype"] == genotype][col].values,
                        miss[miss["Genotype"] == genotype][col].values,
                        ylabel=col, paired=True, title=f"{genotype} Bsl/Miss", ylim=[],
                        colors=["purple", color_dict[genotype][1]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[3, 2 * col_id], bsl[bsl["Genotype"] == "WT"][col].values,
                    bsl[bsl["Genotype"] == "KO-Hypo"][col].values,
                    ylabel=col, paired=False, title="WT/KO-Hypo Bsl", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False,
                    force_markers_identity=False)
        ppt.boxplot(ax[3, 2 * col_id + 1], bsl[bsl["Genotype"] == "WT"][col].values,
                    bsl[bsl["Genotype"] == "KO"][col].values,
                    ylabel=col, paired=False, title="WT/KO Bsl", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO"][0]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of percentage of recruited neurons during hit and miss trials to baseline (no-go trials)",
                 fontsize=12)
    fig.canvas.manager.set_window_title(f"Baseline")
    # SNR
    fig_snr, ax_snr = plt.subplots(nrows=5, ncols=6, figsize=(25, 14), constrained_layout=True)
    for df in [hit, miss, bsl]:
        df["recr_EXC_perc"] = df["act_EXC_perc"] + df["inh_EXC_perc"]
        df["recr_INH_perc"] = df["act_INH_perc"] + df["inh_INH_perc"]
    key_cols = ["Genotype", "ID"]
    bsl_snr_cols = bsl.columns[-6:]  # last 6 columns assumed to be SNR-related
    hit_snr = hit[key_cols + list(bsl_snr_cols)].merge(bsl[key_cols + list(bsl_snr_cols)], on=key_cols,
                                                       suffixes=("_hit", "_bsl"))
    miss_snr = miss[key_cols + list(bsl_snr_cols)].merge(bsl[key_cols + list(bsl_snr_cols)], on=key_cols,
                                                         suffixes=("_miss", "_bsl"))
    for col in bsl_snr_cols:
        hit_snr[col] = hit_snr[f"{col}_hit"] / hit_snr[f"{col}_bsl"]
        miss_snr[col] = miss_snr[f"{col}_miss"] / miss_snr[f"{col}_bsl"]
    hit_snr = hit_snr[key_cols + list(bsl_snr_cols)]
    miss_snr = miss_snr[key_cols + list(bsl_snr_cols)]
    for col_id, col in enumerate(bsl_snr_cols):
        for row_id, genotype in enumerate(bsl["Genotype"].unique()):
            ppt.boxplot(ax_snr[row_id, col_id], hit_snr[hit_snr["Genotype"] == genotype][col].values,
                        miss_snr[miss_snr["Genotype"] == genotype][col].values,
                        ylabel=col, paired=True, title=f"Hit/Miss {genotype}", ylim=[],
                        colors=color_dict[genotype], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_snr[3, col_id],
                    hit_snr[hit_snr["Genotype"] == "WT"][col].values,
                    hit_snr[hit_snr["Genotype"] == "KO-Hypo"][col].values,
                    ylabel=col, paired=True, title="Hit SNR WT/KO-Hypo", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False,
                    force_markers_identity=False)
        ppt.boxplot(ax_snr[4, col_id],
                    miss_snr[miss_snr["Genotype"] == "WT"][col].values,
                    miss_snr[miss_snr["Genotype"] == "KO-Hypo"][col].values,
                    ylabel=col, paired=True, title="Miss SNR WT/KO-Hypo", ylim=[],
                    colors=[color_dict["WT"][1], color_dict["KO-Hypo"][1]], det_marker=False,
                    force_markers_identity=False)
    fig_snr.suptitle(f"Comparison of the SNR between hit and miss trials and between genotypes", fontsize=12)
    fig_snr.canvas.manager.set_window_title("SNR")
    plt.show()
    return bsl, hit, miss, hit_snr, miss_snr

def compare_prestim(mean_activity_df, threshold_only=False):
    """Compare the pre-stimulus period between WT and KO mice"""
    activity_df = mean_activity_df.copy()
    activity_df["cum_AUC_diff"] = activity_df["cum_AUC"] - activity_df["cum_AUC_pre"]
    no_go = activity_df[activity_df.Amplitude == 0]
    activity_df = activity_df[activity_df.Amplitude != 0]
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    amp_grouped = activity_df.drop(columns=["Neuron", "Resp", "FA"]).groupby(["Genotype", "ID", "Threshold", "Behavior", "Amplitude"], as_index=False).mean()
    nogo_grouped = no_go.drop(columns=["Neuron", "Resp", "Amplitude", "Behavior"]).groupby(["Genotype", "ID", "Threshold", "FA"], as_index=False).mean()
    nogo_fa = nogo_grouped[nogo_grouped.FA == True]
    nogo_cr = nogo_grouped[nogo_grouped.FA == False]
    if threshold_only:
        all_grouped = activity_df[activity_df.Amplitude == activity_df.Threshold].drop(columns=["Neuron", "Resp", "FA", "Behavior"]).groupby(["Genotype", "ID", "Threshold"], as_index=False).mean()
        hit = amp_grouped[(amp_grouped.Amplitude == amp_grouped.Threshold) & (amp_grouped.Behavior == True)].set_index("ID")
        miss = amp_grouped[(amp_grouped.Amplitude == amp_grouped.Threshold) & (amp_grouped.Behavior == False)].set_index("ID")
    else:
        all_grouped = activity_df.drop(columns=["Neuron", "Resp", "FA", "Behavior"]).groupby(["Genotype", "ID", "Threshold"], as_index=False).mean()
        behavior_grouped = activity_df.drop(columns=["Neuron", "Resp", "FA"]).groupby(["Genotype", "ID", "Threshold", "Behavior"], as_index=False).mean()
        hit = behavior_grouped[behavior_grouped.Behavior == True].set_index("ID")
        miss = behavior_grouped[behavior_grouped.Behavior == False].set_index("ID")
    numeric_cols = [col for col in hit.columns if col not in ["Genotype", "ID", "Threshold", "Trial", "Behavior", "Amplitude", "Neuron", "Resp", "FA"]]
    delta = hit.copy()
    delta[numeric_cols] = hit[numeric_cols].subtract(miss[numeric_cols])
    fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(48, 16), constrained_layout=True)
    wt_all = all_grouped[all_grouped.Genotype == "WT"]
    wt_hit = hit[hit.Genotype == "WT"].sort_values(by="ID", ascending=True)
    wt_miss = miss[miss.Genotype == "WT"].sort_values(by="ID", ascending=True)
    wt_delta = delta[delta.Genotype == "WT"]
    wt_fa = nogo_fa[nogo_fa.Genotype == "WT"]
    wt_cr_full = nogo_cr[nogo_cr.Genotype == "WT"]
    wt_cr = wt_cr_full[wt_cr_full["ID"].isin(wt_fa["ID"])]
    hypo_all = all_grouped[all_grouped.Genotype == "KO-Hypo"]
    hypo_hit = hit[hit.Genotype == "KO-Hypo"].sort_values(by="ID", ascending=True)
    hypo_miss = miss[miss.Genotype == "KO-Hypo"].sort_values(by="ID", ascending=True)
    hypo_delta = delta[delta.Genotype == "KO-Hypo"]
    hypo_fa = nogo_fa[nogo_fa.Genotype == "KO-Hypo"]
    hypo_cr_full = nogo_cr[nogo_cr.Genotype == "KO-Hypo"]
    hypo_cr = hypo_cr_full[hypo_cr_full["ID"].isin(hypo_fa["ID"])]
    for row, auc_metric in enumerate(["cum_AUC_fixpre", "neg_AUC_fixpre"]):
        # Between Hit/Miss
        ppt.boxplot(ax[row, 0], wt_hit[auc_metric].values, wt_miss[auc_metric].values,
                    ylabel=auc_metric, paired=True, title="WT Hit/Miss", ylim=[],
                    colors=color_dict["WT"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 1], hypo_hit[auc_metric].values, hypo_miss[auc_metric].values,
                    ylabel=auc_metric, paired=True, title="KO-Hypo Hit/Miss", ylim=[],
                    colors=color_dict["KO-Hypo"], det_marker=False, force_markers_identity=False)
        # Between FA/CR
        ppt.boxplot(ax[row, 2], wt_fa[auc_metric].values, wt_cr[auc_metric].values,
                    ylabel=auc_metric, paired=True, title="WT FA/CR", ylim=[],
                    colors=color_dict["WT"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 3], hypo_fa[auc_metric].values, hypo_cr[auc_metric].values,
                    ylabel=auc_metric, paired=True, title="KO-Hypo FA/CR", ylim=[],
                    colors=color_dict["KO-Hypo"], det_marker=False, force_markers_identity=False)
        # Between Genotypes
        ppt.boxplot(ax[row, 4], wt_all[auc_metric].values, hypo_all[auc_metric].values,
                    ylabel=auc_metric, paired=False, title="All", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 5], wt_hit[auc_metric].values, hypo_hit[auc_metric].values,
                    ylabel=auc_metric, paired=False, title="Hit", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 6], wt_miss[auc_metric].values, hypo_miss[auc_metric].values,
                    ylabel=auc_metric, paired=False, title="Miss", ylim=[],
                    colors=[color_dict["WT"][1], color_dict["KO-Hypo"][1]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 7], wt_delta[auc_metric].values, hypo_delta[auc_metric].values,
                    ylabel=auc_metric, paired=False, title="Delta Hit/Miss", ylim=[],
                    colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of prestim AUC\n Threshold only == {threshold_only}")
    fig.canvas.manager.set_window_title("Prestim AUC comp")
    plt.show()
    return all_grouped

# prestim_auc_df = compare_prestim(activity_long_df[activity_long_df.ID != 4456], threshold_only=False)
#prestim_auc_df = compare_prestim(activity_long_dff, threshold_only=False)

def diff_AUC_stim_prestim(mean_activity_df):
    """Is there a difference in how (AUC) the responsive neurons respond between trial outcome and genotypes"""
    activity_df = mean_activity_df.copy()
    activity_df["cum_AUC_diff"] = activity_df["cum_AUC"] - activity_df["cum_AUC_pre"]
    # Filtering out the non-responsive neurons
    activity_df = activity_df[activity_df["Resp"] != 0]
    # Filtering out the no-go trials
    activity_df = activity_df[activity_df["Amplitude"] != 0].drop(columns=["FA"])
    grouped_behavior = activity_df.drop(columns=["Trial", "Amplitude", "Neuron", "Resp"]).groupby(["Genotype", "ID", "Behavior"], as_index=False).mean()
    grouped_all = activity_df.drop(columns=["Trial", "Amplitude", "Neuron", "Resp", "Behavior"]).groupby(["Genotype", "ID"], as_index=False).mean()
    # Defining the different groups
    wt_hit = grouped_behavior[(grouped_behavior.Genotype == "WT") & (grouped_behavior.Behavior == True)].set_index("ID").reindex()
    wt_miss = grouped_behavior[(grouped_behavior.Genotype == "WT") & (grouped_behavior.Behavior == False)].set_index("ID").reindex()
    wt_delta = wt_hit.drop(columns=["Genotype", "Behavior"]) - wt_miss.drop(columns=["Genotype", "Behavior"])
    hypo_hit = grouped_behavior[(grouped_behavior.Genotype == "KO-Hypo") & (grouped_behavior.Behavior == True)].set_index("ID").reindex()
    hypo_miss = grouped_behavior[(grouped_behavior.Genotype == "KO-Hypo") & (grouped_behavior.Behavior == False)].set_index("ID").reindex()
    hypo_delta = hypo_hit.drop(columns=["Genotype", "Behavior"]) - hypo_miss.drop(columns=["Genotype", "Behavior"])
    # Plotting the comparisons
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 16), constrained_layout=True)
    ppt.boxplot(ax[0, 0], grouped_all[grouped_all.Genotype == "WT"]["cum_AUC_diff"].values,
                grouped_all[grouped_all.Genotype == "KO-Hypo"]["cum_AUC_diff"].values,
                ylabel="cum_AUC_diff", paired=False, title="All trials", ylim=[],
                colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 0], wt_delta["cum_AUC_diff"].values, hypo_delta["cum_AUC_diff"].values,
                ylabel="cum_AUC_diff", paired=False, title="Deltas", ylim=[],
                colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[0, 1], wt_hit["cum_AUC_diff"].values, wt_miss["cum_AUC_diff"].values,
                ylabel="cum_AUC_diff", paired=True, title="WT Hit/Miss", ylim=[],
                colors=color_dict["WT"], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 1], hypo_hit["cum_AUC_diff"].values, hypo_miss["cum_AUC_diff"].values,
                ylabel="cum_AUC_diff", paired=True, title="Hypo Hit/Miss", ylim=[],
                colors=color_dict["KO-Hypo"], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[0, 2], wt_hit["cum_AUC_diff"].values, hypo_hit["cum_AUC_diff"].values,
                ylabel="cum_AUC_diff", paired=False, title="Hit", ylim=[],
                colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
    ppt.boxplot(ax[1, 2], hypo_hit["cum_AUC_diff"].values, hypo_miss["cum_AUC_diff"].values,
                ylabel="cum_AUC_diff", paired=False, title="Miss", ylim=[],
                colors=[color_dict["WT"][1], color_dict["KO-Hypo"][1]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison between trials types and genotypes of the difference of cumulative AUC between stim and prestim for responsive neurons", fontsize=12)
    fig.canvas.manager.set_window_title("Cum_AUC_comp")
    plt.show()
    return wt_hit

# diff_AUC_prestim_df = diff_AUC_stim_prestim(activity_long_df[activity_long_df.ID != 4456])


def population_SNR(mean_activity_df):
    """Is the SNR the same across trials, is there a change in SNR between hits and miss trials that could explain the detection"""
    rows = []
    for rec_id in mean_activity_df.ID.unique():
        # Building a Dataframe with one row per trial and one column per neuron
        rec_data_long = mean_activity_df[(mean_activity_df["ID"] == rec_id) & (mean_activity_df["Neuron"].str.startswith("EXC"))].copy()
        genotype = rec_data_long.Genotype.values[0]
        threshold = rec_data_long.Threshold.values[0]
        rec_data = rec_data_long.drop(columns=["Genotype", "Threshold", "ID", "FA"]).pivot(columns="Neuron", values="Stim_mean",
                                                                               index=["Trial", "Amplitude", "Behavior"]).reset_index()
        # TODO: Try to sort the columns
        # sorted_idx = sorted(grouped_hit_data.index, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
        # grouped_hit_data = grouped_hit_data.loc[sorted_idx]
        no_go = rec_data[rec_data.Amplitude == 0]
        trials = rec_data[rec_data.Amplitude != 0]
        # === Computing the baseline references (neurons' average responsane and variance) ===
        mean_nogo = no_go.drop(columns=["Trial", "Amplitude", "Behavior"]).to_numpy().mean(axis=0)
        std_nogo = no_go.drop(columns=["Trial", "Amplitude", "Behavior"]).to_numpy().std(axis=0)
        cov_nogo = np.cov(no_go.drop(columns=["Trial", "Amplitude", "Behavior"]).to_numpy(), rowvar=False)
        # Adding a small regularization factor to be able to invert the matrix
        cov_nogo += np.eye(cov_nogo.shape[0]) * 1e-6
        inv_cov_nogo = np.linalg.inv(cov_nogo)
        # === Computing each trial distance from the baseline ===
        def mahalanobis_distance(x, mu, inv_cov):
            delta = x - mu
            return np.sqrt(delta.T @ inv_cov @ delta)
        for _, trial in trials.iterrows():
            amp = trial.Amplitude
            behavior = trial.Behavior
            vector = trial.drop(labels=["Trial", "Amplitude", "Behavior"])
            mah_dist = mahalanobis_distance(vector, mean_nogo, inv_cov_nogo)
            adj_eucl_dist = np.mean(np.abs(vector - mean_nogo)/std_nogo)
            rows.append({"ID": rec_id, "Genotype": genotype, "Threshold": threshold, "Amplitude": amp, "Behavior": behavior, "Mahalanobis": mah_dist, "Euclidean_adjusted": adj_eucl_dist})
    data = pd.DataFrame(rows)
    # plotting the difference between hit and miss trials for each mouse
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    for distance, ylim in zip(["Mahalanobis", "Euclidean_adjusted"], [[-50000, 500000], [-5, 25]]):
        fig_recs, axes_recs = plt.subplots(nrows=4, ncols=6, figsize=(20, 15), constrained_layout=True)
        ax_recs = axes_recs.flatten()
        for i, rec_id in enumerate(data.ID.unique()):
            hit_data = data[(data["ID"] == rec_id) & (data["Behavior"] == True)][distance].values
            miss_data = data[(data["ID"] == rec_id) & (data["Behavior"] == False)][distance].values
            geno = data[data["ID"] == rec_id].Genotype.values[0]
            thre = data[data["ID"] == rec_id].Threshold.values[0]
            ppt.boxplot(ax_recs[i], hit_data, miss_data, ylabel=f"{distance} distance", paired=False, title=f"{rec_id} - {thre}",
                        ylim=[], colors=color_dict[geno], det_marker=False, force_markers_identity=False)
        fig_recs.suptitle(f"Comparison of the {distance} distance to no-go trials between hit ans miss trials", fontsize=12)
        fig_recs.canvas.manager.set_window_title(f"Recs {distance} distance")
        # Plotting the difference in mean distance between genotypes
        grouped_data = data.groupby(["ID", "Genotype", "Threshold", "Behavior"], as_index=False).mean()
        fig_comp, ax_comp = plt.subplots(nrows=2, ncols=3, figsize=(18, 16), constrained_layout=True)
        wt_hit = grouped_data[(grouped_data.Genotype == "WT") & (grouped_data.Behavior == True)][distance].values
        wt_miss = grouped_data[(grouped_data.Genotype == "WT") & (grouped_data.Behavior == False)][distance].values
        hypo_hit = grouped_data[(grouped_data.Genotype == "KO-Hypo") & (grouped_data.Behavior == True)][distance].values
        hypo_miss = grouped_data[(grouped_data.Genotype == "KO-Hypo") & (grouped_data.Behavior == False)][distance].values
        ppt.boxplot(ax_comp[0, 0], wt_hit, wt_miss, ylabel=f"{distance} distance", paired=True, title=f"WT Hit/Miss",
                    ylim=ylim, colors=color_dict["WT"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_comp[1, 0], hypo_hit, hypo_miss, ylabel=f"{distance} distance", paired=True, title=f"KO-Hypo Hit/Miss",
                    ylim=ylim, colors=color_dict["KO-Hypo"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_comp[0, 1], wt_hit, hypo_hit, ylabel=f"{distance} distance", paired=False, title=f"Hit",
                    ylim=ylim, colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_comp[1, 1], wt_miss, hypo_miss, ylabel=f"{distance} distance", paired=False, title=f"Miss",
                    ylim=ylim, colors=[color_dict["WT"][1], color_dict["KO-Hypo"][1]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax_comp[0, 2], wt_hit - wt_miss, hypo_hit - hypo_miss, ylabel=f"{distance} distance", paired=False, title=f"Delta comparison",
                    ylim=ylim, colors=[color_dict["WT"][0], color_dict["KO-Hypo"][0]], det_marker=False, force_markers_identity=False)

        fig_comp.suptitle(f"Comparison of the mean {distance} distance per animal", fontsize=12)
        fig_comp.canvas.manager.set_window_title(f"Mean {distance} distance")
        # Plotting the mean distance per amplitude
        amp_data = data.groupby(["ID", "Genotype", "Threshold", "Amplitude", "Behavior"], as_index=False).mean()
        fig_curv, ax_curv = plt.subplots(nrows=2, ncols=1, figsize=(10, 16), constrained_layout=True)
        ppt.curveplot(ax_curv[0], amp_data[amp_data["Behavior"] == True], between="Genotype", within="Amplitude", variable=distance,
                      data_points=True, title="Hit trials", ylabel=None, xlabel=None, ylim=None,
                      colors=[ppt.ko_color, ppt.hypo_color, ppt.wt_color], id_display=True,
                      legend_display=True, qq_show=True, transformation=None, consider_normality=True, consider_homogeneity=False)
        # ppt.curveplot(ax_curv[1], amp_data[amp_data["Behavior"] == False], between="Genotype", within="Amplitude", variable=distance,
        #               data_points=True, title="Miss trials", ylabel=None, xlabel=None, ylim=None,
        #               colors=[ppt.ko_color, ppt.hypo_color, ppt.wt_color], id_display=True,
        #               legend_display=True, qq_show=True, transformation=None, consider_normality=False, consider_homogeneity=False)
        fig_curv.suptitle(f"Mean {distance} distance per trial type per animal across amplitudes")
        fig_curv.canvas.manager.set_window_title(f"Amp {distance} distance")
    plt.show()
    return amp_data

# pop_snr_df = population_SNR(activity_long_dff)


def overall_zscore_comp(mean_activity_df):
    """Compares the sum of the zscore of all neurons from trial to trial"""
    pass


# endregion ============================================================================================================
# region ========================================== SNR ==============================================================
def single_neuron_SNR(activity_df, n_type="EXC", amplitude="all", gp1="WT", gp2="KO-Hypo", filter_out_nr=False):
    """Computes and plot the SNR at the single neuron level using 2 different methods: prestim or mean NoGo as reference"""
    full_data = activity_df.copy()
    # Neuron type selection
    if n_type in ["EXC", "INH"]:
        full_data = full_data[full_data["Neuron"].str.startswith(n_type)]
    # Amplitude selection
    if amplitude == "all":
        data = full_data[full_data["Amplitude"] != 0]
    elif amplitude == "threshold":
        data = full_data[full_data["Amplitude"] == full_data["Threshold"]]
    elif isinstance(amplitude, list):
        data = full_data[full_data["Amplitude"].isin(amplitude)]
    # Non-responsive neurons filtering out
    data = data.groupby(['ID', 'Neuron']).filter(lambda g: g['Resp'].ne(0).any()).reset_index(drop=True)
    # Isolate no go trials, remove them form activity and computing their mean used as the reference, merging the value back to the dataframe
    all_no_go_long = full_data[full_data["Amplitude"] == 0]
    all_no_go = all_no_go_long.drop(columns=["Behavior", "FA"]).groupby(["Genotype", "ID", "Neuron"], as_index=False).mean().rename(columns={"Stim_mean": "NoGo_mean"})
        # Splitting FA/CR
    no_go = all_no_go_long.drop(columns=["Behavior"]).groupby(["Genotype", "ID", "Neuron", "FA"], as_index=False).mean().rename(columns={"Stim_mean": "NoGo_mean"})
    fa = no_go[no_go["FA"] == True].copy().rename(columns={"NoGo_mean": "FA_mean"})
    cr = no_go[no_go["FA"] == False].copy().rename(columns={"NoGo_mean": "CR_mean"})
        # Merging the mean activtiy of each neuron during No-Gos to the Dataframe
    data = data.merge(all_no_go[["ID", "Neuron", "NoGo_mean"]], how="left", on=["ID", "Neuron"])
    data = data.merge(fa[["ID", "Neuron", "FA_mean"]], how="left", on=["ID", "Neuron"])
    data = data.merge(cr[["ID", "Neuron", "CR_mean"]], how="left", on=["ID", "Neuron"])
    # Computing the SNR
    data = data.rename(columns={"Abs_Diff_mean": "SNR_prestim"}).drop(columns=["FA"])
    data["SNR_NoGo"] = abs(data["Stim_mean"] - data["NoGo_mean"])
    data["SNR_FA"] = abs(data["Stim_mean"] - data["FA_mean"])
    data["SNR_CR"] = abs(data["Stim_mean"] - data["CR_mean"])
        # Averaging the values per animal
    gp_data = data.drop(columns=["Neuron"]).groupby(["Genotype", "ID", "Behavior"], as_index=False).mean()
    gp_data_all = data.drop(columns=["Neuron", "Behavior"]).groupby(["Genotype", "ID"], as_index=False).mean()
    # Testing for outliers
    mad_threshold = 20
    stim = gp_data["Stim_mean"].values
    med = np.median(stim)
    mad = np.median(np.abs(stim - med))
    if mad == 0:
        print("MAD is zero: cannot compute modified z-scores. Skipping outlier filtering.")
    else:
        modified_z = 0.6745 * (stim - med) / mad
        gp_data["mod_z"] = modified_z
        outlier_mask = np.abs(modified_z) > mad_threshold
        outliers = gp_data.loc[outlier_mask, ["ID", "Behavior", "mod_z"]]
        if not outliers.empty:
            for _, row in outliers.iterrows():
                print(f"Excluding animal ID={row['ID']} (Behavior={row['Behavior']}) with modified_z={row['mod_z']:.2f} > {mad_threshold}")
            # drop all rows for any outlier ID
            exclude_ids = outliers["ID"].unique()
            gp_data = gp_data[~gp_data["ID"].isin(exclude_ids)].copy()
            gp_data_all = gp_data_all[~gp_data_all["ID"].isin(exclude_ids)].copy()
        else:
            print("No outliers detected by MAD test.")
    # Is the SNR prestim/NoGo/FA/CR different between Hit/miss and gp1/KO
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 24), constrained_layout=True)
    for row, SNR_variable in enumerate(["SNR_prestim", "SNR_NoGo"]):
        gp1_hit_raw = gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == True)][SNR_variable]
        gp1_miss_raw = gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == False)][SNR_variable]
        gp2_hit_raw = gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == True)][SNR_variable]
        gp2_miss_raw = gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == False)][SNR_variable]
        gp1_hit = gp1_hit_raw.values[np.isfinite(gp1_hit_raw)]
        gp1_miss = gp1_miss_raw.values[np.isfinite(gp1_miss_raw)]
        gp2_hit = gp2_hit_raw.values[np.isfinite(np.array(gp2_hit_raw))]
        gp2_miss = gp2_miss_raw.values[np.isfinite(np.array(gp2_miss_raw))]

        gp1_hit_ids = set(np.array(gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == True)]["ID"]))
        gp1_miss_ids = set(np.array(gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == False)]["ID"]))
        gp2_hit_ids = set(np.array(gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == True)]["ID"]))
        gp2_miss_ids = set(np.array(gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == False)]["ID"]))

        gp1_mask = gp1_hit_ids & gp1_miss_ids
        gp2_mask = gp2_hit_ids & gp2_miss_ids
        gp1_hit_paired = gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == True) & gp_data["ID"].isin(gp1_mask)][SNR_variable].values
        gp1_miss_paired = gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == False) & gp_data["ID"].isin(gp1_mask)][SNR_variable].values
        gp2_hit_paired = gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == True) & gp_data["ID"].isin(gp2_mask)][SNR_variable].values
        gp2_miss_paired = gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == False) & gp_data["ID"].isin(gp2_mask)][SNR_variable].values

        # All trials
        ppt.boxplot(ax[row, 0],
                    gp_data_all[gp_data_all["Genotype"] == gp1][SNR_variable].values,
                    gp_data_all[gp_data_all["Genotype"] == gp2][SNR_variable].values,
                    paired=False, ylabel=SNR_variable, ylim=[0, 2], title="All trials",
                    colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
        # Hits
        ppt.boxplot(ax[row, 1], gp1_hit, gp2_hit, paired=False, ylabel=SNR_variable, ylim=[0, 2], title="Hit",
                    colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
        # Miss
        ppt.boxplot(ax[row, 2], gp1_miss, gp2_miss, paired=False, ylabel=SNR_variable, ylim=[0, 2], title="Miss",
                    colors=[color_dict[gp1][1], color_dict[gp2][1]], det_marker=False, force_markers_identity=False)
        # gp1 Hit/Miss
        ppt.boxplot(ax[row, 3], gp1_hit_paired, gp1_miss_paired, paired=True, ylabel=SNR_variable, ylim=[0, 2], title=gp1,
                    colors=color_dict[gp1], det_marker=True, force_markers_identity=False)
        # gp2 Hit/Miss
        ppt.boxplot(ax[row, 4], gp2_hit_paired, gp2_miss_paired, paired=True, ylabel=SNR_variable, ylim=[0, 2], title=gp2,
                    colors=color_dict[gp2], det_marker=True, force_markers_identity=False)
    # FA and CR
    ppt.boxplot(ax[2, 0],
                gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == True)]["SNR_FA"].values,
                gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == True)]["SNR_FA"].values,
                paired=False, ylabel="SNR_FA", ylim=[0, 2], title="False Alarm",
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ppt.boxplot(ax[2, 1],
                gp_data[(gp_data["Genotype"] == gp1) & (gp_data["Behavior"] == False)]["SNR_CR"].values,
                gp_data[(gp_data["Genotype"] == gp2) & (gp_data["Behavior"] == False)]["SNR_CR"].values,
                paired=False, ylabel="SNR_CR", ylim=[0, 2], title="Correct Rejection",
                colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
    ax[2, 2].set_axis_off()
    ax[2, 3].set_axis_off()
    ax[2, 4].set_axis_off()
    fig.suptitle(f"Single neuron SNR comparison for {n_type} neurons ({gp1}/{gp2}), {amplitude} amplitude", fontsize=14)
    fig.canvas.manager.set_window_title(f"single_neuron_SNR_{n_type}_{gp1}_{gp2}")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/Zscore_SNR/single_neuron_SNR_{n_type}_{amplitude}_{gp1}_{gp2}.pdf", format="pdf")
    plt.show()
    return gp_data_all, gp_data

single_SNR_df = single_neuron_SNR(activity_long_df, n_type="all", amplitude="all", gp1="WT-BMS", gp2="KO-BMS")

def pop_SNR(activity_df, recruitment_df, amplitude="all", gp1="WT", gp2="KO-Hypo"):
    recruitment = recruitment_df.copy()
    if gp1.split("-")[-1] in ["DMSO", "BMS"]:
        condition = recruitment["Genotype"].astype("string").str.rsplit("-", n=1).str[-1]
        recruitment["ID"] = recruitment["ID"].astype("string") + "-" + condition
    full_data = recruitment.merge(activity_df[["ID", "Trial", "FA"]].drop_duplicates(), how='left', on=["ID", "Trial"]).copy()
    full_data = full_data[full_data["Genotype"].isin([gp1, gp2])]
    # Amplitude selection
    if amplitude == "all":
        all_data = full_data[full_data["amplitude"] != 0]
    elif amplitude == "threshold":
        all_data = full_data[full_data["amplitude"] == full_data["Threshold"]]
    elif isinstance(amplitude, list):
        all_data = full_data[full_data["amplitude"].isin(amplitude)]
    # No-Go
    all_no_go_long = full_data[full_data["amplitude"] == 0]
    all_no_go = all_no_go_long.drop(columns=["behavior", "FA"]).groupby(["Genotype", "ID"], as_index=False).mean()
    # Splitting FA/CR
    no_go = all_no_go_long.drop(columns=["behavior"]).groupby(["Genotype", "ID", "FA"], as_index=False).mean()
    gp_all_data = all_data.drop(columns=["behavior", "FA"]).groupby(["Genotype", "ID"], as_index=False).mean()
    gp_data = all_data.drop(columns=["FA"]).groupby(["Genotype", "ID", "behavior"], as_index=False).mean()
    # Hit/miss and no-go data frame can be merged:
    # all_no_go with gp_all_data & gp_data with no_go (need to rename either behavior or FA to merge on this column)
        # All trials together
    full_nogo_to_merge = all_no_go.drop(columns=["Genotype", "threshold", "bounded_x0", "Trial", "amplitude"])
    full_nogo_to_merge.columns = [col if col == "ID" else f"{col}_Nogo" for col in full_nogo_to_merge.columns]
    grouped = gp_all_data.merge(full_nogo_to_merge, how="left", on=["ID"]).drop(columns=["bounded_x0", "Trial", "amplitude"])
    for col in grouped.columns:
        if col not in ["ID", "Genotype", "threshold"] and not col.endswith("_Nogo"):
            grouped[f"{col}_SNR"] = grouped[col] / grouped[f"{col}_Nogo"]
        # Hit and miss seprated (FA/CR)
    nogo_to_merge = no_go.drop(columns=["Genotype", "threshold", "bounded_x0", "Trial", "amplitude"]).rename(columns={"FA": "behavior"})
    nogo_to_merge.columns = [col if col in ["ID", "behavior"] else f"{col}_Nogo" for col in nogo_to_merge.columns]
    grouped_behavior = gp_data.merge(nogo_to_merge, how="left", on=["ID", "behavior"]).drop(columns=["bounded_x0", "Trial", "amplitude"])
    for col in grouped_behavior.columns:
        if col not in ["ID", "Genotype", "threshold", "behavior"] and not col.endswith("_Nogo"):
            grouped_behavior[f"{col}_SNR"] = grouped_behavior[col] / grouped_behavior[f"{col}_Nogo"]
        # Hit and miss with all no-go
    grouped_beh_no_go = gp_data.merge(full_nogo_to_merge, how="left", on=["ID"]).drop(columns=["bounded_x0", "Trial", "amplitude"])
    for col in grouped_beh_no_go.columns:
        if col not in ["ID", "Genotype", "threshold", "behavior"] and not col.endswith("_Nogo"):
            grouped_beh_no_go[f"{col}_SNR"] = grouped_beh_no_go[col] / grouped_beh_no_go[f"{col}_Nogo"]
    # Now the Pop SNR can be plotted
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(36, 40), constrained_layout=True)
    # Global pop SNR
    for col_id, rec_metric in enumerate([col for col in grouped.columns if col.endswith("_SNR")]):
        # All trials
        gp1_raw = grouped[grouped["Genotype"] == gp1][rec_metric]
        gp2_raw = grouped[grouped["Genotype"] == gp2][rec_metric]
        grp1 = gp1_raw.values[np.isfinite(gp1_raw)]
        grp2 = gp2_raw.values[np.isfinite(gp2_raw)]
        ppt.boxplot(ax[0, col_id], grp1, grp2, paired=False, ylabel=rec_metric, ylim=[-10, 30], title="All Trials/No-Gos",
                    colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
        # Hit and Miss
        gp1_hit_raw = grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp1) & (grouped_beh_no_go["behavior"] == True)][rec_metric]
        gp1_miss_raw = grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp1) & (grouped_beh_no_go["behavior"] == False)][rec_metric]
        gp2_hit_raw = grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp2) & (grouped_beh_no_go["behavior"] == True)][rec_metric]
        gp2_miss_raw = grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp2) & (grouped_beh_no_go["behavior"] == False)][rec_metric]
        gp1_hit = gp1_hit_raw.values[np.isfinite(gp1_hit_raw)]
        gp1_miss = gp1_miss_raw.values[np.isfinite(gp1_miss_raw)]
        gp2_hit = gp2_hit_raw.values[np.isfinite(np.array(gp2_hit_raw))]
        gp2_miss = gp2_miss_raw.values[np.isfinite(np.array(gp2_miss_raw))]

        gp1_hit_id = set(np.array(grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp1) & (grouped_beh_no_go["behavior"] == True) & (np.isfinite(grouped_beh_no_go[rec_metric]))]["ID"]))
        gp1_miss_ids = set(np.array(grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp1) & (grouped_beh_no_go["behavior"] == False) & (np.isfinite(grouped_beh_no_go[rec_metric]))]["ID"]))
        gp2_hit_ids = set(np.array(grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp2) & (grouped_beh_no_go["behavior"] == True) & (np.isfinite(grouped_beh_no_go[rec_metric]))]["ID"]))
        gp2_miss_ids = set(np.array(grouped_beh_no_go[(grouped_beh_no_go["Genotype"] == gp2) & (grouped_beh_no_go["behavior"] == False) & (np.isfinite(grouped_beh_no_go[rec_metric]))]["ID"]))
        gp1_ids = gp1_hit_id & gp1_miss_ids
        gp2_ids = gp2_hit_ids & gp2_miss_ids

        gp1_hit_paired = grouped_beh_no_go[(grouped_beh_no_go["ID"].isin(gp1_ids)) & (grouped_beh_no_go["behavior"] == True)][rec_metric].values
        gp1_miss_paired = grouped_beh_no_go[(grouped_beh_no_go["ID"].isin(gp1_ids)) & (grouped_beh_no_go["behavior"] == False)][rec_metric].values
        gp2_hit_paired = grouped_beh_no_go[(grouped_beh_no_go["ID"].isin(gp2_ids)) & (grouped_beh_no_go["behavior"] == True)][rec_metric].values
        gp2_miss_paired = grouped_beh_no_go[(grouped_beh_no_go["ID"].isin(gp2_ids)) & (grouped_beh_no_go["behavior"] == False)][rec_metric].values

            # Hit vs Miss
        ppt.boxplot(ax[1, col_id], gp1_hit_paired, gp1_miss_paired, paired=True, ylabel=rec_metric, ylim=[-10, 30], title=f"Hit/NoGo vs. Miss/Nogo ({gp1})",
                    colors=color_dict[gp1], det_marker=True, force_markers_identity=False)
        ppt.boxplot(ax[2, col_id], gp2_hit_paired, gp2_miss_paired, paired=True, ylabel=rec_metric, ylim=[-10, 30], title=f"Hit/NoGo vs. Miss/Nogo ({gp2})",
                    colors=color_dict[gp2], det_marker=True, force_markers_identity=False)
            # Between genotypes
        ppt.boxplot(ax[3, col_id], gp1_hit, gp2_hit, paired=False, ylabel=rec_metric, ylim=[-10, 30], title="Hit/NoGo",
                    colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
        ppt.boxplot(ax[4, col_id], gp1_miss, gp2_miss, paired=False, ylabel=rec_metric, ylim=[-10, 30], title="Miss/NoGo",
                    colors=[color_dict[gp1][1], color_dict[gp2][1]], det_marker=False, force_markers_identity=False)
        # FA and CR
        gp1_fa_raw = grouped_behavior[(grouped_behavior["Genotype"] == gp1) & (grouped_behavior["behavior"] == True)][rec_metric]
        gp1_cr_raw = grouped_behavior[(grouped_behavior["Genotype"] == gp1) & (grouped_behavior["behavior"] == False)][rec_metric]
        gp2_fa_raw = grouped_behavior[(grouped_behavior["Genotype"] == gp2) & (grouped_behavior["behavior"] == True)][rec_metric]
        gp2_cr_raw = grouped_behavior[(grouped_behavior["Genotype"] == gp2) & (grouped_behavior["behavior"] == False)][rec_metric]
        gp1_fa = gp1_fa_raw.values[np.isfinite(gp1_fa_raw)]
        gp1_cr = gp1_cr_raw.values[np.isfinite(gp1_cr_raw)]
        gp2_fa = gp2_fa_raw.values[np.isfinite(gp2_fa_raw)]
        gp2_cr = gp2_cr_raw.values[np.isfinite(gp2_cr_raw)]
        # ppt.boxplot(ax[5, col_id], gp1_fa, gp2_fa, paired=False, ylabel=rec_metric, ylim=[-10, 30], title="Hit/FA",
        #             colors=[ppt.wt_color, ppt.hypo_color], det_marker=True, force_markers_identity=True)
        # ppt.boxplot(ax[6, col_id], gp1_cr, gp2_cr, paired=False, ylabel=rec_metric, ylim=[-10, 30], title="Miss/CR",
        #             colors=[ppt.wt_light_color, ppt.hypo_light_color], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Comparison of the population SNR (mean recruitment) between {gp1} and {gp2}\nAmplitude == {amplitude}", fontsize=12)
    fig.canvas.manager.set_window_title(f"Pop_SNR_comp_{amplitude}_{gp1}_{gp2}")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/Zscore_SNR/pop_SNR_{amplitude}_{gp1}_{gp2}.pdf", format="pdf")
    return grouped, grouped_beh_no_go

pop_SNR_df = pop_SNR(activity_long_df, recruitment_df, amplitude="all", gp1="WT-BMS", gp2="KO-BMS")

def correlate_single_SNR(activity_df, recruitment_df, amplitude="all", gp1="WT", gp2="KO-Hypo", filter_out_nr=False):
    single_all_exc, single_beh_exc = single_neuron_SNR(activity_df, n_type="EXC", amplitude=amplitude, gp1=gp1, gp2=gp2, filter_out_nr=filter_out_nr)
    single_all_inh, single_beh_inh = single_neuron_SNR(activity_df, n_type="INH", amplitude=amplitude, gp1=gp1, gp2=gp2, filter_out_nr=filter_out_nr)
    pop_all, pop_beh = pop_SNR(activity_df, recruitment_df, amplitude=amplitude, gp1=gp1, gp2=gp2)
    # Merging the single neuron SNR to the population SNR df
        # All trials grouped
    single_all_exc.rename(columns={"SNR_NoGo": "single_SNR_exc"}, inplace=True)
    single_all_inh.rename(columns={"SNR_NoGo": "single_SNR_inh"}, inplace=True)
    col_to_keep = ["Genotype", "ID", "Threshold", "rec_EXC_perc", "rec_INH_perc", "rec_EXC_perc_SNR", "rec_INH_perc_SNR"]
    data_all = pop_all.drop(columns=[col for col in pop_all.columns if col not in col_to_keep])
    data_all = data_all.merge(single_all_exc[["ID", "single_SNR_exc"]], how="left", on="ID")
    data_all = data_all.merge(single_all_inh[["ID", "single_SNR_inh"]], how="left", on="ID")
        # Hit and Miss independently
    single_beh_exc.rename(columns={"SNR_NoGo": "single_SNR_exc"}, inplace=True)
    single_beh_inh.rename(columns={"SNR_NoGo": "single_SNR_inh"}, inplace=True)
    pop_beh.rename(columns={"behavior": "Behavior"}, inplace=True)
    print(pop_beh.columns)
    col_to_keep_beh = ["Genotype", "ID", "Behavior", "Threshold", "rec_EXC_perc", "rec_INH_perc", "rec_EXC_perc_SNR", "rec_INH_perc_SNR"]
    data_beh = pop_beh.drop(columns=[col for col in pop_beh.columns if col not in col_to_keep_beh])
    data_beh = data_beh.merge(single_beh_exc[["ID", "Behavior", "single_SNR_exc"]], how="left", on=["ID", "Behavior"])
    data_beh = data_beh.merge(single_beh_inh[["ID", "Behavior", "single_SNR_inh"]], how="left", on=["ID", "Behavior"])
    data_hit = data_beh[data_beh.Behavior == True]
    data_miss = data_beh[data_beh.Behavior == False]
    # Doing the correlation
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(24, 18), constrained_layout=True)
    beh_list = ["All", "Hit", "Miss"]
    for row, data in enumerate([data_all, data_hit, data_miss]):
        col_id = 0
        for single_col, pop_cols_list in zip(["single_SNR_exc", "single_SNR_inh"], [["rec_EXC_perc", "rec_EXC_perc_SNR"],
                                                                                    ["rec_INH_perc", "rec_INH_perc_SNR"]]):
            for pop_col in pop_cols_list:
                x = data[single_col]
                y = data[pop_col]
                results = dict(linregress(x, y)._asdict())
                r2 = results["rvalue"] ** 2
                line = results["slope"] * x + results["intercept"]
                # Plot the data points and regression line
                ax[row, col_id].plot(x, line, color="black", lw=2)
                ax[row, col_id].text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}",
                                     transform=ax[row, col_id].transAxes, fontsize=8,
                                     verticalalignment="top", color="black")
                ax[row, col_id].set_title(beh_list[row])
                ax[row, col_id].set_xlabel(single_col, fontsize=10)
                ax[row, col_id].set_ylabel(pop_col, fontsize=10)
                ax[row, col_id].spines[["bottom", "left"]].set_linewidth(2)
                ax[row, col_id].spines[["top", "right"]].set_visible(False)
                offset = 0.1
                for gp in [gp1, gp2]:
                    x = data[data.Genotype == gp][single_col]
                    y = data[data.Genotype == gp][pop_col]
                    results = dict(linregress(x, y)._asdict())
                    r2 = results["rvalue"] ** 2
                    line = results["slope"] * x + results["intercept"]
                    # Plot the data points and regression line
                    ax[row, col_id].plot(x, line, color=color_dict[gp][0], lw=2)
                    ax[row, col_id].scatter(x, y, color=color_dict[gp][0], alpha=0.7, s=10, marker="+")
                    ax[row, col_id].text(0.05, 0.95 - offset, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}",
                                         transform=ax[row, col_id].transAxes, fontsize=8,
                                         verticalalignment="top", color=color_dict[gp][0])
                    offset += 0.1
                col_id +=1
    fig.suptitle(f"Correlation of single neuron SNR with the number of recruited neurons and population snr", fontsize=12)
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/Zscore_SNR/Corr_single_SNR_{gp1}_{gp2}.pdf", format="pdf")
    plt.show()
    return data_all, data_beh

# sing_pop_SNR_corr = correlate_single_SNR(activity_long_df, recruitment_df, amplitude="all", gp1="WT", gp2="KO-Hypo", filter_out_nr=True)
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

    # for rec in recs.values():
    #     rec.auc()
    # endregion
    # region ====== Comparison of threshold to session threshold ======
    rows = []
    for rec in recs.values():
        rec.auc()
        rec.auc_neg()
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.threshold,
                     "session_threshold": rec.session_threshold, "session_x0": rec.x0_psy})
    session_threshold = pd.DataFrame(rows)

    # from percephone.utils.math_formulas import sigmoid_fit
    # fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(20, 12), constrained_layout=True)
    # axs = ax.flatten()
    # for i, rec in enumerate(recs.values()):
    #     axs[i].set_title(f"{rec.filename} {rec.genotype}- {rec.threshold}/{rec.session_threshold}({rec.x0_psy:.2f})", fontsize=12)
    #     axs[i].set_ylim(0, 1)
    #     axs[i].scatter(np.arange(start=2, stop=13, step=2), rec.hit_rates[1:], s=5)
    #     if rec.filename == 7554 and rec.genotype == "KO-DMSO":
    #         x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates, p0=[4.0, 1.0])
    #     else:
    #         x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates)
    #     axs[i].plot(x, y, color='red', lw=2, alpha=0.75)
    # plt.show()
    # endregion

    # region ====== TBT variability ======
    recruitment_df = get_features(recs.values(), amp_delay=False, auc=False)
    # hit_test, hit_posthoc, miss_test, miss_posthoc = compare_tbt_var_per_amp(recruitment_df)
    activity_long_df = get_mean_trial_activity_df(recs.values(), zscore=True, BMS=BMS_analysis)
    activity_long_dff = get_mean_trial_activity_df(recs.values(), zscore=False, BMS=BMS_analysis)

    # Sensitivity
    gp1 = "WT"
    gp2 = "KO-Hypo"
    sens_data = compute_neuronal_sensitivity(activity_long_df, amplitude="All", n_type="EXC", filter_out_nr=True, gp1=gp1, gp2=gp2)
    # sens_export = sens_data[sens_data["Amplitude"] == "All"]
    # sens_export = sens_export[sens_export["Genotype"].isin([gp1, gp2])].sort_values("Genotype")

    # ntn_df, ntn_snr_df = ntn_cosine_similarity(activity_long_df, amplitude="all", nogo=False, metric="Resp", filter_out_nr=True, n_type="INH", gp1="WT", gp2="KO-Hypo")
    # activity_long_dff = get_mean_trial_activity_df(recs.values(), zscore=False)
    # prestim_df = prestim_activated_neurons(filtered_activity_df)
    # prestim_vector_df = prestim_act_vector(activity_long_df, metric="Stim_mean", hit_activated_only=False)
    # prestim_vector_dff = prestim_act_vector(activity_long_dff, metric="Diff_mean", hit_activated_only=True)
    # tbt_recr_var_df = compare_tbt_var_per_amp(recruitement_df)
    # pca_df = pca(activity_long_df, split="Miss_CR")

    sing_pop_SNR_corr, sing_pop_SNR_corr_beh = correlate_single_SNR(activity_long_df, recruitment_df, amplitude="all", gp1="WT-BMS", gp2="KO-BMS", filter_out_nr=True)

    # pop_SNR_df_all, pop_snr_df_behavior = pop_SNR(activity_long_df, recruitment_df, amplitude="all", gp1="WT", gp2="KO-Hypo")
    # single_SNR_all_df, single_SNR_df = single_neuron_SNR(activity_long_df, n_type="EXC", amplitude="all", gp1="WT", gp2="KO-Hypo", filter_out_nr=True)
    # single_SNR_df = single_neuron_SNR(activity_long_dff, n_type="EXC", amplitude="all", gp1="WT-DMSO", gp2="KO-DMSO")

    # nb_reliable_df = compare_nb_reliable_responders(recs.values())
    # reliable_activity_df = filter_reliable(activity_long_df[~activity_long_df["ID"].isin([6606, 6611])], recs.values(),
    #                                        pattern="act", get_non_reliable=False) #6606 and 6611 only 1 reliable EXC
    # non_reliable_activity_df = filter_reliable(activity_long_df[~activity_long_df["ID"].isin([6606, 6611])], recs.values(),
    #                                            pattern="act", get_non_reliable=True)
    # threshold_recruited_df = filter_stim_recruited_neurons(activity_long_df, det_filter=None, amplitude_filter="threshold", recruitment_filter="recruited", n_type_filter=None)
    # no_go_non_recruited_df = filter_stim_recruited_neurons(activity_long_df, det_filter=None, amplitude_filter=[0], recruitment_filter="recruited", n_type_filter=None, get_opposite=True)
    # cosine_sim_df = compare_threshold_trials_cosine_similarity(activity_long_df[~activity_long_df["ID"].isin([5873, 4745, 6606, 6601])], include_gaba=False,
    #                                                            title_precision="All neurons")
    # cosine_sim_df = compare_threshold_trials_cosine_similarity(no_go_non_recruited_df[~no_go_non_recruited_df["ID"].isin([5873, 4745, 6606, 6601])], include_gaba=False,
    #                                                            title_precision="No-go non recruited neurons")
    # without_12 = activity_long_df[activity_long_df["Threshold"] != 12]
    # cosine_sim_df = compare_threshold_trials_cosine_similarity(without_12, include_gaba=False, title_precision="All neurons")
    # cosine_sim_df = compare_threshold_trials_cosine_similarity(activity_long_df, include_gaba=False, title_precision="",
    #                                                            amps={"WT": "supra_threshold", "KO-Hypo": [12]}, centroid=False,
    #                                                            min_nb_trials=2)
    # global_cos_df = compare_global_cosine_sim(activity_long_df, include_gaba=False, centroid=False)
    # nb_trials_df = compare_nb_trials_wt_ko(activity_long_df)
    # endregion

    # region ====== Pre-stimulus ======
    # bsl, hit, miss, hit_snr, miss_snr = baseline_and_SNR(recruitment_df)
    # filtered_activity_df = filter_stim_recruited_neurons(activity_long_dff[activity_long_dff["ID"] != 4456])
    # filtered_activity_df = filter_stim_recruited_neurons(activity_long_dff)
    # grouped_comp, gp_hit, gp_miss = prestim_comp(filtered_activity_df)
    # snr_data = snr_comp(filtered_activity_df)
    # snr_wt, snr_hypo = compare_to_wt_threshold(filtered_activity_df)
    # prestim_act_df = prestim_influence_neuron_activation(filtered_activity_df)
    # endregion

    # region ====== E/I Ratio ======
    # ei_df = get_ei_ratio_df(recs.values())
    # ei_var_df = plot_ei_variance(ei_df)
    # ei_behavior_df = correlate_behavior(ei_df, column="EI_ratio_cste")
    # ei_comp_df = compare_ei_ratio(ei_df, column="EI_ratio_norm")

    # ei_data = population_EI_ratio(recs.values(), pyr_inhibition=False, amp="all", test_INH=False, gp1="WT", gp2="KO-Hypo")
    # ntn_ei_cor_df = correlate_ntn_ei(ntn_df, ei_data, corr_threshold_ei=True)
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

    # rows = []
    # for rec in recs.values():
    #     hit = np.sum((rec.stim_ampl == (rec.session_threshold + 2)) & rec.detected_stim)
    #     miss = np.sum((rec.stim_ampl == (rec.session_threshold + 2)) & ~rec.detected_stim)
    #     rows.append({"ID": rec.filename, "Genotype": rec.genotype, "Threshold": rec.session_threshold, "T+2_hit": hit,
    #                  "T+2_miss": miss})
    # supra_det_df = pd.DataFrame(rows)
    #
    # mean_trials = activity_long_df.copy().drop(columns=["Neuron", "FA"]).groupby(
    #     ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Behavior"], as_index=False).mean()
    # count_trials = mean_trials.groupby(["Genotype", "ID", "Threshold", "Amplitude", "Behavior"], as_index=False).count()
    # mean_count = count_trials.drop(columns=["Threshold"]).groupby(["Genotype", "Amplitude", "Behavior"],
    #                                                               as_index=False).mean().drop(columns=["ID"])

    # === Plotting the number of hits and miss per amplitude per genotype ===
    # 1) define ordering and bar geometry
    # ampls = [i for i in range(2, 13, 2)]  # [0,2,4,…,12]
    # genos = ['WT', 'KO', 'KO-Hypo']
    # labels = [True, False]
    # nA, nG, nL = len(ampls), len(genos), len(labels)
    # bar_w = 0.8 / (nG * nL)  # total cluster width ~0.8
    #
    # # 2) color maps (solid for Hit, lighter for Miss)
    # gen_colors = {'WT': ppt.wt_color, 'KO': ppt.ko_color, 'KO-Hypo': ppt.hypo_color}
    # gen_colors_light = {'WT': ppt.wt_light_color, 'KO': ppt.ko_light_color, 'KO-Hypo': ppt.hypo_light_color}
    #
    # # 3) x‐positions
    # x = np.arange(nA)
    #
    # fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    #
    # for gi, geno in enumerate(genos):
    #     for li, label in enumerate(labels):
    #         # pick your subset
    #         sub = mean_count[(mean_count['Genotype'] == geno) & (mean_count['Behavior'] == label)]
    #         # ensure it’s in amplitude order
    #         means = [sub.loc[sub['Amplitude'] == amp, 'Trial'].values[0] for amp in ampls]
    #         # compute offsets so clusters are centered at x
    #         offset = (gi * nL + li) * bar_w - 0.5 * (nG * nL * bar_w - bar_w)
    #         xpos = x + offset
    #         # choose color
    #         color = gen_colors[geno] if label == True else gen_colors_light[geno]
    #         ax.bar(xpos, means, bar_w, label=f"{geno} {label}", color=color)
    #
    # # 4) polish
    # ax.set_xticks(x)
    # ax.set_xticklabels(ampls)
    # ax.set_xlabel("Stimulation amplitude (µm)", fontsize=10)
    # ax.set_ylabel("Mean nb trials", fontsize=10)
    # ax.set_title("Hit vs Miss trial counts per amplitude & genotype", fontsize=12)
    # plt.show()
