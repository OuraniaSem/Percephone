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
                           "Behavior": behavior_vector[trial_id],
                           "FA": fa_vector[trial_id], "Neuron": f"{neuron_type}_{n_id}",
                           "Resp": resp_mat[n_id, trial_id],
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


def compute_neuronal_sensitivity(data_df, amplitude="All", n_type="all"):
    data = data_df.drop(columns=["Resp"])
    # For each recording
    rows = []
    for rec_id in data["ID"].unique():
        rec_full_data = data[data["ID"] == rec_id]
        genotype = rec_full_data["Genotype"].values[0]
        threshold = rec_full_data["Threshold"].values[0]
        # Taking all amplitudes together
        rec_data = rec_full_data[rec_full_data["Amplitude"] != 0].drop(columns=["FA"])
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
    data["Sensitivity_exc"] = data.apply(lambda row: row["Sensitivity"][: row["n_EXC"]], axis=1)
    data["Sensitivity_inh"] = data.apply(lambda row: row["Sensitivity"][row["n_EXC"]:], axis=1)
    data["Mean_sensitivity"] = data["Sensitivity"].apply(lambda lst: np.mean(lst))
    data["Mean_sensitivity_exc"] = data["Sensitivity_exc"].apply(lambda lst: np.mean(lst))
    data["Mean_sensitivity_inh"] = data["Sensitivity_inh"].apply(lambda lst: np.mean(lst))
    data["Sensitive_neurons"] = data["Sensitivity"].apply(lambda lst: np.nan if any(pd.isna(v) for v in lst) else sum(v > 1 for v in lst))
    data["Sensitive_neurons_exc"] = data["Sensitivity_exc"].apply(lambda lst: np.nan if any(pd.isna(v) for v in lst) else sum(v > 1 for v in lst))
    data["Sensitive_neurons_inh"] = data["Sensitivity_inh"].apply(lambda lst: np.nan if any(pd.isna(v) for v in lst) else sum(v > 1 for v in lst))
    if n_type == "all":
        sens_col = "Sensitivity"
        mean_sens_col = "Mean_sensitivity"
        nb_sens_col = "Sensitive_neurons"
    elif n_type == "EXC":
        sens_col = "Sensitivity_exc"
        mean_sens_col = "Mean_sensitivity_exc"
        nb_sens_col = "Sensitive_neurons_exc"
    elif n_type == "INH":
        sens_col = "Sensitivity_inh"
        mean_sens_col = "Mean_sensitivity_inh"
        nb_sens_col = "Sensitive_neurons_inh"
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
    # Plotting the number of sensitive neurons (d' > 1) between both genotypes
    fig_stats, axs_stats = plt.subplots(nrows=2, ncols=5, figsize=(24, 16), constrained_layout=True)
    ax_stats = axs_stats.flatten()
    # Plotting the global sensitivity of pulled neurons between both genotypes
    # fig_sens, axs_sens = plt.subplots(nrows=2, ncols=5, figsize=(24, 16), constrained_layout=True)
    # ax_sens = axs_sens.flatten()
    # Plotting the mean sensitivity per animal between both genotypes
    fig_mean, axs_mean = plt.subplots(nrows=2, ncols=5, figsize=(24, 16), constrained_layout=True)
    ax_mean = axs_mean.flatten()
    for i, amp in enumerate(data["Amplitude"].unique()):
        # Number
        ppt.boxplot(ax_stats[i], data[(data["Genotype"] == "WT") & (data["Amplitude"] == amp)][nb_sens_col].values,
                    data[(data["Genotype"] == "KO-Hypo") & (data["Amplitude"] == amp)][nb_sens_col].values,
                    ylabel="Nb neurons with d'>1", paired=False, title=f"Amplitude = {amp}", ylim=[], colors=[ppt.wt_color, ppt.hypo_color],
                    det_marker=False, force_markers_identity=False)
        # Mean
        ppt.boxplot(ax_mean[i], data[(data["Genotype"] == "WT") & (data["Amplitude"] == amp)][mean_sens_col].values,
                    data[(data["Genotype"] == "KO-Hypo") & (data["Amplitude"] == amp)][mean_sens_col].values,
                    ylabel="Mean d'", paired=False, title=f"Amplitude = {amp}", ylim=[],
                    colors=[ppt.wt_color, ppt.hypo_color],
                    det_marker=False, force_markers_identity=False)
        # Global sensitivity
        # wt = data[(data["Genotype"] == "WT") & (data["Amplitude"] == amp)]
        # merged_wt = [x for sublist in wt["Sensitivity_exc"] for x in sublist]
        # hypo = data[(data["Genotype"] == "KO-Hypo") & (data["Amplitude"] == amp)]
        # merged_hypo = [x for sublist in hypo["Sensitivity_exc"] for x in sublist]
        # ppt.boxplot(ax_sens[i], merged_wt, merged_hypo, ylabel="d'", paired=False, title=f"Amplitude = {amp}",
        #             ylim=[], colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
        amp_data = data[data["Amplitude"] == amp].copy()
        amp_long = amp_data.explode(sens_col).rename(columns={sens_col: "neuron_sens"})
        amp_long["neuron_sens"] = pd.to_numeric(amp_long["neuron_sens"])
        amp_long["Genotype"] = pd.Categorical(amp_long["Genotype"], categories=["WT", "KO", "KO-Hypo"])
        amp_long = amp_long.dropna(subset=["neuron_sens"])
        model = smf.mixedlm("neuron_sens ~ Genotype", amp_long, groups=amp_long["ID"])
        result = model.fit()
        print(f"=== === === Amplitude: {amp} === === ===")
        print(result.summary())

    # Number - threshold
    ppt.boxplot(ax_stats[8], data[(data["Genotype"] == "WT") & (data["Amplitude"] == data["Threshold"])][nb_sens_col].values,
                data[(data["Genotype"] == "KO-Hypo") & (data["Amplitude"] == data["Threshold"])][nb_sens_col].values,
                ylabel="Nb neurons with d'>1", paired=False, title=f"Amplitude = Threshold", ylim=[],
                colors=[ppt.wt_color, ppt.hypo_color],
                det_marker=False, force_markers_identity=False)
    # Mean - threshold
    ppt.boxplot(ax_mean[8],
                data[(data["Genotype"] == "WT") & (data["Amplitude"] == data["Threshold"])][mean_sens_col].values,
                data[(data["Genotype"] == "KO-Hypo") & (data["Amplitude"] == data["Threshold"])][mean_sens_col].values,
                ylabel="Mean d'", paired=False, title=f"Amplitude = Threshold", ylim=[],
                colors=[ppt.wt_color, ppt.hypo_color],
                det_marker=False, force_markers_identity=False)
    # Global sensitivity - threshold
    thre_data = data[data["Amplitude"] == data["Threshold"]].copy()
    thre_long = thre_data.explode(sens_col).rename(columns={sens_col: "neuron_sens"})
    thre_long["neuron_sens"] = pd.to_numeric(thre_long["neuron_sens"])
    thre_long["Genotype"] = pd.Categorical(thre_long["Genotype"], categories=["WT", "KO", "KO-Hypo"])
    thre_long = thre_long.dropna(subset=["neuron_sens"])
    model_thre = smf.mixedlm("neuron_sens ~ Genotype", thre_long, groups=thre_long["ID"])
    result_thre = model_thre.fit()
    print(f"=== === === Amplitude: Threshold === === ===")
    print(result_thre.summary())
    # wt = data[(data["Genotype"] == "WT") & (data["Amplitude"] == data["Threshold"])]
    # merged_wt = [x for sublist in wt[sens_col] for x in sublist]
    # hypo = data[(data["Genotype"] == "KO-Hypo") & (data["Amplitude"] == data["Threshold"])]
    # merged_hypo = [x for sublist in hypo[sens_col] for x in sublist]
    # ppt.boxplot(ax_sens[8], merged_wt, merged_hypo, ylabel="Nb neurons with d'>1", paired=False, title=f"Amplitude = Threshold",
    #             ylim=[], colors=[ppt.wt_color, ppt.hypo_color], det_marker=False, force_markers_identity=False)
    fig_stats.suptitle(f"Comparison between genotypes of the number of sensitive neurons for different amplitudes", fontsize=14)
    fig_stats.canvas.manager.set_window_title(f"Sensitivity_comp")
    # fig_sens.suptitle(f"Comparison all mice grouped {n_type} neuron's sensitivity for different amplitudes", fontsize=14)
    # fig_sens.canvas.manager.set_window_title(f"Grouped_neuron_sens_comp")
    fig_mean.suptitle(f"Comparison mean {n_type} sensitivity per mouse for different amplitudes", fontsize=14)
    fig_mean.canvas.manager.set_window_title(f"Mean_sens_comp")
    plt.show()
    return data


sensitivity_df = compute_neuronal_sensitivity(activity_long_df, amplitude="All", n_type="EXC")


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
    plt.savefig(
        "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/Figures_april_2025/Pre-stim_comp_(act_vs_non_act).pdf")
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
        hit_exc = np.mean(
            np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, (rec.detected_stim & threshold_stim_vector)] == 1,
                             axis=0)) / n_exc
        miss_exc = np.mean(
            np.count_nonzero(rec.matrices["EXC"]["Responsivity"][:, (~rec.detected_stim & threshold_stim_vector)] == 1,
                             axis=0)) / n_exc
        hit_inh = np.mean(np.count_nonzero(
            rec.matrices[I_type]["Responsivity"][:, (rec.detected_stim & threshold_stim_vector)] == I_activity,
            axis=0)) / I_nb
        miss_inh = np.mean(np.count_nonzero(
            rec.matrices[I_type]["Responsivity"][:, (~rec.detected_stim & threshold_stim_vector)] == I_activity,
            axis=0)) / I_nb
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
    ppt.boxplot(ax[0], wt_hit_un, ko_hit_un, paired=False, ylabel="E:I ratio", ylim=[], title="Hit trials",
                colors=[ppt.wt_color, ppt.hypo_color], det_marker=False)
    ppt.boxplot(ax[1], wt_miss_un, ko_miss_un, paired=False, ylabel="E:I ratio", ylim=[], title="Miss trials",
                colors=[ppt.wt_light_color, ppt.hypo_light_color], det_marker=False)
    ppt.boxplot(ax[2], wt_hit_paired, wt_miss_paired, paired=True, ylabel="E:I ratio", ylim=[], title="WT",
                colors=[ppt.wt_color, ppt.wt_light_color], det_marker=True)
    ppt.boxplot(ax[3], ko_hit_paired, ko_miss_paired, paired=True, ylabel="E:I ratio", ylim=[], title="KO-Hypo",
                colors=[ppt.hypo_color, ppt.hypo_light_color], det_marker=True)
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


# endregion ============================================================================================================


if __name__ == '__main__':
    BMS_analysis = False
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
    recruitment_df = get_features(recs.values(), amp_delay=False)
    hit_test, hit_posthoc, miss_test, miss_posthoc = compare_tbt_var_per_amp(recruitment_df)
    activity_long_df = get_mean_trial_activity_df(recs.values(), zscore=True)
    # activity_long_dff = get_mean_trial_activity_df(recs.values(), zscore=False)
    # prestim_df = prestim_activated_neurons(filtered_activity_df)
    # prestim_vector_df = prestim_act_vector(activity_long_df, metric="Stim_mean", hit_activated_only=False)
    # prestim_vector_dff = prestim_act_vector(activity_long_dff, metric="Diff_mean", hit_activated_only=True)
    # tbt_recr_var_df = compare_tbt_var_per_amp(recruitement_df)
    # pca_df = pca(activity_long_df)

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
    global_cos_df = compare_global_cosine_sim(activity_long_df, include_gaba=False, centroid=False)
    # nb_trials_df = compare_nb_trials_wt_ko(activity_long_df)
    # endregion

    # region ====== Pre-stimulus ======
    bsl, hit, miss, hit_snr, miss_snr = baseline_and_SNR(recruitment_df)
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

    rows = []
    for rec in recs.values():
        hit = np.sum((rec.stim_ampl == (rec.session_threshold + 2)) & rec.detected_stim)
        miss = np.sum((rec.stim_ampl == (rec.session_threshold + 2)) & ~rec.detected_stim)
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "Threshold": rec.session_threshold, "T+2_hit": hit,
                     "T+2_miss": miss})
    supra_det_df = pd.DataFrame(rows)

    mean_trials = activity_long_df.copy().drop(columns=["Neuron", "FA"]).groupby(
        ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Behavior"], as_index=False).mean()
    count_trials = mean_trials.groupby(["Genotype", "ID", "Threshold", "Amplitude", "Behavior"], as_index=False).count()
    mean_count = count_trials.drop(columns=["Threshold"]).groupby(["Genotype", "Amplitude", "Behavior"],
                                                                  as_index=False).mean().drop(columns=["ID"])

    # === Plotting the number of hits and miss per amplitude per genotype ===
    # 1) define ordering and bar geometry
    ampls = [i for i in range(2, 13, 2)]  # [0,2,4,…,12]
    genos = ['WT', 'KO', 'KO-Hypo']
    labels = [True, False]
    nA, nG, nL = len(ampls), len(genos), len(labels)
    bar_w = 0.8 / (nG * nL)  # total cluster width ~0.8

    # 2) color maps (solid for Hit, lighter for Miss)
    gen_colors = {'WT': ppt.wt_color, 'KO': ppt.ko_color, 'KO-Hypo': ppt.hypo_color}
    gen_colors_light = {'WT': ppt.wt_light_color, 'KO': ppt.ko_light_color, 'KO-Hypo': ppt.hypo_light_color}

    # 3) x‐positions
    x = np.arange(nA)

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)

    for gi, geno in enumerate(genos):
        for li, label in enumerate(labels):
            # pick your subset
            sub = mean_count[(mean_count['Genotype'] == geno) & (mean_count['Behavior'] == label)]
            # ensure it’s in amplitude order
            means = [sub.loc[sub['Amplitude'] == amp, 'Trial'].values[0] for amp in ampls]
            # compute offsets so clusters are centered at x
            offset = (gi * nL + li) * bar_w - 0.5 * (nG * nL * bar_w - bar_w)
            xpos = x + offset
            # choose color
            color = gen_colors[geno] if label == True else gen_colors_light[geno]
            ax.bar(xpos, means, bar_w, label=f"{geno} {label}", color=color)

    # 4) polish
    ax.set_xticks(x)
    ax.set_xticklabels(ampls)
    ax.set_xlabel("Stimulation amplitude (µm)", fontsize=10)
    ax.set_ylabel("Mean nb trials", fontsize=10)
    ax.set_title("Hit vs Miss trial counts per amplitude & genotype", fontsize=12)
    plt.show()
