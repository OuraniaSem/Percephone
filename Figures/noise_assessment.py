# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import percephone.core.recording as pc
import percephone.plts.stats as ppt
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
    return pd.DataFrame(rows)


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


def compare_tbt_var_per_amp(df):
    """
    For each recording, compute the standard deviation
    Parameters
    ----------
    df

    Returns
    -------

    """
    rows = []
    # For each recording, we compute the std of nb of recruited neurons per amplitude





# endregion ============================================================================================================
# region ======================================== Pre-stimulus =========================================================


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
    activity_df["Diff_mean"] = activity_df["Stim_mean"] - activity_df["Prestim_mean"]
    activity_df["Diff_std"] = activity_df["Stim_std"] - activity_df["Prestim_std"]
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
    if metric is None:
        metric = activity_df.columns[-1]
    rows = []
    for rec_id in activity_df["ID"].unique():
        rec_data = activity_df[(activity_df["ID"] == rec_id) & (activity_df["Amplitude"] == activity_df["Threshold"])].copy()
        # Keeping only the EXC neurons
        rec_data = rec_data[rec_data["Neuron"].str.startswith("EXC")]
        if hit_activated_only:
            # Selecting the neurons that are activated at least once during detected trials
            act_hit_neurons = set(rec_data[(rec_data["Behavior"] == True) & (rec_data["Resp"] == 1)]["Neuron"].values)
            rec_data[metric] = rec_data[metric].where(rec_data["Neuron"].isin(act_hit_neurons), np.nan)
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
        # Optional: Filter out self-similarity and duplicate pairs
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
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 12), constrained_layout=True)
    for row, genotype in enumerate(results["Genotype"].unique()):
        data = results[results["Genotype"] == genotype].copy()
        ppt.boxplot(ax[row, 0], data["Hit_sim"].values, data["Miss_sim"].values, ylabel="Mean_abs_cos_cim", paired=True, title=f"{genotype} - Hit/Miss", ylim=[],
                    det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 1], data["Hit_sim"].values, data["Between_sim"].values, ylabel="Mean_abs_cos_cim", paired=True, title=f"{genotype} - Hit/btw", ylim=[],
                    det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 2], data["Miss_sim"].values, data["Between_sim"].values, ylabel="Mean_abs_cos_cim", paired=True, title=f"{genotype} - Miss/btw", ylim=[],
                    det_marker=False, force_markers_identity=False)
    plt.suptitle(f"")
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


# endregion ============================================================================================================
# region ========================================== Noise ==============================================================


# endregion ============================================================================================================


if __name__ == '__main__':
    # region ====== Initialisation of recs instances ======
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
    # endregion
    # Dropping 5886 from the noise assessment analysis because its computed threshold is 3 (10% hit rate for 2µm and 90% for 4µm)
    excluded_rec = recs.pop(5886)
    # region ====== Comparison of threshold to session threshold ======
    # rows = []
    # for rec in recs.values():
    #     rows.append({"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.threshold, "session_threshold": rec.session_threshold, "session_x0": rec.x0_psy})
    # session_threshold = pd.DataFrame(rows)
    #
    # from percephone.utils.math_formulas import sigmoid_fit
    # fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(20, 12), constrained_layout=True)
    # axs = ax.flatten()
    # for i, rec in enumerate(recs.values()):
    #     axs[i].set_title(f"{rec.filename} - {rec.threshold}/{rec.session_threshold}({rec.x0_psy:.2f})")
    #     axs[i].set_ylim(0, 1)
    #     axs[i].scatter(np.arange(start=2, stop=13, step=2), rec.hit_rates[1:])
    #     x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates)
    #     axs[i].plot(x, y, color='red')
    # plt.show()
    # endregion
    # region ====== TBT variability ======
    activity_long_df = get_mean_trial_activity_df(recs.values(), zscore=True)
    prestim_df = prestim_activated_neurons(activity_long_df)
    prestim_vector_df = prestim_act_vector(activity_long_df, metric="Stim_mean", hit_activated_only=False)
    # pca_df = pca(activity_long_df)
    # endregion
    # region ====== E/I Ratio ======
    # ei_df = get_ei_ratio_df(recs.values())
    # ei_behavior_df = correlate_behavior(ei_df, column="EI_ratio_cste")
    # ei_comp_df = compare_ei_ratio(ei_df, column="EI_ratio_norm")
    # endregion