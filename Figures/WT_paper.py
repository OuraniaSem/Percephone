import pandas as pd
import os
import numpy as np
from multiprocessing import cpu_count, pool

from matplotlib import pyplot as plt

import percephone.core.recording as pc
from Figures.stimulus_encoding import get_features, filter_amplitude
from percephone.plts.heatmap import interactive_heatmap, ordered_heatmap
import percephone.plts.stats as ppt
import percephone.plts.style as sty



def get_stim_activity_df(recs, zscore=True):
    """
    Get a dataframe with each row being the activity of a specific neuron during a trial.
    """
    prestim_frames = 15
    rows = []
    for rec in recs:
        rec_id = f"{int(rec.filename)}-{rec.condition}"
        exc_activity = rec.zscore_exc if zscore else rec.df_f_exc
        inh_activity = rec.zscore_inh if zscore else rec.df_f_inh
        amplitude_vector = rec.stim_ampl
        stim_time_vector = rec.stim_time
        # stim_duration_vector = rec.stim_durations # always 15 frames, only for motivated
        stim_duration_vector = np.full_like(stim_time_vector, 15, dtype=int)
        for neuron_type, activity in zip(["EXC", "INH"], [exc_activity, inh_activity]):
            resp_mat = rec.matrices[neuron_type]["Responsivity"]
            # AUC = rec.matrices[neuron_type]["AUC"]
            # cum_AUC = rec.matrices[neuron_type]["cum_AUC"]
            for n_id, neuron_activity in enumerate(activity):
                for trial_id in range(len(stim_time_vector)):
                    stim_start = stim_time_vector[trial_id]
                    stim_duration = int(stim_duration_vector[trial_id])
                    trial_activity = neuron_activity[stim_start: stim_start + stim_duration]
                    trial_mean_activity = np.mean(trial_activity)
                    trial_std_activity = np.std(trial_activity)
                    prestim_activity = neuron_activity[stim_start - prestim_frames: stim_start]
                    prestim_mean_activity = np.mean(prestim_activity)
                    prestim_std_activity = np.std(prestim_activity)
                    row = {"Condition": rec.condition, "ID": rec_id, "Threshold": rec.session_threshold,
                           "Trial": trial_id, "Amplitude": amplitude_vector[trial_id],
                           "Neuron": f"{neuron_type}_{n_id}",
                           "Resp": resp_mat[n_id, trial_id],
                           "Stim_mean": trial_mean_activity, "Stim_std": trial_std_activity,
                           "Prestim_mean": prestim_mean_activity, "Prestim_std": prestim_std_activity,}
                           # "AUC": AUC[n_id, trial_id], "cum_AUC": cum_AUC[n_id, trial_id]}
                    rows.append(row)
    activity_df = pd.DataFrame(rows)
    activity_df["Diff_mean"] = activity_df["Stim_mean"] - activity_df["Prestim_mean"]
    activity_df["Abs_Diff_mean"] = abs(activity_df["Stim_mean"] - activity_df["Prestim_mean"])
    activity_df["Diff_std"] = activity_df["Stim_std"] - activity_df["Prestim_std"]
    activity_df["Ratio_mean"] = activity_df["Stim_mean"] / activity_df["Prestim_mean"]
    activity_df["Ratio_std"] = activity_df["Stim_std"] / activity_df["Prestim_std"]
    activity_df["Abs_Ratio_std"] = abs(activity_df["Stim_std"]) / abs(activity_df["Prestim_std"])
    return activity_df

def plot_perc_resp_amp(feature_df):
    """
    Comparing the number of recruited neurons per amplitude across conditions
    Parameters
    ----------
    feature_df

    Returns
    -------

    """
    data = feature_df.groupby(["ID", "Condition", "threshold", "bounded_x0", "amplitude"],
                                      as_index=False).mean().drop(columns=["Trial"])
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15), constrained_layout=True)
    for row, n_type in enumerate(["EXC", "INH"]):
        for col, pattern in enumerate(["act", "inh", "rec"]):
            var_col = f"{pattern}_{n_type}_perc"
            ppt.curveplot(ax[row, col], data, between="Condition", within="amplitude", variable=var_col, data_points=True,
                          title=var_col, ylabel=None, xlabel=None, ylim=None,
                          colors=[sty.motivated_color, sty.naive_color, sty.trained_color], id_display=True, legend_display=True,
                          qq_show=False, transformation=None, consider_normality=False, consider_homogeneity=True)
    fig.canvas.manager.set_window_title('Percentage of Responsive Neurons per Amplitude')
    plt.show()
    return data

def compare_neuronal_features(feature_df, amps=["all", "all", "all"]):
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
    raw_data = feature_df.copy().drop(columns=["Trial"])
    groups = ["naive", "trained", "motivated"]
    # Filtering the DataFrame to keep only the rows of desired amplitude
    data = []
    for amp, group in zip(amps, groups):
        data.append(filter_amplitude(raw_data[raw_data["Condition"] == group], amplitude=amp, no_go=False))
    data = pd.concat(data, axis=0, ignore_index=True)
    grouped_data = data.groupby(["ID", "Condition", "threshold", "bounded_x0"], as_index=False).mean()
    # Plotting the comparisons
    not_variables = ["ID", "Condition", "amplitude", "threshold", "bounded_x0"]
    variables = [col for col in data.columns if col not in not_variables]
    variables = [var for var in variables if var.split("_")[-1] != "auc"]
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
            ppt.boxplot_df(ax, grouped_data, group_col="Condition", data_col=variable, ylabel=variable, title="", ylim=ylim,
                           colors=[sty.naive_color, sty.trained_color, sty.motivated_color],
                           gp_sort=["naive", "trained", "motivated"], force_normality=False)
    fig.suptitle(f"Comparison between {amps[0]} trials of {groups[0]}, {amps[1]} trials of {groups[1]} & {amps[2]} trials of {groups[2]}"
                 f"\n n={len(grouped_data[grouped_data["Condition"] == groups[0]]["ID"].unique())}{groups[0]}/"
                 f"{len(grouped_data[grouped_data["Condition"] == groups[1]]["ID"].unique())}{groups[1]}/"
                 f"{len(grouped_data[grouped_data["Condition"] == groups[2]]["ID"].unique())}{groups[2]}", fontsize=20)
    title = f"comp_{amps[0]}({groups[0]})_{amps[1]}({groups[1]})_{amps[2]}({groups[2]})"
    fig.canvas.manager.set_window_title(title)
    plt.show()
    return data

if __name__ == '__main__':
    ### Initialisation of recs instances ###
    dir_naive = "C:/Users/cvandromme/Desktop/WT paper/naive"
    dir_trained = "C:/Users/cvandromme/Desktop/WT paper/trained"
    dir_motivated = "C:/Users/cvandromme/Desktop/WT paper/motivated"
    roi_path = "C:/Users/cvandromme/Desktop/WT paper/FmKO_StimOnly_selected_recordings.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/WT paper/"
    roi_info = pd.read_excel(roi_path)
    # files = os.listdir(dir_naive) + os.listdir(dir_trained) + os.listdir(dir_motivated)
    # files_ = [file for file in files if file.endswith("synchro")]

    # def opening_rec(fil, i):
    #     rec = pc.RecordingStimulusOnly(directory + fil + "/", 0, roi_path, cache=True, correction=False)
    #     return rec
    # workers = cpu_count()
    # pool = pool.ThreadPool(processes=workers)
    # async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    #
    # recs = {f"{ar.get().filename}-{ar.get().genotype.split("-")[1]}": ar.get() for ar in async_results}


    files_dict = {}
    for directory in [dir_naive, dir_trained, dir_motivated]:
        for f in os.listdir(directory):
            if f.endswith("synchro"):
                files_dict[f] = directory

    def opening_rec(fil, directory):
        if (os.path.basename(directory) == "trained" and int(fil[9:13]) in [4447, 4456, 4458]) or (os.path.basename(directory) == "naive" and int(fil[9:13]) in [4545]):
            rec = pc.RecordingStimulusOnly(os.path.join(directory, fil + "/"), roi_path, mean_f_bsl=False, correction=False)
        else:
            rec = pc.RecordingAmplDet(f"{directory}/{fil}/", 0, roi_path, tuple_mesc=None, cache=True, correction=False, habituation=True)
        return rec

    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, directory)) for file, directory in files_dict.items()]
    recs = {f"{str(int(ar.get().filename))}-{ar.get().condition}": ar.get() for ar in async_results}

    # rec = pc.RecordingAmplDet(os.path.join(dir_naive, "20221128_4939_00_synchro/"), 0, roi_path, tuple_mesc=None, cache=True, correction=False, habituation=True)
    rec = pc.RecordingStimulusOnly(os.path.join(dir_naive, "20220720_4545_00_synchro/"), roi_path, mean_f_bsl=False, correction=False)
    # Plotting the heatmap to assert everything is properly integrated
    # interactive_heatmap(rec, rec.zscore_exc)
    # ordered_heatmap(rec, exc_neurons=True, inh_neurons=False, time_span="stim", window=0.5, estimator=None,
    #                 det_sorted=False, amp_sorted=False, det_ordering=False, avg_trials_amp=False, threshold_only=False)

    rows = []
    for rec in recs.values():
        row = {"ID": rec.filename, "condition": rec.condition, "zscore": rec.zscore_exc[0, 0]}
        rows.append(row)
        rec.responsivity(dff=True)
    summary = pd.DataFrame(rows)

    # for rec in recs.values():
    #     rec.peak_delay_amp()

    # Building the DataFrames
    # activity_df = get_stim_activity_df(recs.values(), zscore=True)
    feature_df = get_features(recs.values(), amp_delay=True, auc=False, habituation=True)
    feature_dff = get_features(recs.values(), amp_delay=True, auc=False, habituation=True, dff_resp=True)

    # # Comparing the number of recruited neurons per amplitude across conditions
    mean_feature = plot_perc_resp_amp(feature_df)
    mean_feature_dff = plot_perc_resp_amp(feature_dff)

    neuron_features = compare_neuronal_features(feature_df, amps=[[12], [12], [12]])
    neuron_features = compare_neuronal_features(feature_df, amps=["all", "all", "all"])
    neuron_features = compare_neuronal_features(feature_df, amps=["threshold", "threshold", "threshold"])
    # rec = recs["4745-trained"]
    # interactive_heatmap(rec, rec.df_f_exc)

