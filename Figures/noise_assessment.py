# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from sklearn.decomposition import PCA

import percephone.core.recording as pc
import percephone.plts.stats as ppt
# endregion
# region ======================================== TBT variability ======================================================

def get_mean_trial_activity_df(recs, zscore=True):
    rows = []
    for rec in recs:
        exc_activity = rec.zscore_exc if zscore else rec.df_f_exc
        inh_activity = rec.zscore_inh if zscore else rec.df_f_inh
        behavior_vector = rec.detected_stim
        amplitude_vector = rec.stim_ampl
        stim_time_vector = rec.stim_time
        stim_duration_vector = rec.stim_durations
        for neuron_type, activity in zip(["EXC", "INH"], [exc_activity, inh_activity]):
            for n_id, neuron_activity in enumerate(activity):
                for trial_id in range(len(behavior_vector)):
                    stim_start = stim_time_vector[trial_id]
                    stim_duration = int(stim_duration_vector[trial_id])
                    trial_activity = np.mean(neuron_activity[stim_start: stim_start + stim_duration])
                    row = {"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.session_threshold,
                           "Trial": trial_id, "Amplitude": amplitude_vector[trial_id],
                           "Behavior": behavior_vector[trial_id], "Neuron": f"{neuron_type}_{n_id}", "Activity": trial_activity}
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
                              values="Activity").reset_index()
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




# endregion ============================================================================================================
# region ======================================== Pre-stimulus =========================================================





# endregion ============================================================================================================
# region ======================================== E/I Ratio ============================================================

def get_ei_ratio_df(recs):
    """
    Returns a Dataframe wiht each row being the E/I ratio for a specific trial for a specific animal. E/I ratio is
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
        act_exc_vector = rec.get_perc_resp(pattern=1, n_type="EXC")
        act_inh_vector = rec.get_perc_resp(pattern=1, n_type="INH")
        ei_ratio_vector = np.divide(act_exc_vector, act_inh_vector)
        for trial_id in range(len(behavior_vector)):
            rows.append({"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.session_threshold,
                         "Trial": trial_id, "Amplitude": amplitude_vector[trial_id],
                         "Behavior": behavior_vector[trial_id], "EI_ratio": ei_ratio_vector[trial_id]})
    return pd.DataFrame(rows)

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
    rows = []
    for rec in recs.values():
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.threshold, "session_threshold": rec.session_threshold, "session_x0": rec.x0_psy})
    session_threshold = pd.DataFrame(rows)

    from percephone.utils.math_formulas import sigmoid_fit
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(20, 12), constrained_layout=True)
    axs = ax.flatten()
    for i, rec in enumerate(recs.values()):
        axs[i].set_title(f"{rec.filename} - {rec.threshold}/{rec.session_threshold}({rec.x0_psy:.2f})")
        axs[i].set_ylim(0, 1)
        axs[i].scatter(np.arange(start=2, stop=13, step=2), rec.hit_rates[1:])
        x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates)
        axs[i].plot(x, y, color='red')
    plt.show()
    # endregion
    # region ====== TBT variability ======
    activity_long_df = get_mean_trial_activity_df(recs.values(), zscore=True)
    pca_df = pca(activity_long_df)
    # endregion

    # region ====== E/I Ratio ======
    ei_df = get_ei_ratio_df(recs.values())
    # endregion