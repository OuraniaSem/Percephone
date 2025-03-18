# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from scipy.stats import pointbiserialr

import percephone.core.recording as pc
from Figures.stimulus_encoding import get_features


# endregion
# region ======================================== Correlation ==========================================================

def perceptual_magnitude_behavior_corr(recs):
    """
    Correlates a single parameter witht he behavioral outcome for threshold trials using point biserial corrrelation.
    Can not be used to correlate all trials because need to include the amplitude as a covariate.

    Parameters
    ----------
    recs

    Returns
    -------

    """
    rows = []
    for rec in recs:
        # Keeping only the stimulation at threshold amplitude to limit bias
        threshold_trials_mask = rec.stim_ampl_filter(stim_ampl="all")
        # Getting the vector of the parameter to correlate with the behavior
        act_exc_vector = rec.get_perc_resp(pattern=1, n_type="EXC")[threshold_trials_mask]
        # Getting the vector of behavioral outcome
        behavior_vector = rec.detected_stim[threshold_trials_mask]
        if len(behavior_vector) > 2:
            # Point biserial correlation of the vectors
            r, p_val = pointbiserialr(act_exc_vector, behavior_vector)
            rows.append({"ID": rec.filename, "Genotype": rec.genotype, "session_threshold": rec.session_threshold,
                         "nb_threshold_trials": len(behavior_vector), "R2": r**2, "p_val": p_val})
        else:
            print(f"{rec.filename} {rec.genotype} excluded → only {len(behavior_vector)} threshold trial(s)")
    return pd.DataFrame(rows)


def get_zscore_by_frame_df(recs):
    """
    Generates a pd.DataFrame, each containing the neuronal data for a single neuron 30 frames preceding and following for a single trial.
    Frame 30 = stim_time.

    Parameters
    ----------
    recs

    Returns
    -------

    """
    rows = []
    for rec in recs:
        for n_type, zscore in zip(["EXC", "INH"], [rec.zscore_exc, rec.zscore_inh]):
            resp = np.array(rec.matrices[n_type]["Responsivity"])
            for neuron_id, neuron in enumerate(np.array(zscore)):
                for (trial_id, trial_time), trial_duration, trial_amp, trial_label in zip(enumerate(rec.stim_time),
                                                                                          rec.stim_durations,
                                                                                          rec.stim_ampl, rec.detected_stim):
                    row = {"Genotype": rec.genotype, "ID": rec.filename, "Threshold": rec.session_threshold,
                           "Trial": trial_id, "Amplitude": trial_amp, "Duration": trial_duration,
                           "Behavior": trial_label, "n_type": n_type, "resp": resp[neuron_id, trial_id], "n_ID": neuron_id}
                    for frame_id, frame in enumerate(range(trial_time - 30, trial_time + 30)):
                        row[frame_id] = neuron[frame]
                    rows.append(row)
    return pd.DataFrame(rows)


def correlate_mean_zscore_behavior_frame(frame_data):
    # Computing the mean zscore per trial
    header_columns = ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Duration", "Behavior", "n_type", "resp", "n_ID"]
    mean_accross = ["n_ID"]
    grouping_columns = [col for col in header_columns if col not in mean_accross]
    data = frame_data.groupby(grouping_columns, as_index=False).mean().drop(columns=mean_accross)
    rows = []
    for rec_id in data["ID"].unique():
        rec_data = data[data["ID"] == rec_id]
        genotype = rec_data["Genotype"].values[0]
        for neuron_type in rec_data["n_type"].unique():
            n_type_data = rec_data[rec_data["n_type"] == neuron_type]
            for response in n_type_data["resp"].unique():
                resp_data = n_type_data[rec_data["resp"] == response]
                row_r = {"Genotype": genotype, "ID": rec_id, "n_type": neuron_type, "resp": response, "metric": "r"}
                row_pval = {"Genotype": genotype, "ID": rec_id, "n_type": neuron_type, "resp": response, "metric": "pval"}
                y = resp_data["Behavior"].values
                if len(resp_data) > 1:
                    for col in [c for c in data.columns if c not in header_columns]:
                        x = resp_data[col].values
                        row_r[col], row_pval[col] = pointbiserialr(x, y)
                    rows.append(row_r)
                    rows.append(row_pval)
    return pd.DataFrame(rows)


def plot_frame_correlation(corr_data):
    color_dict = {"EXC": ["skyblue", "blue", "navy"], "INH": ["pink", "magenta", "darkviolet"]}
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    for col, genotype in enumerate(corr_data["Genotype"].unique()):
        for row, metric in enumerate(corr_data["metric"].unique()):
            data = corr_data[(corr_data["Genotype"] == genotype) & (corr_data["metric"] == metric)].drop(columns=["Genotype", "metric"])
            header_columns = ["ID", "n_type", "resp"]
            mean_accross = ["ID"]
            grouping_columns = [col for col in header_columns if col not in mean_accross]
            data = data.groupby(grouping_columns, as_index=False).mean().drop(columns=mean_accross)
            # Plotting
            ax[row, col].set_title(f"{metric} for {genotype}")
            for i, curve_row in data.iterrows():
                y = curve_row.drop(labels=["n_type", "resp"]).values.tolist()
                x = np.arange(len(y))
                ax[row, col].plot(x, y, color=color_dict[curve_row["n_type"]][int(curve_row["resp"])], lw=1)
                ax[row, col].axvline(x=30, ls="--", lw=1, color="red")
                ax[row, col].axvline(x=45, ls="--", lw=1, color="black")
                if metric == "pval":
                    ax[row, col].axhline(y=0.05, ls="--", lw=1, color="green")
    fig.suptitle("Frame by frame correlation of zscore with behavior")
    fig.canvas.manager.set_window_title("Frame_corr_zscore_behavior")
    plt.show()



# endregion ============================================================================================================
# region ======================================== Modelling ============================================================

def model_behavior(data):
    """
    Aims to model the behavioral outcome using a list of predictors defined as important for stimulus encoding (GLMM).

    Parameters
    ----------
    rec

    Returns
    -------

    """
    data["Genotype"] = pd.Categorical(data["Genotype"], categories=["WT", "KO", "KO-Hypo"], ordered=True)
    # data["behavior"] = pd.Categorical(data["behavior"], categories=["False", "True"], ordered=True)
    # data["behavior"] = data["behavior"].astype(int)
    # === === Fitting the model === ===
    # --- GEE ---
    formula = ("behavior ~ amplitude*Genotype + amplitude:act_EXC_perc*Genotype + amplitude:inh_EXC_perc*Genotype + "
               "amplitude:act_INH_perc*Genotype + amplitude:inh_INH_perc*Genotype")# + "
    #            # "act_EXC_amp + inh_EXC_amp + act_INH_amp + inh_INH_amp + "
    #            # "act_EXC_delay + inh_EXC_delay + act_INH_delay + inh_INH_delay + "
    #            # "Genotype + amplitude:Genotype")
    # gee_model = smf.gee(formula, groups="ID", data=data[data["Genotype"] == "WT"], family=sm.families.Binomial())
    # gee_result = gee_model.fit()
    # print(gee_result.summary())
    # --- GLMM ---
    vc_formulas = {"ID": "0 + C(ID)"}
    model = sm.genmod.BinomialBayesMixedGLM.from_formula(formula, vc_formulas, data=data)#, family=sm.families.Binomial())
    result = model.fit_vb()
    print(result.summary())


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
    for rec in recs.values():
        rec.peak_delay_amp()
    # endregion
    # test = perceptual_magnitude_behavior_corr(recs.values())
    # mean_corr = test.groupby("Genotype").mean()
    # model_behavior(data)

    data = get_features(recs.values())
    frame_data = get_zscore_by_frame_df(recs.values())
    corr_data = correlate_mean_zscore_behavior_frame(frame_data[frame_data["Amplitude"] == 12])
    plot_frame_correlation(corr_data)