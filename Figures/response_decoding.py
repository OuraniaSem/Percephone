# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

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


def get_activity_by_frame_df(recs, zscore=True):
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
        activity_vector = [rec.zscore_exc, rec.zscore_inh] if zscore else [rec.df_f_exc, rec.df_f_inh]
        for n_type, activity in zip(["EXC", "INH"], activity_vector):
            resp = np.array(rec.matrices[n_type]["Responsivity"])
            for neuron_id, neuron in enumerate(np.array(activity)):
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
    """
    Builds a DataFrame, each row being a neuron type for a specific animal, and the values of correlation (r and pval)
    of the vector of zscore at each frame and the vector of behavior.

    Parameters
    ----------
    frame_data

    Returns
    -------

    """
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

def glmm_behavior(data):
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


def frame_model(frame_data):
    """
    Return the hit versus miss classification graph.
    A logistic regression model is trained on for each frame for each animal. CV is used to assess the hit accuracy
    (True positive) and  miss accuracy (True negative). Under sampling was performed to avoid biases linked to the class
    unbalance

    Parameters
    ----------
    frame_data

    Returns
    -------

    """
    # Grouping the different neurons
    header_columns = ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Duration", "Behavior", "n_type", "resp",
                      "n_ID"]
    mean_accross = ["n_ID"]
    grouping_columns = [col for col in header_columns if col not in mean_accross]
    data = frame_data.groupby(grouping_columns, as_index=False).mean().drop(columns=mean_accross)
    # Creating a pivot DataFrame to obtain a dataframe per frame, each column being a neuron type/resp combinaison
    index_columns = [col for col in grouping_columns if col not in ["n_type", "resp"]]
    numeric_columns = [c for c in data.columns if c not in header_columns]
    rows = []
    for frame in numeric_columns:
        frame_data = data.pivot(index=index_columns, columns=["n_type", "resp"], values=frame).reset_index()
        # Training a model for each recording and storing the evaluation metrics in a new DataFrame
        for rec_id in frame_data["ID"].unique():
            filtered_data = frame_data[frame_data["ID"] == rec_id]
            X = filtered_data.drop(columns=index_columns)
            y = filtered_data["Behavior"]
            # Split your data into training and test sets (using stratification to preserve class distribution)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            # Create a pipeline that first undersamples then fits a logistic regression model
            undersampler = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)
            lr = LogisticRegression(solver='lbfgs', max_iter=5000)
            # Set up a grid of hyperparameters to tune
            param_grid = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "penalty": ['l2']}
            # Use cross-validation (here, 5-fold) to search for the best hyperparameters
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train_res, y_train_res)
            print("Best parameters found:", grid_search.best_params_)
            # Evaluate on the test set
            y_pred = grid_search.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            # Calculate true positive and true negative rates
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn)  # Sensitivity / Recall
            tnr = tn / (tn + fp)  # Specificity
            print(f"True Positive Rate (Sensitivity): {tpr:.3f}")
            print(f"True Negative Rate (Specificity): {tnr:.3f}")
            # Optionally, print a full classification report
            print(classification_report(y_test, y_pred))
            row = {"Genotype": filtered_data["Genotype"].values[0], "ID": filtered_data["ID"].values[0],
                   "Threshold": filtered_data["Threshold"].values[0], "Frame": frame, "TPR": tpr, "TNR": tnr}
            rows.append(row)
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
    for rec in recs.values():
        rec.peak_delay_amp()
    # endregion
    # test = perceptual_magnitude_behavior_corr(recs.values())
    # mean_corr = test.groupby("Genotype").mean()
    # glmm_behavior(data)

    data = get_features(recs.values())
    frame_data = get_activity_by_frame_df(recs.values(), zscore=True)
    corr_data = correlate_mean_zscore_behavior_frame(frame_data[frame_data["Amplitude"] == 12])
    plot_frame_correlation(corr_data)

    frame_dff = get_activity_by_frame_df(recs.values(), zscore=False)
    frame_model_df = frame_model(frame_dff[frame_data["ID"] == 4445])