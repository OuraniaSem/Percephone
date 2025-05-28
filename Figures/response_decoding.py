# region ======================================== Imports ==============================================================
import os

import mplcursors
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from scipy.stats import pointbiserialr, linregress
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from itertools import product
from tqdm import tqdm

import percephone.core.recording as pc
import percephone.plts.stats as ppt
from Figures.stimulus_encoding import get_features


# endregion

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


def frame_model_n_type_avg(frame_data):
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
        print(f"Frame n°{frame}")
        frame_data = data.pivot(index=index_columns, columns=["n_type", "resp"], values=frame).reset_index()
        # Dropping no go trials and the inhibited INH neurons because they are too few and induce NaN values
        frame_data = frame_data.drop(columns=("INH", -1))
        frame_data = frame_data[frame_data["Amplitude"] != 0]
        # Imputing the NaN by the mean value per animal per amplitude
        for col in [("EXC", 0), ("EXC", 1), ("EXC", -1), ("INH", 0), ("INH", 1)]:
            frame_data[col] = frame_data.groupby(["Genotype", "ID", "Threshold", "Amplitude"], as_index=False)[[col]].transform(lambda x: x.fillna(x.mean()))
        # Training a model for each recording and storing the evaluation metrics in a new DataFrame
        for rec_id in frame_data["ID"].unique():
            filtered_data = frame_data[frame_data["ID"] == rec_id]
            X = filtered_data.drop(columns=index_columns)
            y = filtered_data["Behavior"]
            # Splitting the data into training and test sets (using stratification to preserve class distribution)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
            # 1) Use CV to find best C value for L2 regularization of LR, undersample each fold
            # pipeline = Pipeline(steps=[('under', RandomUnderSampler(random_state=42)),
            #                            ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))])
            # 2) Train the LR with the best parameters on the global undersampled X_train
            # 3) Assess model performance on test set
            # Create a pipeline that first undersamples then fits a logistic regression model
            undersampler = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)
            lr = LogisticRegression(solver='lbfgs', C=1, max_iter=5000)
            # Set up a grid of hyperparameters to tune
            # param_grid = {"clf__C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
            # Use cross-validation (here, 5-fold) to search for the best hyperparameters
            # grid_search = GridSearchCV(pipeline, param_grid, cv=4, scoring='accuracy')
            # grid_search.fit(X_train, y_train)
            # Evaluate on the test set
            # y_pred = grid_search.predict(X_test)
            lr.fit(X_train_res, y_train_res)
            y_pred = lr.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            # Calculate true positive and true negative rates
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn)  # Sensitivity / Recall
            fpr = fp / (tn + fp)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            # Optionally, print a full classification report
            # print(classification_report(y_test, y_pred))
            row = {"Genotype": filtered_data["Genotype"].values[0], "ID": filtered_data["ID"].values[0],
                   "Threshold": filtered_data["Threshold"].values[0], "Frame": frame, "TPR": tpr, "FPR": fpr, "Accuracy": accuracy}
            rows.append(row)
    return pd.DataFrame(rows)
    # return frame_data


def frame_model(frame_data, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True):
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
    header_columns = ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Duration", "Behavior", "n_type", "resp", "n_ID"]
    # Filtering the neuron types and activity
    data = frame_data[frame_data["n_type"].isin(neuron_type)]
    data = data[data["resp"].isin(resp_type)]
    data = data.drop(columns=["Threshold", "Amplitude", "Duration", "resp"])
    index_columns = ["Genotype", "ID", "Trial", "Behavior", "n_type", "n_ID"]
    numeric_columns = [c for c in data.columns if c not in header_columns]
    rows = []
    # Creating a pivot DataFrame to obtain a dataframe per frame, each column being a neuron
    for frame in tqdm(numeric_columns):
        # frame_data = frame_data[frame_data["Amplitude"] != 0]
        # Training a model for each recording and storing the evaluation metrics in a new DataFrame
        for rec_id in data["ID"].unique():
            rec_data = data[data["ID"] == rec_id].copy()
            rec_data["Neuron"] = rec_data["n_ID"].astype(str) + "_" + rec_data["n_type"].astype(str)
            final_data = rec_data.pivot(index=["Trial", "Behavior"], columns="Neuron", values=frame).reset_index()
            y = final_data["Behavior"]
            X = final_data.drop(columns=["Trial", "Behavior"])
            if db_cv:
                metrics = double_cv(np.array(X), np.array(y), cv_out_fold=4, cv_in_fold=4,
                                    param_grid={"C": [0.0001, 0.001, 0.01, 0.1, 1], "penalty": ["l2"]},
                                    scoring_metric="Accuracy", resampler=RandomUnderSampler(random_state=42),
                                    random_state=42, get_df=False)
            else:
                # Splitting the data into training and test sets (using stratification to preserve class distribution)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                undersampler = RandomUnderSampler(random_state=42)
                X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)
                lr = LogisticRegression(solver='lbfgs', C=1, max_iter=5000)
                lr.fit(X_train_res, y_train_res)
                y_pred = lr.predict(X_test)
                metrics = get_metrics(y_test, y_pred)
            row = {"Genotype": rec_data["Genotype"].values[0], "ID": rec_data["ID"].values[0],
                   "Frame": frame, "TPR": metrics["TPR"], "FPR": metrics["FPR"], "Accuracy": metrics["Accuracy"]}
            rows.append(row)
    frame_model_df = pd.DataFrame(rows)
    frame_mean_sem = plot_hit_miss_classif(frame_model_df, title_precision=f"{neuron_type}{resp_type} - Double CV=={db_cv}")
    frame_comp = plot_hit_miss_classif_comp(frame_model_df, gp1="WT", gp2="KO-Hypo", title_precision=f"{neuron_type}{resp_type} - db_cv={db_cv}")
    return frame_model_df, frame_mean_sem, frame_comp


def get_metrics(y_test, y_pred):
    metrics = {}
    cm = confusion_matrix(y_test, y_pred, labels=[False, True])
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    metrics["TPR"] = TP / (TP + FN)  # Sensitivity / Recall
    metrics["FPR"] = FP / (TN + FP)
    metrics["Accuracy"] = (TP + TN) / (TP + TN + FP + FN)
    return metrics


def double_cv(X, y, cv_out_fold=5, cv_in_fold=5, param_grid={"C": [0.0001, 0.001, 0.01, 0.1, 1], "penalty": ["l2"]},
              scoring_metric="Accuracy", resampler=RandomUnderSampler(random_state=42), random_state=42, get_df=False):
    """
    Perform double cross validation
    Parameters
    ----------
    X
    y
    cv_out_fold
    cv_in_fold
    param_grid
    scoring_metric
    resampler
    random_state
    get_df

    Returns
    -------

    """
    cv_out = StratifiedKFold(n_splits=cv_out_fold, random_state=random_state, shuffle=True)
    rows = []
    # Splitting the data into training and validation sets
    for fold_out, (train_index, val_index) in enumerate(cv_out.split(X, y)):
        row = {"Fold": fold_out}
        X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
        # Splitting the train data into tuning and tuning assessment group
        cv_in = StratifiedKFold(n_splits=cv_in_fold, random_state=random_state, shuffle=True)
        inner_scores = {}
        param_names = list(param_grid.keys())
        param_combinations = list(product(*[param_grid[p] for p in param_names]))
        for params in param_combinations:
            params_dict = dict(zip(param_names, params))
            # Performing CV to find the best hyperparameters
            fold_scores = []
            for fold_in, (tuning_index, test_index) in enumerate(cv_in.split(X_train, y_train)):
                X_tuning, X_test, y_tuning, y_test = X_train[tuning_index], X_train[test_index], y_train[tuning_index], y_train[test_index]
                # Tuning hyper-parameters on the tuning data set
                model = LogisticRegression(**params_dict, max_iter=5000, random_state=random_state)
                model.fit(X_tuning, y_tuning)
                # Predicting the test set and storing metrics
                y_pred_in = model.predict(X_test)
                metrics = get_metrics(y_test, y_pred_in)
                fold_scores.append(metrics[scoring_metric])
            inner_scores[tuple(params)] = np.mean(fold_scores)
        # Choosing the best parameters from the inner loop
        best_params_tuple = max(inner_scores, key=inner_scores.get)
        best_params = dict(zip(param_names, best_params_tuple))
        row["Best_param"] = best_params
        # Resampling the training set and training the final model with this set
        if resampler is not None:
            X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        final_model = LogisticRegression(**best_params, max_iter=5000, random_state=random_state)
        final_model.fit(X_train_res, y_train_res)
        # Assessing the final model's performance
        y_val_pred = final_model.predict(X_val)
        outer_metrics = get_metrics(y_val, y_val_pred)
        row.update(outer_metrics)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    results_metrics = results_df.mean(numeric_only=True).to_dict()
    return results_df if get_df else results_metrics


def plot_hit_miss_classif(frame_model_df, title_precision=""):
    """
    Plot the hit vs miss classification graph from a Dataframe with each line containing infos about TPR and TNR for 1
    frame for one animal.

    Parameters
    ----------
    frame_model_df

    Returns
    -------

    """
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color], "KO": [ppt.ko_color, ppt.ko_light_color]}
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8), constrained_layout=True)
    for i, genotype in enumerate(frame_model_df["Genotype"].unique()):
        data = frame_model_df[frame_model_df["Genotype"] == genotype].drop(columns=["Genotype", "ID"])
        tpr_mean = data.groupby("Frame")["TPR"].mean().values
        tpr_sem = data.groupby("Frame")["TPR"].sem().values
        fpr_mean = data.groupby("Frame")["FPR"].mean().values
        fpr_sem = data.groupby("Frame")["FPR"].sem().values
        x = np.arange(len(tpr_mean))
        ax[i].plot(x, tpr_mean, label="Hit accuracy", color=color_dict[genotype][0], lw=2)
        ax[i].fill_between(x, tpr_mean - tpr_sem, tpr_mean + tpr_sem, color=color_dict[genotype][0], alpha=0.3)
        ax[i].plot(x, fpr_mean, label="Miss accuracy", color=color_dict[genotype][1], lw=2)
        ax[i].fill_between(x, fpr_mean - fpr_sem, fpr_mean + fpr_sem, color=color_dict[genotype][1], alpha=0.3)
        ax[i].axvline(x=30, ls="--", lw=1, color="red")
        ax[i].axvline(x=45, ls="--", lw=1, color="black")
        ax[i].axhline(y=0.5, ls="--", lw=1, color="gray")
        ax[i].set_title(genotype, color=color_dict[genotype][0], fontsize=20)
        ax[i].set_ylim(0, 1)
        ax[i].set_xlabel("Time (s)")
        ax[i].set_xticks([0, 15, 30, 45, 60])
        ax[i].set_xticklabels([-1, -0.5, 0, 0.5, 1])
    fig.suptitle(f"Hit versus Miss classification graph using the mean ΔF/F\n{title_precision}", fontsize=20)
    fig.canvas.manager.set_window_title(f"Hit_Miss_classif_{title_precision}")
    plt.show()
    return data


def plot_hit_miss_classif_comp(frame_model_df, gp1="WT", gp2="KO-Hypo", title_precision=""):
    color_dict = {"WT": [ppt.wt_color, ppt.wt_light_color], "KO-Hypo": [ppt.hypo_color, ppt.hypo_light_color],
                  "KO": [ppt.ko_color, ppt.ko_light_color]}
    period_dict = {"stim": [30, 45], "start_stim (250ms)": [30, 37], "end_stim (250ms)": [37, 45], "pre_stim (200ms)": [24, 30]}
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 12), constrained_layout=True)
    for col, period in enumerate(period_dict.keys()):
        start, end = period_dict[period]
        data = frame_model_df[frame_model_df["Frame"].isin(range(start, end))].groupby(["Genotype", "ID"], as_index=False).mean().drop(columns="Frame")
        # Plotting the comparison of accuracy between genotypes
        ppt.boxplot(ax[0, col], data[data["Genotype"] == gp1]["TPR"], data[data["Genotype"] == gp2]["TPR"], ylabel="Hit accuracy",
                    paired=False, title=period, ylim=[0, 1], colors=[color_dict[gp1][0], color_dict[gp2][0]], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[1, col], data[data["Genotype"] == gp1]["FPR"], data[data["Genotype"] == gp2]["FPR"], ylabel="Miss error",
                    paired=False, title=period, ylim=[0, 1], colors=[color_dict[gp1][1], color_dict[gp2][1]], det_marker=False, force_markers_identity=False)
    fig.suptitle(f"Hit accuracy miss error comparison\n{title_precision}", fontsize=20)
    fig.canvas.manager.set_window_title(f"Hit_Miss_classif_comp_{title_precision}")
    plt.show()
    return data

def compare_accuracy(recs, random=42):
    """
    Train a logistic regression model with different subsets of neurons and compare the decoding accuracy.
    Parameters
    ----------
    recs

    Returns
    -------

    """
    rows = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random)
    for rec in recs:
        # Retrieving the neuronal activity
        exc_mat = rec.get_estimated_activity(zscore=True, n_type="EXC", estimator="mean", real_duration=True)
        inh_mat = rec.get_estimated_activity(zscore=True, n_type="INH", estimator="mean", real_duration=True)
        full_mat = np.concatenate([exc_mat, inh_mat], axis=0).T
        nb_exc = exc_mat.shape[0]
        y = rec.detected_stim
        for fold, (train_idx, test_idx) in enumerate(skf.split(full_mat, y), start=1):
            # X_train, X_test, y_train, y_test = train_test_split(full_mat, y, test_size=0.2, stratify=y, random_state=random)
            X_train, X_test = full_mat[train_idx], full_mat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_res, y_res = RandomUnderSampler(random_state=random).fit_resample(X_train, y_train)
            model = LogisticRegression(max_iter=5000, random_state=random)
            # Fitting the model with all neurons
            model.fit(X_res, y_res)
            y_pred_all = model.predict(X_test)
            acc_all = get_metrics(y_test, y_pred_all)["Accuracy"]
            # Fitting the model with EXC neurons
            model.fit(X_res[:, :nb_exc], y_res)
            y_pred_exc = model.predict(X_test[:, :nb_exc])
            acc_exc = get_metrics(y_test, y_pred_exc)["Accuracy"]
            # Fitting the model with INH neurons
            model.fit(X_res[:, nb_exc:], y_res)
            y_pred_inh = model.predict(X_test[:, nb_exc:])
            acc_inh = get_metrics(y_test, y_pred_inh)["Accuracy"]
            rows.append({"ID": rec.filename, "Genotype": rec.genotype, "Fold": fold, "acc_all": acc_all, "acc_exc": acc_exc, "acc_inh": acc_inh})
    full_data = pd.DataFrame(rows)
    data = full_data.groupby(["Genotype", "ID"], as_index=False).mean()
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 24), constrained_layout=True)
    for row, (group, gp_label) in enumerate(zip([data, data[data["Genotype"] == "WT"], data[data["Genotype"] == "KO-Hypo"]],
                                                ["all", "WT", "KO-Hypo"])):
        ppt.boxplot(ax[row, 0], group["acc_all"].values, group["acc_exc"].values, ylabel="Accuracy", paired=True, title=f"All neurons/EXC ({gp_label})", ylim=[0, 1],
                    colors=["#859717", "#229708"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 1], group["acc_all"].values, group["acc_inh"].values, ylabel="Accuracy", paired=True, title=f"All neurons/INH ({gp_label})", ylim=[0, 1],
                    colors=["#859717", "#cba61b"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 2], group["acc_exc"].values, group["acc_inh"].values, ylabel="Accuracy", paired=True, title=f"EXC/INH ({gp_label})", ylim=[0, 1],
                    colors=["#229708", "#cba61b"], det_marker=False, force_markers_identity=False)
    fig.suptitle("Comparison of decoding accuracy between neuron types")
    fig.canvas.manager.set_window_title(f"Accuracy n_types")
    plt.show()
    return data


def correlate_nb_accuracy(recs, accuracy_df, threshold="median"):
    accuracy_df["n_EXC"] = accuracy_df["ID"].map(lambda id_: recs[id_].zscore_exc.shape[0])
    accuracy_df["n_INH"] = accuracy_df["ID"].map(lambda id_: recs[id_].zscore_inh.shape[0])
    accuracy_df["n_all"] = accuracy_df["n_EXC"] + accuracy_df["n_INH"]

    def plot_lin_reg(ax, data, x_col=None, y_col=None, id_col="ID", group_col=None,
                     title=None, xlab=None, ylab=None, colors=None, line_color="red", id_display=True):
        if colors is None:
            colors = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color}
        xlab = xlab or x_col
        ylab = ylab or y_col
        # Correlation
        results = dict(linregress(data[x_col], data[y_col])._asdict())
        r2 = results["rvalue"] ** 2
        line = results["slope"] * data[x_col] + results["intercept"]
        # Plot the data points and regression line
        ax.plot(data[x_col], line, color=line_color, lw=2)
        if group_col is not None:
            for g in sorted(data[group_col].unique()):
                group = data[data[group_col] == g]
                sc = ax.scatter(group[x_col], group[y_col], color=colors[g], alpha=0.7, label=g, s=10, marker="+")
                if id_display:
                    # Save the IDs for this group so that they can be accessed in the callback.
                    ids = group[id_col].values
                    mplcursors.cursor(sc, hover=True).connect("add", lambda sel, ids=ids: (sel.annotation.set_text(f"ID: {ids[sel.index]}"), sel.annotation.set_fontsize(8)))
        else:
            sc = ax.scatter(data[x_col], data[y_col], color=colors[g], alpha=0.7, label=g, s=10, marker="+")
            if id_display:
                ids = data["ID"].values
                mplcursors.cursor(sc, hover=True).connect("add", lambda sel, ids=ids: (sel.annotation.set_text(f"ID: {ids[sel.index]}"), sel.annotation.set_fontsize(8)))
        # Annotate the plot with R² and p-value
        ax.text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {results["pvalue"]:.3f}", transform=ax.transAxes, fontsize=8, verticalalignment="top", color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(8)
        return results

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18, 24), constrained_layout=True)
    plot_lin_reg(ax[0 ,0], accuracy_df, x_col="n_EXC", y_col="acc_exc", group_col="Genotype", title="EXC")
    plot_lin_reg(ax[0 ,1], accuracy_df, x_col="n_INH", y_col="acc_inh", group_col="Genotype", title="INH")
    plot_lin_reg(ax[0 ,2], accuracy_df, x_col="n_all", y_col="acc_all", group_col="Genotype", title="All")
    # Comparing the number of neurons between individuals with high and low accuracy
    for col, accuracy_metric in enumerate(["acc_exc", "acc_inh", "acc_all"]):
        if threshold == "median":
            t_val = np.percentile(accuracy_df[accuracy_metric].values, 50)
        elif threshold == "mean":
            t_val = np.mean(accuracy_df[accuracy_metric].values)
        elif threshold == "middle":
            t_val = (max(accuracy_df[accuracy_metric].values) + min(accuracy_df[accuracy_metric].values))/2
        elif isinstance(threshold, float):
            t_val = threshold
        ax[0, col].axhline(y=t_val, linestyle="--", color="gray", lw=0.5)
        low_perf = accuracy_df[accuracy_df[accuracy_metric] < t_val]
        high_perf = accuracy_df[accuracy_df[accuracy_metric] >= t_val]
        ppt.boxplot(ax[1, col], low_perf["n_EXC"].values, high_perf["n_EXC"].values, ylabel="n_EXC", paired=False,
                    title=f"{accuracy_metric} Low/High acc", ylim=[],
                    colors=["#229708", "#229708"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[2, col], low_perf["n_INH"].values, high_perf["n_INH"].values, ylabel="n_INH", paired=False,
                    title=f"{accuracy_metric} Low/High acc", ylim=[],
                    colors=["#cba61b", "#cba61b"], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[3, col], low_perf["n_all"].values, high_perf["n_all"].values, ylabel="n_all", paired=False,
                    title=f"{accuracy_metric} Low/High acc", ylim=[],
                    colors=["#859717", "#859717"], det_marker=False, force_markers_identity=False)


    fig.suptitle(f"Correlation of the model accuracy with the number of neurons.\nComparison of the number of neurons "
                 f"between high and low accuracy animal (threshold={threshold})", fontsize=12)
    fig.canvas.manager.set_window_title("Corr acc nb neurons")
    plt.show()
    return accuracy_df



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
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec
    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}


    # for rec in recs.values():
    #     rec.peak_delay_amp()
    # endregion ============
    # test = perceptual_magnitude_behavior_corr(recs.values())
    # mean_corr = test.groupby("Genotype").mean()
    # glmm_behavior(data)

    # frame_data = get_activity_by_frame_df(recs.values(), zscore=True)
    # corr_data = correlate_mean_zscore_behavior_frame(frame_data[frame_data["Amplitude"] == 12])
    # plot_frame_correlation(corr_data)

    frame_dff = get_activity_by_frame_df(recs.values(), zscore=False)
    frame_dff_threshold = frame_dff[frame_dff["Amplitude"] == frame_dff["Threshold"]]
    frame_model_df, frame_mean_sem, frame_comp = frame_model(frame_dff_threshold)

    # saved_framed_model_df = pd.read_csv("C:/Users/cvandromme/Desktop/frame_model_df.csv")
    # frame_model_df_amp_gp = frame_model_df.groupby(["Genotype", "ID", "Threshold", "Amplitude"], as_index=False).mean().drop(columns=["Trial", "Duration", "Behavior"])
    # plot_hit_miss_classif(saved_framed_model_df)
    # data = plot_hit_miss_classif_comp(saved_framed_model_df, gp1="WT", gp2="KO", title_precision="['EXC',_'INH'][0,_1,_-1]_-_db_cv=True")

    # accuracy_comp_df = compare_accuracy(recs.values(), random=42)
    # acc_nb_df = correlate_nb_accuracy(recs, accuracy_comp_df, threshold="median")