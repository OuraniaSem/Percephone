# region ======================================== Imports ==============================================================
import os
import random

import mplcursors
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pingouin as pg
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from scipy.stats import pointbiserialr, linregress, binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from itertools import product
from tqdm import tqdm

import percephone.core.recording as pc
import percephone.plts.stats as ppt
import percephone.plts.style as sty
# from Figures.noise_assessment import get_mean_trial_activity_df, ntn_cosine_similarity
from Figures.stimulus_encoding import get_features

# endregion

def get_activity_by_frame_df(recs, zscore=True, BMS=False):
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
        if BMS:
            rec_id = f"{rec.filename}-{rec.genotype.split("-")[1]}"
        else:
            rec_id = rec.filename
        activity_vector = [rec.zscore_exc, rec.zscore_inh] if zscore else [rec.df_f_exc, rec.df_f_inh]
        for n_type, activity in zip(["EXC", "INH"], activity_vector):
            resp = np.array(rec.matrices[n_type]["Responsivity"])
            for neuron_id, neuron in enumerate(np.array(activity)):
                for (trial_id, trial_time), trial_duration, trial_amp, trial_label in zip(enumerate(rec.stim_time),
                                                                                          rec.stim_durations,
                                                                                          rec.stim_ampl, rec.detected_stim):
                    row = {"Genotype": rec.genotype, "ID": rec_id, "Threshold": rec.session_threshold,
                           "Trial": trial_id, "Amplitude": trial_amp, "Duration": trial_duration,
                           "Behavior": trial_label, "n_type": n_type, "resp": resp[neuron_id, trial_id], "n_ID": neuron_id}
                    for frame_id, frame in enumerate(range(trial_time - 30, trial_time + 30)):
                        row[frame_id] = neuron[frame]
                    rows.append(row)
    return pd.DataFrame(rows)


def sliding_window_average(df, header_cols, window_size, sum=False):
    """
    Applies a sliding-window average across the numeric columns of a DataFrame.

    Parameters:
    - df: pandas.DataFrame containing header columns and numeric columns labeled 0..N-1.
    - header_cols: list of column names in df that should remain unchanged.
    - window_size: int, size of the sliding window (must be >= 1).

    Returns:
    - A new DataFrame with the same columns and shape as df.
      Header columns are copied as-is, and numeric columns are replaced by their windowed averages.

    Behavior:
    - For interior columns, the value at position i is the mean of columns [i - k, ..., i, ..., i + k],
      where k = window_size // 2.
    - For edge columns (where a full window would extend beyond 0 or N-1), the window is truncated
      and only the available columns are averaged (min_periods=2).
    """
    numeric_cols = [col for col in df.columns if col not in header_cols]
    if sum:
        rolled = df[numeric_cols].rolling(window=window_size, axis=1, center=True, min_periods=2).sum()
    else:
        rolled = df[numeric_cols].rolling(window=window_size, axis=1, center=True, min_periods=2).mean()
    # Reconstruct the DataFrame: keep headers, replace numeric columns with rolled values.
    result_df = pd.concat([df[header_cols].reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)
    # Ensure numeric column names stay as ints (or as they were originally).
    result_df.columns = header_cols + [int(c) for c in numeric_cols]
    return result_df


def aggregate_every_3cols(df, header_cols, window_size=3):
    """
    For a DataFrame with some non-numeric "header" columns and 60 numeric columns,
    returns a new DataFrame with the same header columns plus one column per
    consecutive block of 3 numeric columns, each equal to their rowwise mean.
    """
    # 1) Split off non-numeric headers
    numeric_cols = [col for col in df.columns if col not in header_cols]
    header = df[header_cols]
    nums = df[numeric_cols]
    # 2) Check we have a multiple of 3 (optional)
    if len(nums.columns) % window_size != 0:
        raise ValueError(f"Expected a multiple of {window_size} numeric columns, got {len(nums.columns)}")
    # 3) For each block of 3, compute the mean
    new_cols = {}
    cols = list(nums.columns)
    for block_idx in range(0, len(cols), window_size):
        trio = cols[block_idx:block_idx + window_size]
        # name the new column after the first of the trio (or anything you like)
        new_name = f"{trio[0]}_to_{trio[-1]}_mean"
        new_cols[new_name] = nums[trio].mean(axis=1)
    # 4) Concatenate header + new means
    result = pd.concat([header, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return result

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


def frame_model(frame_data, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling",
                sliding_window=None, window_sum=False, window=None):
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
    data = frame_data.copy()
    assert (sliding_window is None or window is None), "Please choose between each frame, sliding window, and window"
    header_columns = ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Duration", "Behavior", "n_type", "resp", "n_ID"]
    if sliding_window is not None:
        data = sliding_window_average(frame_data, header_columns, sliding_window, sum=window_sum)
    if window is not None:
        data = aggregate_every_3cols(frame_data, header_columns, window_size=window)
    # Filtering the neuron types and activity
    data = data[data["n_type"].isin(neuron_type)]
    data = data[data["resp"].isin(resp_type)]
    data = data.drop(columns=["Threshold", "Amplitude", "Duration", "resp"])
    index_columns = ["Genotype", "ID", "Trial", "Behavior", "n_type", "n_ID"]
    numeric_columns = [c for c in data.columns if c not in header_columns]
    rows = []
    # Creating a pivot DataFrame to obtain a dataframe per frame, each column being a neuron
    for frame in tqdm(numeric_columns):
        frame_data = frame_data[frame_data["Amplitude"] != 0]
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
                if balancing_method == "resampling":
                    undersampler = RandomUnderSampler(random_state=42)
                    X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)
                    lr = LogisticRegression(solver='lbfgs', C=1, max_iter=5000)
                    lr.fit(X_train_res, y_train_res)
                elif balancing_method == "weights":
                    # Uses the weight parameter of the LR to balance the weight of samples rather than resampling
                    lr = LogisticRegression(solver='lbfgs', class_weight="balanced", C=1, max_iter=5000)
                    lr.fit(X_train, y_train)
                elif balancing_method == None:
                    lr = LogisticRegression(solver='lbfgs', C=1, max_iter=5000)
                    lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)
                metrics = get_metrics(y_test, y_pred)
            row = {"Genotype": rec_data["Genotype"].values[0], "ID": rec_data["ID"].values[0],
                   "Frame": frame, "TPR": metrics["TPR"], "FPR": metrics["FPR"], "Accuracy": metrics["Accuracy"],
                   "TPR_shuffle": metrics["TPR_shuffle"], "FPR_shuffle": metrics["FPR_shuffle"], "Accuracy_shuffle": metrics["Accuracy_shuffle"]}
                   # "p_hit": metrics["p_hit"], "p_miss": metrics["p_miss"]}
            rows.append(row)
    frame_model_df = pd.DataFrame(rows)
    frame_mean_sem = plot_hit_miss_classif(frame_model_df, title_precision=f"{neuron_type}{resp_type} - Double CV=={db_cv}", timescale_division_factor=(1 if window is None else window))
    # frame_comp = plot_hit_miss_classif_comp(frame_model_df, gp1="WT", gp2="KO-Hypo", title_precision=f"{neuron_type}{resp_type} - db_cv={db_cv}")
    return frame_model_df, frame_mean_sem #, frame_comp


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
    y_true = []
    y_pred = []
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
        # 1) Training the model on shuffled labels to compare its performance
        y_train_res_shuffled = y_train_res.copy()
        random.shuffle(y_train_res_shuffled)
        final_model.fit(X_train_res, y_train_res_shuffled)
        # Assessing the final model's performance
        y_val_pred_shuffle = final_model.predict(X_val)
        outer_metrics_shuffle = get_metrics(y_val, y_val_pred_shuffle)
        outer_metrics_shuffle = {f"{k}_shuffle": v for k, v in outer_metrics_shuffle.items()}
        row.update(outer_metrics_shuffle)
        rows.append(row)
        # 2) Rebuilding true and predicted label vectors to compare the obtained results to chance with binomial test
        # y_true.extend(y_val)
        # y_pred.extend(y_val_pred)
    # Computing the chance that the obtained classification is better than chance levels
    # nb_TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    # nb_hit = int(np.sum(y_true))
    # p_hit = np.nan if nb_hit==0 else binomtest(nb_TP, nb_hit, p=0.5, alternative="greater").pvalue
    # print(f"Nb TP: {nb_TP}, Nb hit: {nb_hit}, P: {p_hit}")
    # nb_FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    # nb_miss = int(len(y_true) - nb_hit)
    # p_miss = np.nan if nb_miss==0 else binomtest(nb_FP, nb_miss, p=0.5, alternative="less").pvalue
    # print(f"Nb FP: {nb_FP}, Nb miss: {nb_miss}, P: {p_miss}")
    results_df = pd.DataFrame(rows)
    results_metrics = results_df.mean(numeric_only=True).to_dict()
    # results_metrics.update({"p_hit": p_hit, "p_miss": p_miss})
    return results_df if get_df else results_metrics


def plot_hit_miss_classif(frame_model_df, title_precision="", shuffle=False, shuffle_frame_significance=False,
                          timescale_division_factor=1):
    """
    Plot the hit vs miss classification graph from a Dataframe with each line containing infos about TPR and TNR for 1
    frame for one animal.

    Parameters
    ----------
    frame_model_df

    Returns
    -------

    """
    draw_style = "default" if timescale_division_factor == 1 else "steps-post"
    fill_style = None if timescale_division_factor == 1 else "post"
    if "WT-BMS" in frame_model_df["Genotype"].unique():
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(28, 8), constrained_layout=True)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 8), constrained_layout=True)
    for i, genotype in enumerate(frame_model_df["Genotype"].unique()):
        data = frame_model_df[frame_model_df["Genotype"] == genotype].drop(columns=["Genotype", "ID"])
        # Curves definition
        tpr_mean = data.groupby("Frame")["TPR"].mean().values
        tpr_sem = data.groupby("Frame")["TPR"].sem().values
        fpr_mean = data.groupby("Frame")["FPR"].mean().values
        fpr_sem = data.groupby("Frame")["FPR"].sem().values
        # Curves plotting
        x = np.arange(len(tpr_mean))
        ax[i].plot(x, tpr_mean, label="Hit accuracy", color=sty.color_dict[genotype][0], lw=2, drawstyle=draw_style)
        ax[i].fill_between(x, tpr_mean - tpr_sem, tpr_mean + tpr_sem, color=sty.color_dict[genotype][0], alpha=0.3, step=fill_style)
        ax[i].plot(x, fpr_mean, label="Miss accuracy", color=sty.color_dict[genotype][1], lw=2, drawstyle=draw_style)
        ax[i].fill_between(x, fpr_mean - fpr_sem, fpr_mean + fpr_sem, color=sty.color_dict[genotype][1], alpha=0.3, step=fill_style)
        if shuffle:
            # Shuffle curves definition
            tpr_mean_shuffle = data.groupby("Frame")["TPR_shuffle"].mean().values
            tpr_sem_shuffle = data.groupby("Frame")["TPR_shuffle"].sem().values
            fpr_mean_shuffle = data.groupby("Frame")["FPR_shuffle"].mean().values
            fpr_sem_shuffle = data.groupby("Frame")["FPR_shuffle"].sem().values
            # Shuffle curves plotting
            x = np.arange(len(tpr_mean))
            ax[i].plot(x, tpr_mean_shuffle, label="Hit accuracy", color=sty.color_dict[genotype][0], lw=1, ls="dotted", alpha=0.25, drawstyle=draw_style)
            ax[i].fill_between(x, tpr_mean_shuffle - tpr_sem_shuffle, tpr_mean_shuffle + tpr_sem_shuffle, color=sty.color_dict[genotype][0], alpha=0.1, step=fill_style)
            ax[i].plot(x, fpr_mean_shuffle, label="Miss accuracy", color=sty.color_dict[genotype][1], lw=1, ls="dotted", alpha=0.25, drawstyle=draw_style)
            ax[i].fill_between(x, fpr_mean_shuffle - fpr_sem_shuffle, fpr_mean_shuffle + fpr_sem_shuffle, color=sty.color_dict[genotype][1], alpha=0.1, step=fill_style)
            if shuffle_frame_significance:
                # Computing the statistical difference with shuffle
                pvals = {}
                for frame in data.Frame.unique():
                    real_tpr = data[data.Frame == frame]["TPR"]
                    shuffle_tpr = data[data.Frame == frame]["TPR_shuffle"]
                    real_fpr = data[data.Frame == frame]["FPR"]
                    shuffle_fpr = data[data.Frame == frame]["FPR_shuffle"]
                    # paired test across animals
                    # drop any animals missing one of the two
                    idx = real_tpr.index.intersection(shuffle_tpr.index)
                    if len(idx) >= 3:
                        stat, p_tpr = wilcoxon(real_tpr.loc[idx], shuffle_tpr.loc[idx], alternative="greater")
                        stat, p_fpr = wilcoxon(real_fpr.loc[idx], shuffle_fpr.loc[idx], alternative="less")
                    else:
                        p_tpr = np.nan
                        p_fpr = np.nan
                    pvals[frame] = [p_tpr, p_fpr]
                # Drawing a bar above significantly different frames
                for f, p_list in pvals.items():
                    p_tpr = p_list[0]
                    p_fpr = p_list[1]
                    if p_tpr < 0.05:
                        # draw a tiny horizontal line spanning the width of one frame
                        ax[i].hlines(0.95, f - 0.4, f + 0.4, color=sty.color_dict[genotype][0], linewidth=2)
                    if p_fpr < 0.05:
                        # draw a tiny horizontal line spanning the width of one frame
                        ax[i].hlines(0.05, f - 0.4, f + 0.4, color=sty.color_dict[genotype][1], linewidth=2)

        # Delimitation of the stimulus period and chance level
        ax[i].axvline(x=30/timescale_division_factor, ls="--", lw=1, color="red")
        ax[i].axvline(x=45/timescale_division_factor, ls="--", lw=1, color="black")
        ax[i].axhline(y=0.5, ls="--", lw=1, color="gray")
        # Title and axes formatting
        ax[i].set_title(genotype, color=sty.color_dict[genotype][0], fontsize=20)
        ax[i].set_ylim(0, 1)
        ax[i].set_xlabel("Time (s)")
        ax[i].set_xticks(np.divide([0, 15, 30, 45, 60], timescale_division_factor))
        ax[i].set_xticklabels([-1, -0.5, 0, 0.5, 1])
    fig.suptitle(f"Hit versus Miss classification graph using the mean ΔF/F\n{title_precision}", fontsize=20)
    fig.canvas.manager.set_window_title(f"Hit_Miss_classif_{title_precision}")
    plt.show()
    return data


def plot_hit_miss_classif_comp(frame_model_df, gp1="WT", gp2="KO-Hypo", title_precision=""):
    period_dict = {"stim": [30, 45], "start_stim (250ms)": [30, 37], "end_stim (250ms)": [37, 45], "pre_stim (200ms)": [24, 30]}
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 16), constrained_layout=True)
    fig_shuf, ax_shuf = plt.subplots(nrows=4, ncols=4, figsize=(24, 32), constrained_layout=True)
    data_dict = {}
    for col, period in enumerate(period_dict.keys()):
        start, end = period_dict[period]
        data = frame_model_df[frame_model_df["Frame"].isin(range(start, end))].groupby(["Genotype", "ID"], as_index=False).mean().drop(columns="Frame")
        # Plotting the comparison of accuracy between genotypes
        ppt.boxplot(ax[0, col], data[data["Genotype"] == gp1]["TPR"], data[data["Genotype"] == gp2]["TPR"], ylabel="Hit accuracy",
                    paired=False, title=period, ylim=[0, 1], colors=[sty.color_dict[gp1][0], sty.color_dict[gp2][0]], det_marker=True, force_markers_identity=True)
        ppt.boxplot(ax[1, col], data[data["Genotype"] == gp1]["FPR"], data[data["Genotype"] == gp2]["FPR"], ylabel="Miss error",
                    paired=False, title=period, ylim=[0, 1], colors=[sty.color_dict[gp1][1], sty.color_dict[gp2][1]], det_marker=False, force_markers_identity=False)
        # Plotting the comparisons with shuffled data
        ppt.boxplot(ax_shuf[0, col], data[data["Genotype"] == gp1]["TPR"].values, data[data["Genotype"] == gp1]["TPR_shuffle"].values, ylabel="Hit accuracy",
                    paired=True, title=f"{period}\n{gp1}: Real vs. Shuffled", ylim=[0, 1], colors=[sty.color_dict[gp1][0], "darkgray"], det_marker=True, force_markers_identity=True)
        ppt.boxplot(ax_shuf[1, col], data[data["Genotype"] == gp1]["FPR"].values, data[data["Genotype"] == gp1]["FPR_shuffle"].values, ylabel="Miss error",
                    paired=True, title=f"{period}\n{gp1}: Real vs. Shuffled", ylim=[0, 1], colors=[sty.color_dict[gp1][1], "gray"], det_marker=False, force_markers_identity=True)
        ppt.boxplot(ax_shuf[2, col], data[data["Genotype"] == gp2]["TPR"].values, data[data["Genotype"] == gp2]["TPR_shuffle"].values, ylabel="Hit accuracy",
                    paired=True, title=f"{period}\n{gp2}: Real vs. Shuffled", ylim=[0, 1], colors=[sty.color_dict[gp2][0], "darkgray"], det_marker=True, force_markers_identity=True)
        ppt.boxplot(ax_shuf[3, col], data[data["Genotype"] == gp2]["FPR"].values, data[data["Genotype"] == gp2]["FPR_shuffle"].values, ylabel="Miss error",
                    paired=True, title=f"{period}\n{gp2}: Real vs. Shuffled", ylim=[0, 1], colors=[sty.color_dict[gp2][1], "gray"], det_marker=False, force_markers_identity=True)
        data_dict[period] = data
    fig.suptitle(f"Hit accuracy miss error comparison ({gp1}/{gp2})\n{title_precision}", fontsize=20)
    fig.canvas.manager.set_window_title(f"Hit_Miss_classif_comp_{gp1}_{gp2}_{title_precision}")
    fig_shuf.suptitle(f"Shuffle comparison\n{title_precision}", fontsize=20)
    fig_shuf.canvas.manager.set_window_title(f"Hit_Miss_classif_comp_shuffle{title_precision}")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/14/shuffle_{gp1}_{gp2}.pdf", format="pdf")
    # plt.savefig(f"Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/5_Figure2/shuffle_{gp1}_{gp2}.pdf", format="pdf")
    plt.show()
    return data_dict

def anova_accuracy(frame_model_df):
    period_dict = {"stim": [30, 45], "start_stim (250ms)": [30, 37], "end_stim (250ms)": [37, 45],
                   "pre_stim (200ms)": [24, 30]}
    start, end = period_dict["stim"]
    data = frame_model_df[frame_model_df["Frame"].isin(range(start, end))].groupby(["Genotype", "ID"], as_index=False).mean().drop(columns="Frame")
    aov_hit = pg.anova(data=data, dv="TPR", between="Genotype")
    aov_miss = pg.anova(data=data, dv="FPR", between="Genotype")
    print("=== Hit accuracy ===")
    print(aov_hit)
    print("=== Miss error ===")
    print(aov_miss)
    return data

# anova_data = anova_accuracy(frame_model_df_res)

def compare_accuracy(recs, random=42):
    """
    Train a logistic regression model with different subsets of neurons and compare the decoding accuracy.
    The mean neuronal activity during the stimulus period is used to train the model
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
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(24, 24), constrained_layout=True)
    for row, (group, gp_label) in enumerate(zip([data, data[data["Genotype"] == "WT"], data[data["Genotype"] == "KO-Hypo"]],
                                                ["all", "WT", "KO-Hypo"])):
        ppt.boxplot(ax[row, 0], group["acc_all"].values, group["acc_exc"].values, ylabel="Accuracy", paired=True, title=f"All neurons/EXC ({gp_label})", ylim=[0, 1],
                    colors=[sty.exc_inh_color, sty.exc_color], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 1], group["acc_all"].values, group["acc_inh"].values, ylabel="Accuracy", paired=True, title=f"All neurons/INH ({gp_label})", ylim=[0, 1],
                    colors=[sty.exc_inh_color, sty.inh_color], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[row, 2], group["acc_exc"].values, group["acc_inh"].values, ylabel="Accuracy", paired=True, title=f"EXC/INH ({gp_label})", ylim=[0, 1],
                    colors=[sty.exc_color, sty.inh_color], det_marker=False, force_markers_identity=False)
    wt = data[data["Genotype"] == "WT"]
    hypo = data[data["Genotype"] == "KO-Hypo"]
    for row, acc_type in enumerate(["acc_all", "acc_exc", "acc_inh"]):
        ppt.boxplot(ax[row, 3], wt[acc_type].values, hypo[acc_type].values, ylabel="Accuracy", paired=False, title=f"WT/KO-Hypo ({acc_type})", ylim=[0, 1],
                    colors=[sty.wt_color, sty.hypo_color], det_marker=False, force_markers_identity=False)
    fig.suptitle("Comparison of decoding accuracy between neuron types")
    fig.canvas.manager.set_window_title(f"Accuracy n_types")
    # plt.show()
    return data


def correlate_nb_accuracy(recs, accuracy_df, threshold="median"):
    accuracy_df["n_EXC"] = accuracy_df["ID"].map(lambda id_: recs[id_].zscore_exc.shape[0])
    accuracy_df["n_INH"] = accuracy_df["ID"].map(lambda id_: recs[id_].zscore_inh.shape[0])
    accuracy_df["n_all"] = accuracy_df["n_EXC"] + accuracy_df["n_INH"]

    def plot_lin_reg(ax, data, x_col=None, y_col=None, id_col="ID", group_col=None,
                     title=None, xlab=None, ylab=None, colors=None, line_color="red", id_display=True):
        if colors is None:
            colors = {"WT": sty.wt_color, "KO": sty.ko_color, "KO-Hypo": sty.hypo_color}
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
                    colors=[sty.exc_color, sty.exc_color], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[2, col], low_perf["n_INH"].values, high_perf["n_INH"].values, ylabel="n_INH", paired=False,
                    title=f"{accuracy_metric} Low/High acc", ylim=[],
                    colors=[sty.inh_color, sty.inh_color], det_marker=False, force_markers_identity=False)
        ppt.boxplot(ax[3, col], low_perf["n_all"].values, high_perf["n_all"].values, ylabel="n_all", paired=False,
                    title=f"{accuracy_metric} Low/High acc", ylim=[],
                    colors=[sty.exc_inh_color, sty.exc_inh_color], det_marker=False, force_markers_identity=False)


    fig.suptitle(f"Correlation of the model accuracy with the number of neurons.\nComparison of the number of neurons "
                 f"between high and low accuracy animal (threshold={threshold})", fontsize=12)
    fig.canvas.manager.set_window_title("Corr acc nb neurons")
    plt.show()
    return accuracy_df


def correlate_frame_accuracy_metrics(frame_model_df, features_df, period="stim"):
    """Correlates the model accuracy during the stim period with various metrics to identify which feature grasp the
    more information"""
    # Getting a Dataframe of the model mean accuracy during stim, one row per animal
    period_dict = {"stim": [30, 45], "start": [30, 37], "end": [37, 45], "pre_stim": [24, 30]}
    start, end = period_dict[period]
    data_acc = frame_model_df[frame_model_df["Frame"].isin(range(start, end))].groupby(["Genotype", "ID"], as_index=False).mean().drop(columns="Frame")
    # Getting the features DataFrame, defining a single metric per animal, only responsive neurons are considered
    data_features = features_df[features_df.amplitude != 0].groupby(["ID", "Genotype", "behavior"], as_index=False).mean().drop(columns=["bounded_x0", "amplitude", "threshold"])
    data_features_all = features_df[features_df.amplitude != 0].drop(columns="behavior").groupby(["ID", "Genotype"], as_index=False).mean().drop(columns=["bounded_x0", "amplitude", "threshold"])
    data = data_features.merge(data_acc[["ID", "TPR", "FPR", "Accuracy"]], on="ID", how="left")
    data_hit = data[data.behavior == True]
    data_miss = data[data.behavior == False]
    data_all = data_features_all.merge(data_acc[["ID", "TPR", "FPR", "Accuracy"]], on="ID", how="left")
    cols_id = ["ID", "Genotype", "behavior"]
    cols_acc = ["Accuracy", "TPR", "FPR"]
    cols_features = [col for col in data.columns if ((col not in cols_id) and (col not in cols_acc))]
    # Plotting the correlation of the different features with the model accuracy for all genotypes
    fig, ax = plt.subplots(nrows=9, ncols=20, figsize=(100, 45), constrained_layout=True)
    rows = []
    for row, (data, trial_type) in enumerate(zip([data_all, data_hit, data_miss], ["All", "Hit", "Miss"])):
        for sub_row, acc in enumerate(cols_acc):
            for col, feature in enumerate(cols_features):
                # Correlation global
                y_col = data[acc]
                x_col = data[feature]
                results = dict(linregress(x_col, y_col)._asdict())
                r2 = results["rvalue"] ** 2
                line = results["slope"] * x_col + results["intercept"]
                # Correlation WT
                y_col_wt = data[data.Genotype == "WT"][acc]
                x_col_wt = data[data.Genotype == "WT"][feature]
                results_wt = dict(linregress(x_col_wt, y_col_wt)._asdict())
                r2_wt = results_wt["rvalue"] ** 2
                line_wt = results_wt["slope"] * x_col_wt + results_wt["intercept"]
                # Correlation KO-Hypo
                y_col_hypo = data[data.Genotype == "KO-Hypo"][acc]
                x_col_hypo = data[data.Genotype == "KO-Hypo"][feature]
                results_hypo = dict(linregress(x_col_hypo, y_col_hypo)._asdict())
                r2_hypo = results_hypo["rvalue"] ** 2
                line_hypo = results_hypo["slope"] * x_col_hypo + results_hypo["intercept"]
                # Plot the data points and regression lines
                ax[3 * row + sub_row, col].plot(x_col, line, color="black", lw=2)
                ax[3 * row + sub_row, col].plot(x_col_wt, line_wt, color=sty.wt_color, lw=2)
                ax[3 * row + sub_row, col].plot(x_col_hypo, line_hypo, color=sty.hypo_color, lw=2)
                for g in sorted(data["Genotype"].unique()):
                    group = data[data["Genotype"] == g]
                    sc = ax[3 * row + sub_row, col].scatter(group[feature], group[acc], color=sty.color_dict[g][0], alpha=0.7, s=10, marker="+")
                # Annotate the plot with R² and p-value
                ax[3 * row + sub_row, col].text(0.05, 0.95, f"$r^2={r2:.3f}$ p-val={results["pvalue"]:.3f}", transform=ax[3 * row + sub_row, col].transAxes, fontsize=8, verticalalignment="top", color="black")
                ax[3 * row + sub_row, col].text(0.05, 0.90, f"$r^2={r2_wt:.3f}$ p-val={results_wt["pvalue"]:.3f}", transform=ax[3 * row + sub_row, col].transAxes, fontsize=8, verticalalignment="top", color=sty.wt_color)
                ax[3 * row + sub_row, col].text(0.05, 0.85, f"$r^2={r2_hypo:.3f}$ p-val={results_hypo["pvalue"]:.3f}", transform=ax[3 * row + sub_row, col].transAxes, fontsize=8, verticalalignment="top", color=sty.hypo_color)
                ax[3 * row + sub_row, col].set_title(f"{trial_type} trials", fontsize=10)
                ax[3 * row + sub_row, col].set_xlabel(feature, fontsize=10)
                ax[3 * row + sub_row, col].set_ylabel(acc, fontsize=10)
                ax[3 * row + sub_row, col].set_ylim(ymin=0, ymax=1)
                ax[3 * row + sub_row, col].tick_params(axis='both', which='major', labelsize=5)
                rows.append({"Trials": trial_type, "Acc_metric": acc, "Feature": feature,
                             "r2": r2, "pval": results["pvalue"],
                             "r2_wt": r2_wt, "pval_wt": results_wt["pvalue"],
                             "r2_hypo": r2_hypo, "pval_hypo": results_hypo["pvalue"]})
    results = pd.DataFrame(rows)
    fig.suptitle(f"Correlation of the frame model accuracy during the whole {period} period with the different metrics computed on responsive neurons")
    # plt.savefig(f"{server_address}/response_decoding/corr_acc_model_features_{period}.pdf", format="pdf")
    return data, results

def correlate_frame_accuracy_ntn_df(frame_model_df, ntn_df, period="stim"):
    """Correlates the model accuracy during the period with the ntn_cosim"""
    # Getting a Dataframe of the model mean accuracy during stim, one row per animal
    period_dict = {"stim": [30, 45], "start": [30, 37], "end": [37, 45], "pre_stim": [24, 30]}
    start, end = period_dict[period]
    data_acc = frame_model_df[frame_model_df["Frame"].isin(range(start, end))].groupby(["Genotype", "ID"], as_index=False).mean().drop(columns="Frame")
    # Getting the features DataFrame, defining a single metric per animal, only responsive neurons are considered
    data = ntn_df.merge(data_acc[["ID", "TPR", "FPR", "Accuracy"]], on="ID", how="left").drop(columns=["Threshold"])
    data_hit = data[data.Behavior == "Hit"]
    data_miss = data[data.Behavior == "Miss"]
    data_all = data[data.Behavior == "All"]
    cols_id = ["ID", "Genotype", "Behavior"]
    cols_acc = ["Accuracy", "TPR", "FPR"]
    cols_features = [col for col in data.columns if ((col not in cols_id) and (col not in cols_acc))]
    n_features = len(cols_features)
    # Plotting the correlation of the different features with the model accuracy for all genotypes
    fig, ax = plt.subplots(nrows=9, ncols=n_features, figsize=(n_features * 5, 45), constrained_layout=True)
    rows = []
    for row, (data, trial_type) in enumerate(zip([data_all, data_hit, data_miss], ["All", "Hit", "Miss"])):
        for sub_row, acc in enumerate(cols_acc):
            for col, feature in enumerate(cols_features):
                # Correlation global
                y_col = data[acc]
                x_col = data[feature]
                results = dict(linregress(x_col, y_col)._asdict())
                r2 = results["rvalue"] ** 2
                line = results["slope"] * x_col + results["intercept"]
                # Correlation WT
                y_col_wt = data[data.Genotype == "WT"][acc]
                x_col_wt = data[data.Genotype == "WT"][feature]
                results_wt = dict(linregress(x_col_wt, y_col_wt)._asdict())
                r2_wt = results_wt["rvalue"] ** 2
                line_wt = results_wt["slope"] * x_col_wt + results_wt["intercept"]
                # Correlation KO-Hypo
                y_col_hypo = data[data.Genotype == "KO-Hypo"][acc]
                x_col_hypo = data[data.Genotype == "KO-Hypo"][feature]
                results_hypo = dict(linregress(x_col_hypo, y_col_hypo)._asdict())
                r2_hypo = results_hypo["rvalue"] ** 2
                line_hypo = results_hypo["slope"] * x_col_hypo + results_hypo["intercept"]
                # Defining the axis
                if n_features == 1:
                    axis = ax[3 * row + sub_row]
                else:
                    axis = ax[3 * row + sub_row, col]
                # Plot the data points and regression lines
                axis.plot(x_col, line, color="black", lw=2)
                axis.plot(x_col_wt, line_wt, color=sty.wt_color, lw=2)
                axis.plot(x_col_hypo, line_hypo, color=sty.hypo_color, lw=2)
                for g in sorted(data["Genotype"].unique()):
                    group = data[data["Genotype"] == g]
                    sc = axis.scatter(group[feature], group[acc], color=sty.color_dict[g][0], alpha=0.7, s=10, marker="+")
                # Annotate the plot with R² and p-value
                axis.text(0.05, 0.95, f"$r^2={r2:.3f}$ p-val={results["pvalue"]:.3f}", transform=axis.transAxes, fontsize=8, verticalalignment="top", color="black")
                axis.text(0.05, 0.90, f"$r^2={r2_wt:.3f}$ p-val={results_wt["pvalue"]:.3f}", transform=axis.transAxes, fontsize=8, verticalalignment="top", color=sty.wt_color)
                axis.text(0.05, 0.85, f"$r^2={r2_hypo:.3f}$ p-val={results_hypo["pvalue"]:.3f}", transform=axis.transAxes, fontsize=8, verticalalignment="top", color=sty.hypo_color)
                axis.set_title(f"{trial_type} trials", fontsize=10)
                axis.set_xlabel(feature, fontsize=10)
                axis.set_ylabel(acc, fontsize=10)
                axis.set_ylim(ymin=0, ymax=1)
                axis.tick_params(axis='both', which='major', labelsize=5)
                rows.append({"Trials": trial_type, "Acc_metric": acc, "Feature": feature,
                             "r2": r2, "pval": results["pvalue"],
                             "r2_wt": r2_wt, "pval_wt": results_wt["pvalue"],
                             "r2_hypo": r2_hypo, "pval_hypo": results_hypo["pvalue"]})
    results = pd.DataFrame(rows)
    fig.suptitle(f"Correlation of the frame model accuracy during the whole {period} period with neuron to neuron global cosine similarity")
    # plt.savefig(f"{server_address}response_decoding/corr_acc_model_ntn_cosim_{period}.pdf", format="pdf")
    return data, results

# endregion ============================================================================================================



if __name__ == '__main__':
    BMS_analysis = True
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
    if BMS_analysis:
        recs = {f"{ar.get().filename}-{ar.get().genotype.split("-")[1]}": ar.get() for ar in async_results}
    else:
        recs = {ar.get().filename: ar.get() for ar in async_results}

    for rec in recs.values():
        # rec.auc_neg()
        rec.peak_delay_amp()
    # endregion ============
    # test = perceptual_magnitude_behavior_corr(recs.values())
    # mean_corr = test.groupby("Genotype").mean()
    # glmm_behavior(data)

    # frame_data = get_activity_by_frame_df(recs.values(), zscore=True)
    # corr_data = correlate_mean_zscore_behavior_frame(frame_data[frame_data["Amplitude"] == 12])
    # plot_frame_correlation(corr_data)

    frame_dff = get_activity_by_frame_df(recs.values(), zscore=False, BMS=BMS_analysis)
    # features_df = get_features(recs.values(), amp_delay=True, auc=True)
    # activity_long_df = get_mean_trial_activity_df(recs.values(), zscore=True)
    # ntn_df = ntn_cosine_similarity(activity_long_df, amplitude="all", nogo=False, metric="Resp", filter_out_nr=False)
    # frame_dff_3 = aggregate_every_3cols(frame_dff, ["Genotype", "ID", "Threshold", "Trial", "Amplitude", "Duration", "Behavior", "n_type", "resp", "n_ID"], window_size=3)

    frame_model_df_res = pd.read_csv("Z:/Current_members/Ourania_Semelidou/2p/Figures_paper & submissions/202507/5_Figure2/frame_model_df_res.csv")
    # frame_model_df_avg = pd.read_csv("C:/Users/cvandromme/Desktop/Tactile_detection/Analysis_data/frame_model_df_avg3.csv")
    # frame_model_df_bms, frame_mean_sem_bms = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling")
    # frame_model_df_res, frame_mean_sem_res = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling")
    # frame_model_df_avg, frame_mean_sem_avg = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling", sliding_window=3, window_sum=False)
    # frame_model_df_3, frame_mean_sem_3 = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling", window=3)
    # frame_model_df_5, frame_mean_sem_5 = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling", window=5)
    # frame_model_df_2, frame_mean_sem_2 = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling", window=2)
    # frame_model_df_4, frame_mean_sem_4 = frame_model(frame_dff, neuron_type=["EXC", "INH"], resp_type=[0, 1, -1], db_cv=True, balancing_method="resampling")
    plot_df = plot_hit_miss_classif(frame_model_df_res, title_precision="All neurons, all patterns, doubleCV", shuffle=True, timescale_division_factor=1)
    data_comp = plot_hit_miss_classif_comp(frame_model_df_res, gp1="WT", gp2="KO", title_precision="All neurons, all patterns, doubleCV")
    # plot_df = plot_hit_miss_classif(frame_model_df_4, title_precision="All neurons, all patterns, doubleCV, non-sliding avg4", shuffle=True, timescale_division_factor=4)
    plot_hit_miss_classif_comp(frame_model_df_bms, gp1="WT-DMSO", gp2="WT-BMS", title_precision="All neurons, all patterns, doubleCV")
    plot_hit_miss_classif_comp(frame_model_df_bms, gp1="KO-DMSO", gp2="KO-BMS", title_precision="All neurons, all patterns, doubleCV")
    data_comp = plot_hit_miss_classif_comp(frame_model_df_bms, gp1="WT-DMSO", gp2="KO-DMSO", title_precision="All neurons, all patterns, doubleCV")
    plot_hit_miss_classif_comp(frame_model_df_bms, gp1="WT-BMS", gp2="KO-BMS", title_precision="All neurons, all patterns, doubleCV")

    # corr_acc_features_df, corr_results_df = correlate_frame_accuracy_metrics(frame_model_df_res, features_df, period="stim")
    # corr_acc_ntn_df, corr_ntn_results_df = correlate_frame_accuracy_ntn_df(frame_model_df_res, ntn_df, period="stim")

    # saved_framed_model_df = pd.read_csv("C:/Users/cvandromme/Desktop/frame_model_df.csv")
    # frame_model_df_amp_gp = frame_model_df.groupby(["Genotype", "ID", "Threshold", "Amplitude"], as_index=False).mean().drop(columns=["Trial", "Duration", "Behavior"])
    # plot_hit_miss_classif(saved_framed_model_df)
    # data = plot_hit_miss_classif_comp(saved_framed_model_df, gp1="WT", gp2="KO", title_precision="['EXC',_'INH'][0,_1,_-1]_-_db_cv=True")

    # recs_wt = {k: v for k, v in recs.items() if v.genotype == "WT"}
    # recs_hypo = {k: v for k, v in recs.items() if v.genotype == "KO-Hypo"}
    accuracy_comp_df = compare_accuracy(recs.values(), random=42)
    # acc_nb_df = correlate_nb_accuracy(recs, accuracy_comp_df[accuracy_comp_df["Genotype"] == "KO"], threshold="median")