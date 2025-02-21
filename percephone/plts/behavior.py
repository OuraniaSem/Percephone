"""
Théo Gauvrit 18/01/2024
Different plots link to behavior
"""
import mplcursors
from scipy.stats import stats
from sklearn.linear_model import MultiTaskLasso

import percephone.core.recording as pc
import percephone.analysis.utils as pu
import percephone.plts.stats as ppt
import percephone.utils.math_formulas as mf
import numpy as np
import pandas as pd
import scipy.stats as ss
from multiprocessing import Pool, cpu_count, pool
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import mean_squared_error


font_s = 10


def psycho_like_plot(rec, roi_info, ax):
    """
    Generates the psychometric curve from the data found in the ROI file.

    Parameters
    ----------
    rec : Recording
        The rec object to be plotted.
    roi_info : dict
        The dictionary that contains the data from the ROI file.
    ax
        The matplotlib axis object to be plotted.
    """
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    # Plotting the retrieved values
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[['right', 'top', 'bottom', 'left']].set_color("black")
    ax.tick_params(axis='both', labelsize=font_s)


def psycho_like_plot_and_synchro(rec, roi_info, ax):
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    to_plot = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        # 0 is plotted if there is no trial of the specified amplitude that has been detected
        if len(rec.detected_stim[rec.stim_ampl == amp]) == 0:
            to_plot.append(0)
        else:
            #TODO: verify this formula, what does it compute ?
            res = sum(rec.detected_stim[rec.stim_ampl == amp]) / len(rec.detected_stim[rec.stim_ampl == amp])
            to_plot.append(res)
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.plot([0, 2, 4, 6, 8, 10, 12], to_plot, linestyle='--')
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=font_s)


def correlation_beh_neur(rec, roi_info, n_type="EXC", detected_trials=True, undetected_trials=True):
    """
    Computes the Pearson correlation coefficient (linear relationship) between the number of responsive neurons and the
    detection level for all amplitudes.
    """
    assert detected_trials or undetected_trials, "Please select at least one trial type."
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    neura_activity = []
    resp_mat = np.array(rec.matrices[n_type]["Responsivity"])
    total_n = rec.zscore_exc.shape[0] if n_type == "EXC" else rec.zscore_inh.shape[0]
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        # Getting the neural responsivity corresponding to the selected trial type
        if detected_trials and undetected_trials:
            stim_filter = rec.stim_ampl == amp
        elif detected_trials:
            stim_filter = np.logical_and(rec.detected_stim, rec.stim_ampl == amp)
        elif undetected_trials:
            stim_filter = np.logical_and(np.invert(rec.detected_stim), rec.stim_ampl == amp)
        trials = resp_mat[:, stim_filter]
        # Counting and computing the percentage of responsive neurons for this amplitude
        recruited_det = np.mean(np.count_nonzero(trials, axis=0))
        perc_n_det = (recruited_det / total_n) * 100
        neura_activity.append(perc_n_det)
    # Testing if there is a linear relationship between the neural activity and the detection level
    coef_cor, p_value = ss.pearsonr(neura_activity, converted_list)
    return coef_cor, p_value


def zscore_by_amp(rec, neuron_zscore):
    """
    Computes the mean zscore during the trials of each amplitude for a single neuron.

    Parameters
    ----------
    rec
    neuron_zscore

    Returns
    -------
    list
        A list of the mean zscore of the neuron during the trials for each amplitude.
    """
    firing_curve = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        timings = rec.stim_time[rec.stim_ampl == amp]
        firing_curve.append(np.mean(neuron_zscore[np.linspace(timings, timings + 15, dtype=int)]))
    return firing_curve


def activation_proportion(rec, neur_id):
    """
    Computes the proportion of trials in which the neuron was activated for each amplitude.

    Parameters
    ----------
    rec
    neur_id

    Returns
    -------
    list
        A list of the proportion of trials in which the neuron was activated for each amplitude.
    """
    proportion_by_amp = []
    for amp in np.unique(rec.stim_ampl):
        amp_filter = rec.stim_ampl_filter([amp])
        exc_mat = rec.matrices["EXC"]["Responsivity"]
        exc_mat = exc_mat[neur_id, amp_filter]
        exc_mat[exc_mat != 1] = 0
        exc = np.mean(np.count_nonzero(exc_mat, axis=0))
        proportion_by_amp.append(exc/sum(rec.stim_ampl==amp))
    return proportion_by_amp


def ei_ratio_per_amp(rec):
    """
    Computes the level of E/I ratio compared to the maximum level for each amplitude.

    Parameters
    ----------
    rec

    Returns
    -------

    """
    ei_ratios = []
    for amp in np.unique(rec.stim_ampl):
        amp_filter = rec.stim_ampl_filter([amp])
        # Counting th mean number of activated EXC neurons per trial of the selected amplitude
        exc_mat = rec.matrices["EXC"]["Responsivity"]
        exc_mat = exc_mat[:, amp_filter]
        exc_mat[exc_mat != 1] = 0
        exc = np.mean(np.count_nonzero(exc_mat, axis=0))
        # Counting th mean number of activated INH neurons per trial of the selected amplitude
        inh_mat = rec.matrices["INH"]["Responsivity"]
        inh_mat = inh_mat[:, amp_filter]
        inh_mat[inh_mat != 1] = 0
        inh = np.mean(np.count_nonzero(inh_mat, axis=0))
        if inh == 0:
            ei_ratios.append(1)
        else:
            ei_ratios.append(exc/inh)
    return np.array(ei_ratios)/max(ei_ratios)


def compute_neurons_tuning(rec, sigm_on_norm=True):
    rows = []
    for type, zscore in zip(["EXC", "INH"], [rec.zscore_exc, rec.zscore_inh]):
        for neuron_id, neuron_zscore in enumerate(zscore):
            # Getting the mean activity during de trials of each amplitude
            mean_zscore_by_amp = zscore_by_amp(rec, neuron_zscore)
            # Normalization of the activity by the maximal mean activity of the neuron for an amplitude
            max_abs_value = np.max(np.abs(mean_zscore_by_amp)) * np.sign(mean_zscore_by_amp[np.argmax(np.abs(mean_zscore_by_amp))])
            normalised_mean_zscore_by_amp = np.array(mean_zscore_by_amp) / max_abs_value
            # Trying to fit a sigmoid curve on each neuron
            try:
                x, y, x0, k = sigmoid_fit(np.array(np.linspace(0, 1, 7)), normalised_mean_zscore_by_amp if sigm_on_norm else mean_zscore_by_amp)
            except:
                x, y, x0, k = np.nan, np.nan, np.nan, np.nan
                continue
            row = {"ID": neuron_id, "type": type,
                   "zscore_by_amp": mean_zscore_by_amp, "norm_zscore_by_amp": normalised_mean_zscore_by_amp,
                   "x_sigmoid": x, "y_sigmoid": y, "x0": x0, "k": k, "abs_k": abs(k)}
            rows.append(row)
    return pd.DataFrame(rows)


def cluster_neurons(neurons_df, method="kmeans"):
    if method == "manual":
        delta_x = 2/12
        tolerance = 0.025
        delta_y = 0.25
        min_k_bin = mf.minimal_k_bin(tolerance, delta_x)
        min_k_amp = mf.minimal_k_amp(delta_y)
        neurons_df.loc[neurons_df["abs_k"] < min_k_amp, "manual_cluster"] = 0
        # k=110 approximately corresponds to the value of k for which y=0 to y=1 is achieved within 2µm (1% tolerance)
        neurons_df.loc[neurons_df["abs_k"] > min_k_bin, "manual_cluster"] = 1
        neurons_df.loc[(neurons_df["abs_k"] >= min_k_amp) & (neurons_df["k"].abs() <= min_k_bin), "manual_cluster"] = 2
    elif method in ["kmeans", "k_kmeans", "x0_kmeans"]:
        features_dict = {"kmeans": ["abs_k", "x0"], "k_kmeans": ["abs_k"], "x0_kmeans": ["x0"]}
        features = neurons_df[features_dict[method]]
        kmeans = KMeans(n_clusters=3, random_state=42)
        neurons_df[f"{method}_cluster"] = kmeans.fit_predict(features)
        if method != "x0_kmeans":
            # Retrieve the original cluster labels for the min and max k values neurons
            original_cluster_min = neurons_df.loc[neurons_df["abs_k"].idxmin(), f"{method}_cluster"]
            original_cluster_max = neurons_df.loc[neurons_df["abs_k"].idxmax(), f"{method}_cluster"]
            # Identify the remaining cluster
            all_clusters = set(neurons_df[f"{method}_cluster"].unique())
            remaining_clusters = all_clusters - {original_cluster_min, original_cluster_max}
            if remaining_clusters:
                remaining_cluster = remaining_clusters.pop()
            else:
                raise ValueError("Expected 3 clusters, but got less.")
            # Map the original cluster labels to the desired labels
            mapping = {original_cluster_min: 0, original_cluster_max: 1, remaining_cluster: 2}
            # Reassign the clusters
            neurons_df[f"{method}_cluster"] = neurons_df[f"{method}_cluster"].map(mapping)
    neurons_df[f"{method}_cluster"] = neurons_df[f"{method}_cluster"].astype(int)
    return neurons_df


def get_general_cluster_parameters(recs, normalized_by_max=True, clustering_method="manual"):
    """
    For each recording in recs, get the generalized parameters of neuron clustering.

    Parameters
    ----------
    recs

    Returns
    -------

    """
    rows = []
    for rec in recs:
        # Getting the number of neurons per recording
        neurons_df = compute_neurons_tuning(rec, sigm_on_norm=normalized_by_max)
        nb_neurons_df = neurons_df["type"].value_counts()
        nb_exc = nb_neurons_df["EXC"]
        nb_inh = nb_neurons_df["INH"]
        nb_neurons = nb_exc + nb_inh
        # Getting the psychometric data
        x_psy, y_psy, x0_psy, k_psy = sigmoid_fit(np.array(np.linspace(0, 1, 7)), rec.hit_rates)
        # Clustering the neurons, pay attention to EXC/INH common or parallel clustering
        cluster_df = cluster_neurons(neurons_df, method=clustering_method)
        row = {"ID": rec.filename, "genotype": rec.genotype,
               "x": x_psy, "x0_psy": x0_psy, "k_psy": k_psy,
               "nb_exc": nb_exc, "nb_inh": nb_inh, "nb_neurons": nb_neurons}
        for cluster in [1, 2]:
            for n_type in ["EXC", "INH"]:
                # Count of neurons in the given cluster and type
                condition = (cluster_df[f"{clustering_method}_cluster"] == cluster) & (cluster_df["type"] == n_type)
                row[f"n_{cluster}_{n_type}"] = len(cluster_df[condition])
                for param in ["abs_k", "x0"]:
                    for estimator in ["mean", "std"]:
                        # Compute the statistic only if there are rows matching the condition
                        if not cluster_df[condition].empty:
                            if estimator == "mean":
                                value = cluster_df[condition][param].mean()
                            elif estimator == "std":
                                value = cluster_df[condition][param].std()
                        else:
                            value = np.nan
                        row[f"{estimator}_{param}_{cluster}_{n_type}"] = value
                # Compute the mean and std of the logarithm of k
                if not cluster_df[condition].empty:
                    log_k = np.log(cluster_df[condition]["abs_k"])
                    mean_log_k = log_k.mean()
                    std_log_k = log_k.std()
                else:
                    mean_log_k = np.nan
                    std_log_k = np.nan
                row[f"mean_log_k_{cluster}_{n_type}"] = mean_log_k
                row[f"std_log_k_{cluster}_{n_type}"] = std_log_k
        rows.append(row)
    return pd.DataFrame(rows)


def features_selection(features_df, single_split=True, genotype="both", model_name="forest"):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15), constrained_layout=True)
    model_dict = {"forest": RandomForestRegressor(n_estimators=10000, max_depth=5),
                  "lasso": MultiTaskLasso(alpha=0.1, max_iter=10000)}
    assert model_name in model_dict.keys(), f"Please provide a valid model name: {model_dict.keys()}"
    model = model_dict[model_name]
    if genotype == "WT":
        features_df = features_df[features_df["genotype"] == "WT"]
    elif genotype == "KO-Hypo":
        features_df = features_df[features_df["genotype"] == "KO-Hypo"]
    elif genotype == "KO":
        features_df = features_df[features_df["genotype"] == "KO"]
    elif genotype == "KOs":
        features_df = features_df[features_df["genotype"] != "WT"]
    features_list = [features_df.drop(columns=["ID", "genotype", "x", "x0_psy", "k_psy", "nb_exc", "nb_inh", "nb_neurons"]),
                features_df.drop(columns=["ID", "genotype", "x", "x0_psy", "k_psy", "nb_exc", "nb_inh", "nb_neurons", "x0_psy"]),
                features_df.drop(columns=["ID", "genotype", "x", "x0_psy", "k_psy", "nb_exc", "nb_inh", "nb_neurons", "k_psy"])]
    target_col_list = [["k_psy", "x0_psy"], "k_psy", "x0_psy"]
    for (row, target_col), features in zip(enumerate(target_col_list), features_list):
        targets = features_df[target_col]
        if isinstance(target_col, list) and len(target_col) > 1:
            # targets = targets.ravel()
            target_mean = features_df[target_col].mean().mean()
            target_std = features_df[target_col].std().mean()
        else:
            target_mean = features_df[target_col].mean()
            target_std = features_df[target_col].std()
        if single_split:
            title_cv = "single split"
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1)
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            # Get the feature importance
            importances = model.feature_importances_ if model_name == "forest" else np.mean(np.abs(model.coef_), axis=0)
            feat_importance = pd.Series(importances, index=features.columns).sort_values(ascending=False)
        else:
            title_cv = "Leave-One-Out CV"
            # Prepare cross-validation
            loo = LeaveOneOut()
            fold_importances = []
            fold_errors = []
            for train_index, test_index in loo.split(features):
                X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]
                # Fit the model
                model.fit(X_train, y_train)
                # Predict on test fold
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                fold_errors.append(rmse)
                # Store feature importances for this fold
                fold_importances.append(model.feature_importances_ if model_name == "forest" else np.mean(np.abs(model.coef_), axis=0))
            # Average the feature importances across folds
            avg_importances = np.mean(fold_importances, axis=0)
            feat_importance = pd.Series(avg_importances, index=features.columns).sort_values(ascending=False)
            rmse = np.mean(fold_errors)
        # Plotting the feature importances
        feat_importance.plot(kind='bar', ax=ax[row, 0])
        ax[row, 0].set_ylabel(f"Importance for {target_col} - RMSE={rmse:.3f}\nRMSE/Mean={rmse/target_mean:.3f} - RMSE/Std={rmse/target_std:.3f}", fontsize=10)
        ax[row, 0].set_xticklabels(ax[row, 0].get_xticklabels(), fontsize=8, rotation=45, ha="right")
        ax[row, 0].set_yticklabels(ax[row, 0].get_yticklabels(), fontsize=8)
        for p in ax[row, 0].patches:
            height = p.get_height()
            ax[row, 0].annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        # Plotting the cumulative features importance
        cumulative_importance = feat_importance.cumsum()
        cumulative_importance.plot(ax=ax[row, 1], marker="+", ms=10, lw=2)
        ax[row, 1].set_ylabel("Cumulative Importance", fontsize=10)
        ax[row, 1].set_xticks(range(len(cumulative_importance.index)))
        ax[row, 1].set_xticklabels(cumulative_importance.index, fontsize=8, rotation=45, ha="right")
        ax[row, 1].set_yticklabels(ax[row, 1].get_yticklabels(), fontsize=8)
        for x, y in enumerate(cumulative_importance):
            ax[row, 1].annotate(f'{y:.3f}', (x, y), ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    fig.suptitle(f"Feature Importances for {genotype}\n{title_cv}[{model_name}]", fontsize=14)
    fig.canvas.manager.set_window_title(f"Features importance [{genotype}]_{title_cv}[{model_name}]")
    plt.show()

def features_behavior_corr(features_df, genotype="all"):
    if genotype in ["WT", "KO", "KO-Hypo"]:
        features_df = features_df[features_df["genotype"] == genotype]
    colors = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color, "all": "blueviolet"}
    signif_dfs = []
    for target in ["k_psy", "x0_psy"]:
        fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(25, 15), constrained_layout=True)
        ax = axes.flatten()
        # Getting the column names containing the features to correlate with the targets
        exclude_columns = ["ID", "genotype", "x", "x0_psy", "k_psy", "nb_exc", "nb_inh", "nb_neurons"]
        features_columns_list = [col for col in features_df.columns if col not in exclude_columns]
        # For each feature, compute and plot the correlation with the target
        signif_rows = []
        for ax_id, feature_col in enumerate(features_columns_list):
            feat_na_df = features_df.dropna(subset=[feature_col])
            # Compute linear regression for the best fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(feat_na_df[feature_col], feat_na_df[target])
            r2 = r_value ** 2
            line = slope * feat_na_df[feature_col] + intercept
            # Plot the data points and regression line
            ax[ax_id].plot(feat_na_df[feature_col], line, color=colors[genotype], lw=2)
            for g in feat_na_df["genotype"].unique():
                group = feat_na_df[feat_na_df["genotype"] == g]
                sc = ax[ax_id].scatter(group[feature_col], group[target], color=colors[g], alpha=0.7, label=g, s=10, marker="+")
                # Save the IDs for this group so that they can be accessed in the callback.
                ids = group["ID"].values
                mplcursors.cursor(sc, hover=True).connect("add", lambda sel, ids=ids: (sel.annotation.set_text(f"ID: {ids[sel.index]}"), sel.annotation.set_fontsize(8)))
            # ax[ax_id].scatter(features_df[feature_col], features_df[target], label=features_df["ID"], color=colors[features_df["genotype"]])
            # Label the axes and add a title
            ax[ax_id].set_xlabel(feature_col, fontsize=10)
            ax[ax_id].set_ylabel(target, fontsize=10)
            # Annotate the plot with R² and p-value
            ax[ax_id].text(0.05, 0.95, f"$r^2 = {r2:.3f}$\np-value = {p_value:.3f}", transform=ax[ax_id].transAxes, fontsize=10, verticalalignment='top')
            ax[ax_id].set_xticklabels(ax[ax_id].get_xticklabels(), fontsize=8)
            ax[ax_id].set_yticklabels(ax[ax_id].get_yticklabels(), fontsize=8)
            if p_value < 0.1 and r2 > 0.1:
                signif_rows.append({"feature": feature_col, "r2": r2, "pval": p_value})
        signif_dfs.append(pd.DataFrame(signif_rows).sort_values(by="r2", ascending=False) if len(signif_rows) > 0 else None)
        fig.suptitle(f"Linear Regression of the different neuronal features with {target} for {genotype}", fontsize=14)
        fig.canvas.manager.set_window_title(f"Lin_reg_{target}_{genotype}")
        plt.show()
    k_psy_df, x0_psy_df = signif_dfs
    return k_psy_df, x0_psy_df


def plot_neuron_clustering(recs, method, type="EXC"):
    assert type in ["EXC", "INH", "EXC/INH"], "Please provide a valid neuron type (EXC, INH)"
    cluster_colors_EXC = {0: "pink", 1: "blueviolet", 2: "magenta"}
    cluster_colors_INH = {0: "#c0ffc3", 1: "#004200", 2: "lime"}
    genotype_colors = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color}
    fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, sharey=True, figsize=(24, 20), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, rec in zip(axes_flat, recs):
        # Plotting the psychometric curve
        x_psy, y_psy, x0_psy, k_psy = sigmoid_fit(np.array(np.linspace(0, 1, 7)), rec.hit_rates)
        ax.plot(x_psy, y_psy, color="black", lw=2)
        # If the detection threshold lies between 0 and 12µm, it is plotted on the graph and x0 is displayed
        if x0_psy < 1:
            ax.text(x0_psy, -0.125, f"{x0_psy * 11:.2f}µm", color="black", ha='center', fontsize=10)
            # Plot a vertical dashed line at x0_psy, from y=0 to f(x0_psy)
            y_at_x0 = np.interp(x0_psy, x_psy, y_psy)
            ax.vlines(x=x0_psy, ymin=0, ymax=y_at_x0, colors="black", linestyles='dashed', lw=1)
        # Plotting the individual neurons curves
        all_neurons = compute_neurons_tuning(rec)
        nb_neurons_df = all_neurons["type"].value_counts()
        nb_exc = nb_neurons_df["EXC"]
        nb_inh = nb_neurons_df["INH"]
        nb_neurons = nb_exc + nb_inh
        if type == "EXC/INH":
            # Trying performing the same grouped clusterization on both neuron types
            neurons = all_neurons.copy()
            cluster_df = cluster_neurons(neurons, method=method)
            for idx, neuron in cluster_df.iterrows():
                n_type = neuron["type"]
                ax.plot(neuron["x_sigmoid"], neuron["y_sigmoid"], color=eval(f"cluster_colors_{n_type}[neuron['{method}_cluster']]"), lw=1, alpha=0.75)
            ax.set_title(f"{rec.filename} ({rec.genotype}) - {rec.threshold}", color=genotype_colors[rec.genotype], fontsize=12, fontweight="bold")
            # Plotting the generalization of the neuronal activity
            mean_x0_exc = cluster_df[(cluster_df[f"{method}_cluster"] == 1) & (cluster_df["type"] == "EXC")]["x0"].mean()
            mean_x0_inh = cluster_df[(cluster_df[f"{method}_cluster"] == 1) & (cluster_df["type"] == "INH")]["x0"].mean()
            mean_k_exc = cluster_df[(cluster_df[f"{method}_cluster"].isin([1, 2])) & (cluster_df["type"] == "EXC")]["abs_k"].mean()
            mean_k_inh = cluster_df[(cluster_df[f"{method}_cluster"].isin([1, 2])) & (cluster_df["type"] == "INH")]["abs_k"].mean()
            mean_x0 = mean_x0_exc * (nb_exc/nb_neurons) - mean_x0_inh * (nb_inh/nb_neurons)
            mean_k = mean_k_exc * (nb_exc/nb_neurons) - mean_k_inh * (nb_inh/nb_neurons)
        else:
            neurons = all_neurons.loc[all_neurons["type"] == type].copy()
            cluster_df = cluster_neurons(neurons, method=method)
            for idx, neuron in cluster_df.iterrows():
                ax.plot(neuron["x_sigmoid"], neuron["y_sigmoid"], color=eval(f"cluster_colors_{type}[neuron['{method}_cluster']]"), lw=1, alpha=0.75)
            ax.set_title(f"{rec.filename} ({rec.genotype}) - {rec.threshold:.1f}", color=genotype_colors[rec.genotype], fontsize=12, fontweight="bold")
            # Plotting the generalization of the neuronal activity
            mean_x0 = cluster_df[cluster_df[f"{method}_cluster"] == 1]["x0"].mean()
            mean_k = cluster_df[cluster_df[f"{method}_cluster"].isin([1, 2])]["abs_k"].mean()
        mean_y = 1 / (1 + np.exp(-mean_k * (x_psy - mean_x0)))
        ax.plot(x_psy, mean_y, lw=2, color="red")
    fig.suptitle(f"{type} neurons clustering - method: {method}", fontsize=14)
    fig.canvas.manager.set_window_title(f"{type} Neurons clustering[{method}]")
    plt.show()


def individuals_neurons_tuning(rec, roi_info, normalize=False):
    """Plot the psychometric curves. Compute the activity curve for single neurons for zscore and activation rate (proportion of trials per
    amplitude for which the neurons is considered active"""
    colors = {"WT": ppt.wt_color, "KO": ppt.ko_color, "KO-Hypo": ppt.hypo_color}
    color_behavior = colors[rec.genotype]
    # === Plot 1 ===
    # Retrieving the detection for each amplitude from the ROI file
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = [float(x) for x in seq[0].split(',')]
    fig, ax = plt.subplots(4, 1, figsize=(8, 20), sharex=True)
    # Fitting a sigmoid curve on the data (psychometric curve)
    x_psy, y_psy, x0_psy, k_psy = sigmoid_fit(np.array(np.linspace(0, 1, 7)), converted_list)
    ax[0].plot(x_psy, y_psy, color=color_behavior)  # the fitted curve
    ax[0].plot(np.array(np.linspace(0, 1, 7)), converted_list, ".", color=color_behavior)  # the data points
    # If the detection threshold lies between 0 and 12µm, it is plotted on the graph and x0 is displayed
    if x0_psy < 1:
        ax[0].vlines(x=x0_psy, ymin=0, ymax=1, color='red', linestyle='dashed', lw=2)
        ax[0].text(x0_psy, -0.25, f"x0={x0_psy:.2f}", color='red', ha='center', fontsize=10)
    # Plotting the level of E/I ratio
    ei_ratio = ei_ratio_per_amp(rec)
    if ei_ratio.all() != np.nan:
        ax[0].plot(np.array(np.linspace(0, 1, 7)), ei_ratio)

    # === Plot 2 ===
    # Defining the annotation to display the k value when hovering the curve
    annot = ax[1].annotate(
        text='',
        fontsize=10,
        xy=(0, 0),
        xytext=(15, 15),
        textcoords='offset points',
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    # Initializing variables
    dtype = [("ID", "int"), ("x0", "float"), ("k", "float"), ("cluster", "int")]
    neurons_array = np.empty(0, dtype=dtype)
    cluster_dict = {0: ["No response", "gray"], 1: ["Binary activation", "purple"], 2: ["Amplitude related", "pink"]}
    zscores = []
    lines_and_ks = []
    skipped_neurons = 0
    act_n, desact_n = pu.idx_resp_neur(rec)
    # For each neuron:
    for idx, zscore in enumerate(rec.zscore_exc):
        # Computing the mean zscore for each amplitude
        zsc = zscore_by_amp(rec, zscore)
        # Normalization
        if normalize:
            # zsc_normalised_theo = (np.array(zsc) - zsc[0]) / zsc[-1]
            max_abs_value = np.max(np.abs(zsc)) * np.sign(zsc[np.argmax(np.abs(zsc))])
            zsc_normalised = np.array(zsc) / max_abs_value
        else:
            zsc_normalised = np.array(zsc)
        # Handling the case where the neuron is less active as the amplitude increases
        # zsc_normalised = zsc_normalised * -1 if decrease_activity(zsc_normalised) else zsc_normalised
        # Discard neurons activity that are not fitting sigmoid fit
        try:
            x, y, x0, k = sigmoid_fit(np.array(np.linspace(0, 1, 7)), zsc_normalised)
        except:
            skipped_neurons += 1
            continue
        # Clustering of neurons based on the steepness of the sigmoid (k)
        if abs(k) < 1:
            cluster = 0
        elif abs(k) > 10:
            cluster = 1
        else:
            cluster = 2
        # Adding the new neuron's data to the array
        new_row = np.array([(idx, x0, k, cluster)], dtype=dtype)
        neurons_array = np.append(neurons_array, new_row)
        # Storing the lines to be able to display the k values when hovering them
        line, = ax[1].plot(x, y, color=cluster_dict[cluster][1], lw=2, alpha=0.75)
        lines_and_ks.append((line, k))
        # === Plot 3 ===
        # Computing the proportion of trials in which the neuron was activated for each amplitude
        prop_act = activation_proportion(rec, idx)
        ax[2].plot(np.linspace(0, 1, 7), prop_act)

        # === Plot 4 ===
        # Plotting the raw mean zscore of the neuron for each amplitude
        ax[3].plot(np.linspace(0, 1, 7), zsc)
        zscores.append(zsc)

    def update_annotation(event):
        if event.inaxes == ax[1]:
            for line, k_value in lines_and_ks:
                cont, ind = line.contains(event)
                if cont:
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(f"k={k_value:.2f}")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            annot.set_visible(False)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", update_annotation)

    # Computing and plotting the average for all neurons of the mean zscore of each neuron for each amplitude
    ax[3].plot(np.linspace(0, 1, 7), np.average(zscores, axis=0), lw=5, linestyle="dashed", color="black")
    # Computing the mean correaltion coeficient to have an idea of how similarly the neurons respond to the different amplitudes of stimulation
    cor_coef = np.mean(np.corrcoef(zscores))
    ax[0].set_title(f"{rec.filename}-{rec.genotype} {round(cor_coef, 2)} {skipped_neurons}", fontsize=40)
    ax[0].set_ylim([0, 1])
    ax[2].set_ylim([0, 1])
    ax[0].set_xlim([0.16, 1])
    ax[1].set_title(f"Sigmoid fit with z-score for single neur norm={normalize}")
    ax[2].set_title("Activition rate by amp for single neur")
    ax[3].set_title("Z-score by amp for single neur")
    ax[3].set_xticks(np.linspace(0, 1, 7))
    ax[3].set_xticklabels([0, 2, 4, 6, 8, 10, 12])
    fig.tight_layout()
    plt.show()
    print(np.min(neurons_array["k"]), np.max(neurons_array["k"]))
    return neurons_array


def decrease_activity(neur_act):
    if neur_act[-1] < neur_act[0]:
        return True
    else:
        return False


def sigmoid_fit(xdata, ydata):
    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

    def sigmoid_Hill(x, n, k):
        y = (x ** n) / (x ** n + k ** n)
        # y = 1/(1+(k/x)**n)
        return y

    popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=100000)
    fix_value = xdata[-1]  # + 1
    # slope, intercept = np.polyfit(x, y, 1)
    # slope = float("{:.2f}".format(slope))
    # Get r2 score
    xdata = xdata.astype(float)
    # y_pred = sigmoid(xdata, *popt)
    # r2 = r2_score(ydata, y_pred)
    # liste_r2.append(r2)

    x = np.linspace(0, fix_value, 50)
    y = sigmoid(x, *popt)
    return x, y, popt[0], popt[1]

def group_tuning_comp(recs, normalize=False):
    dtype = [("ID", "str"), ("Genotype", "str"),
             ("0_nb", "int"), ("0_mean_x0", "float"), ("0_std_x0", "float"), ("0_mean_k", "float"), ("0_std_k", "float"),
             ("1_nb", "int"), ("1_mean_x0", "float"), ("1_std_x0", "float"), ("1_mean_k", "float"), ("1_std_k", "float"),
             ("2_nb", "int"), ("2_mean_x0", "float"), ("2_std_x0", "float"), ("2_mean_k", "float"), ("2_std_k", "float")]
    array = np.empty(0, dtype=dtype)




if __name__ == '__main__':
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

    # test_neurons = compute_neurons_tuning(recs[5886])
    # exc_neurons = test_neurons.loc[test_neurons["type"] == "EXC"].copy()
    # test_cluster_5886 = cluster_neurons(exc_neurons, method="manual")
    plot_neuron_clustering(recs.values(), method="manual", type="EXC/INH")
    features_df = get_general_cluster_parameters(recs.values(), normalized_by_max=True, clustering_method="manual")
    # features_selection(features_df, single_split=False, genotype="all", model_name="forest")
    k_psy_df, x0_psy_df = features_behavior_corr(features_df, genotype="all")
    k_psy_df_wt, x0_psy_df_wt = features_behavior_corr(features_df, genotype="WT")
    k_psy_df_ko, x0_psy_df_ko = features_behavior_corr(features_df, genotype="KO-Hypo")


    # neural tuning curves for individuals neurons
    # wt, ko = [], []
    # for rec in recs.values():
    #     avg_k, std_k = individuals_neurons_tuning(rec, roi_info, normalize=True)
    #     if rec.genotype == "WT":
    #         wt.append([avg_k, std_k, rec.threshold])
    #     elif rec.genotype == "KO-Hypo":
    #         ko.append([avg_k, std_k, rec.threshold])
    #
    # fig, axs = plt.subplots(ncols=2)
    # ppt.boxplot(axs[0], np.array(wt)[:, 0], np.array(ko)[:, 0], "average k")
    # print(f"Avg half-activation point: \n WT {np.mean(np.array(wt)[:, 0])}, KO {np.median(np.array(ko)[:, 0])}")
    # print(f"Detection threshold: \n WT {np.mean(np.array(wt)[:, 2])}, KO {np.mean(np.array(ko)[:, 2])}")
    # ppt.boxplot(axs[1], np.array(wt)[:, 1], np.array(ko)[:, 1], "std k")
    # fig.tight_layout()
    # fig.show()
    #
    # # Correlation of the detection threshold and the average half activation of the neurons and
    # # the std half activation of the neurons
    #
    # fig, axs = plt.subplots(nrows=2)
    # axs[0].plot(np.array(wt)[:, 0], np.array(wt)[:, 2], ".", color=ppt.wt_color)
    # axs[0].plot(np.array(ko)[:, 0], np.array(ko)[:, 2], ".", color=ppt.hypo_color)
    # axs[1].plot(np.array(wt)[:, 1], np.array(wt)[:, 2], ".", color=ppt.wt_color)
    # axs[1].plot(np.array(ko)[:, 1], np.array(ko)[:, 2], ".", color=ppt.hypo_color)
    # fig.tight_layout()
    # fig.show()
