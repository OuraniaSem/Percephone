import numpy as np
import matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
from percephone.core.recording import RecordingAmplDet
import percephone.plts.heatmap as hm


def neuron_mean_std_corr(array, estimator):
    if estimator == "Mean":
        return np.mean(array, axis=0)
    if estimator == "Std":
        return np.std(array, axis=0)


def get_zscore(rec, exc_neurons=True, inh_neurons=False, time_span="stim", window=0.5, estimator=None, sort=False):
    # Retrieving zscore
    if exc_neurons and inh_neurons:
        zscore = np.row_stack((rec.zscore_exc, rec.zscore_inh)).T
    elif exc_neurons:
        zscore = rec.zscore_exc.T
    elif inh_neurons:
        zscore = rec.zscore_inh.T

    # Defining the iterator range (number of the considered event)
    if time_span == "stim" or time_span == "pre_stim":
        iter_range = rec.stim_time.shape[0]
    elif time_span == "reward":
        iter_range = rec.reward_time.shape[0]
    elif time_span == "timeout":
        iter_range = rec.timeout_time.shape[0]

    # Building zscore matrix
    first = True
    for i in range(iter_range):
        # Defining start and end timepoints
        if time_span == "stim":
            start = rec.stim_time[i]
            end = rec.stim_time[i] + int(rec.stim_durations[i])
        elif time_span == "pre_stim":
            start = rec.stim_time[i] - int(window * rec.sf)
            end = rec.stim_time[i]
        elif time_span == "reward":
            start = rec.reward_time[i]
            end = rec.reward_time[i] + int(window * rec.sf)
        elif time_span == "timeout":
            start = rec.timeout_time[i]
            end = rec.timeout_time[i] + int(window * rec.sf)

        # Building the row
        if first:
            if (time_span == "stim" or time_span == "pre_stim") and sort:
                if rec.detected_stim[i]:
                    if estimator is None:
                        X_det = zscore[start: end] if rec.detected_stim[i] else np.empty((0, zscore.shape[1]))
                        X_undet = zscore[start: end] if not rec.detected_stim[i] else np.empty((0, zscore.shape[1]))
                    else:
                        X_det = neuron_mean_std_corr(zscore[start: end], estimator) if rec.detected_stim[i] else np.empty((0,))
                        X_undet = neuron_mean_std_corr(zscore[start: end], estimator) if not rec.detected_stim[i] else np.empty((0,))
                else:
                    if estimator is None:
                        X_det = zscore[start: end] if rec.detected_stim[i] else np.empty((0, zscore.shape[1]))
                        X_undet = zscore[start: end] if not rec.detected_stim[i] else np.empty((0, zscore.shape[1]))
                    else:
                        X_det = neuron_mean_std_corr(zscore[start: end], estimator) if rec.detected_stim[i] else np.empty((0,))
                        X_undet = neuron_mean_std_corr(zscore[start: end], estimator) if not rec.detected_stim[i] else np.empty((0,))
            else:
                if estimator is None:
                    X = zscore[start: end]
                else:
                    X = neuron_mean_std_corr(zscore[start: end], estimator)
            first = False
        else:
            # Building new row
            if estimator is None:
                new_row = zscore[start: end]
            else:
                new_row = neuron_mean_std_corr(zscore[start: end], estimator)
            # Stacking new row
            if (time_span == "stim" or time_span == "pre_stim") and sort and rec.detected_stim[i]:
                X_det = np.row_stack((X_det, new_row))
            elif (time_span == "stim" or time_span == "pre_stim") and sort and not rec.detected_stim[i]:
                X_undet = np.row_stack((X_undet, new_row))
            else:
                X = np.row_stack((X, new_row))
    if (time_span == "stim" or time_span == "pre_stim") and sort:
        X = np.row_stack((X_det, X_undet))
    return X.T

def plot_heatmap(rec, data, type="stim", window=0.5, sorted=False):
    dt = 1 / rec.sf
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    # time_range = np.linspace(0, (len(data[0]) / rec.sf) - 1, len(data[0]))
    Z = linkage(data, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    # extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(data) - 0.5, -0.5]
    ax.imshow(data[dn_exc['leaves']], cmap="inferno", interpolation='none', aspect='auto',
              vmin=np.nanpercentile(np.ravel(data), 1),
              vmax=np.nanpercentile(np.ravel(data), 99))

    if type == "stim":
        cumulative_stim_duration = 0
        for stim in rec.stim_durations:
            cumulative_stim_duration += stim
            ax.vlines(cumulative_stim_duration, ymin=-0.5, ymax=len(data)-0.5, color='w', linewidth=0.5)

        if sorted:
            det_stim_duration = rec.stim_durations[rec.detected_stim]
            ax.vlines(det_stim_duration.sum(), ymin=-0.5, ymax=len(data)-0.5, color='b', linewidth=1)

    elif type == "pre_stim":
        for i in range(len(rec.detected_stim)):
            ax.vlines(i * int(window * rec.sf), ymin=-0.5, ymax=len(data)-0.5, color='w', linewidth=0.5)

        if sorted:
            ax.vlines(rec.detected_stim.sum() * int(window * rec.sf), ymin=-0.5, ymax=len(data)-0.5, color='b', linewidth=1)

    plt.show()


if __name__ == '__main__':
    # Record import
    plt.ion()
    roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    folder = "C:/Users/cvandromme/Desktop/Data/20220715_4456_00_synchro/"
    rec = RecordingAmplDet(folder, 0, roi_path, cache=True)
    rec.peak_delay_amp()
    rec.auc()

    data = get_zscore(rec, exc_neurons=True, inh_neurons=False, time_span="stim", window=0.5, estimator=None, sort=True)
    plot_heatmap(rec, data, type="stim", window=0.5, sorted=True)
    print(rec.stim_durations.sum())



    # hm.plot_dff_stim_detected(rec, rec.df_f_exc)
    # hm.plot_dff_stim_detected(rec, rec.df_f_inh)
    # hm.intereactive_heatmap(rec, rec.zscore_exc)
    # hm.intereactive_heatmap(rec, rec.zscore_inh)

