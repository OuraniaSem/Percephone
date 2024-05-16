import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
from percephone.core.recording import RecordingAmplDet
import percephone.plts.heatmap as hm


def neuron_mean_std_corr(array, estimator):
    """
    Parameters
    ----------
    array : np.ndarray
        The input array or matrix of shape (nb frames * nb neurons)

    estimator : str
        The type of estimator to use. Available options are "Mean" or "Std". If None is provided, the original array will be returned.

    Returns
    -------
    np.ndarray
        The result of the estimation based on the given estimator. If estimator is "Mean", returns the mean along the
        specified axis. If estimator is "Std", returns the standard deviation along the specified axis. If estimator is
        None, returns the original array.

    """
    if estimator is None:
        return array
    if estimator == "Mean":
        return np.mean(array, axis=0)
    if estimator == "Std":
        return np.std(array, axis=0)


def get_iter_range(rec, time_span):
    if time_span == "stim" or time_span == "pre_stim":
        iter_range = rec.stim_time.shape[0]
    elif time_span == "reward":
        iter_range = rec.reward_time.shape[0]
    elif time_span == "timeout":
        iter_range = rec.timeout_time.shape[0]
    return iter_range


def get_timepoints(rec, i, time_span, window=0.5):
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
    return start, end


def build_row(zscore, start, end, estimator):
    if estimator is None:
        row = zscore[start:end]
    else:
        row = neuron_mean_std_corr(zscore[start: end], estimator)
    return row


def get_zscore(rec, exc_neurons=True, inh_neurons=False, time_span="stim", window=0.5, estimator=None, sort=False, amp_sort=False):
    if amp_sort:
        assert sort
    if sort or amp_sort:
        assert time_span == "stim" or time_span == "pre_stim"
    # Retrieving zscore
    if exc_neurons and inh_neurons:
        zscore = np.row_stack((rec.zscore_exc, rec.zscore_inh)).T
    elif exc_neurons:
        zscore = rec.zscore_exc.T
    elif inh_neurons:
        zscore = rec.zscore_inh.T

    # Getting the iter range
    iter_range = get_iter_range(rec, time_span)

    # Initializing the list of stim duration and amplitude
    t_stim = list(rec.stim_durations)

    # Initializing X
    if sort:
        if amp_sort:
            all_amp = sorted([int(i) for i in set(rec.stim_ampl)])
            X_amp_det = {i: np.empty((0, zscore.shape[1])) for i in all_amp}
            X_amp_undet = {i: np.empty((0, zscore.shape[1])) for i in all_amp}
            t_amp_det = {i: [] for i in all_amp}
            t_amp_undet = {i: [] for i in all_amp}

        X_det = np.empty((0, zscore.shape[1]))
        X_undet = np.empty((0, zscore.shape[1]))
        t_det = []
        t_undet = []
    else:
        X = np.empty((0, zscore.shape[1]))

    # Building zscore matrix
    for i in range(iter_range):
        start, end = get_timepoints(rec, i, time_span, window)
        new_row = build_row(zscore, start, end, estimator)
        # Stacking new row
        if sort:
            if amp_sort:
                amp = rec.stim_ampl[i]
                if rec.detected_stim[i]:
                    X_amp_det[amp] = np.row_stack((X_amp_det[amp], new_row))
                    t_amp_det[amp].append(rec.stim_durations[i])
                else:
                    X_amp_undet[amp] = np.row_stack((X_amp_undet[amp], new_row))
                    t_amp_undet[amp].append(rec.stim_durations[i])
            else:
                if rec.detected_stim[i]:
                    X_det = np.row_stack((X_det, new_row))
                    t_det.append(rec.stim_durations[i])
                else:
                    X_undet = np.row_stack((X_undet, new_row))
                    t_undet.append(rec.stim_durations[i])
        else:
            X = np.row_stack((X, new_row))

    if sort:
        if amp_sort:
            for a in all_amp:
                X_det = np.row_stack((X_det, X_amp_det[a]))
                X_undet = np.row_stack((X_undet, X_amp_undet[a]))
                t_det.extend(t_amp_det[a])
                t_undet.extend(t_amp_undet[a])

        X = np.row_stack((X_det, X_undet))
        t_stim = t_det + t_undet

    if estimator is not None:
        if time_span == "stim":
            X = np.repeat(X, t_stim, axis=0)
        else:
            X = np.repeat(X, int(window * rec.sf), axis=0)
    return X.T, t_stim


def plot_heatmap(rec, data, type="stim", stim_dur=None, window=0.5, sorted=False, amp_sorted=False, estimator=None):
    if stim_dur is None:
        stim_dur = rec.stim_durations

    # figure global parameters
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax) if (type == "stim" or type == "pre_stim") else None
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cmap = "inferno"
    extent = [0, data.shape[1], data.shape[0] - 0.5, -0.5]

    # plotting the stimulation amplitudes
    if sorted:
        stim_det = []
        stim_undet = []
        for i in range(rec.stim_time.shape[0]):
            if type == "stim":
                ampl_vec = [rec.stim_ampl[i]] * int(rec.stim_durations[i])
            elif type == "pre_stim":
                ampl_vec = [rec.stim_ampl[i]] * int(window * rec.sf)
            (stim_det if rec.detected_stim[i] else stim_undet).extend(ampl_vec)
        if amp_sorted:
            stim_det.sort()
            stim_undet.sort()
        stim_array = np.array(stim_det + stim_undet)
    else:
        stim_bar = []
        for i in range(rec.stim_time.shape[0]):
            if type == "stim":
                stim_bar.extend([rec.stim_ampl[i]] * int(rec.stim_durations[i]))
            elif type == "pre_stim":
                stim_bar.extend([rec.stim_ampl[i]] * int(window * rec.sf))
        stim_array = np.array(stim_bar)

    if (type == "stim" or type == "pre_stim"):
        tax1.imshow(stim_array.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
        tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # neurons clustering and data display
    Z = linkage(data, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(data[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(data), 1),
                   vmax=np.nanpercentile(np.ravel(data), 99), extent=extent)

    # plotting lines to separate stimulation
    if type == "stim":
        cumulative_stim_duration = 0
        for stim in stim_dur:
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
    else:
        iterator = len(rec.reward_time) if type == "reward" else (len(rec.timeout_time) if type == "timeout" else 0)
        for i in range(iterator):
            ax.vlines(i * int(window * rec.sf), ymin=-0.5, ymax=len(data)-0.5, color='w', linewidth=0.5)

    # color scale parameters
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label("Z-score") if estimator is None else cbar.set_label(f"Z-score ({estimator})")

    ax.set_ylabel('Neurons')
    ax.set_xlabel(f"Frames ({type})")
    tax1.set_title(f"{rec.filename} ({rec.genotype}) - {rec.threshold}") if (type == "stim" or type == "pre_stim") else ax.set_title(f"{rec.filename} ({rec.genotype}) - {rec.threshold}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Record import
    plt.ion()
    roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    folder = "C:/Users/cvandromme/Desktop/Data/20220930_4745_01_synchro/"
    rec = RecordingAmplDet(folder, 0, roi_path, cache=True)
    rec.peak_delay_amp()
    rec.auc()

    det_sorting = True
    amp_sorting = True
    period = "stim"
    window = 0.5
    estimator = "Mean"

    data, stim_time = get_zscore(rec, exc_neurons=True, inh_neurons=False,
                                 time_span=period, window=window, estimator=estimator,
                                 sort=det_sorting, amp_sort=amp_sorting)

    plot_heatmap(rec, data, type=period, stim_dur=stim_time, window=window,
                 sorted=det_sorting, amp_sorted=amp_sorting, estimator=estimator)



    # hm.plot_dff_stim_detected(rec, rec.df_f_exc)
    # hm.plot_dff_stim_detected(rec, rec.df_f_inh)
    # hm.intereactive_heatmap(rec, rec.zscore_exc)
    # hm.intereactive_heatmap(rec, rec.zscore_inh)

