"""
Théo Gauvrit 18/01/2024
Utility function for plots
"""
import numpy as np

from percephone.analysis.utils import get_iter_range, neuron_mean_std_corr, get_timepoints


def get_zscore(rec, exc_neurons=True, inh_neurons=False, time_span="stim", window=0.5, estimator=None, sort=False,
               amp_sort=False):
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

    # Defining a function to compute the new row to append
    def build_row(zscore, start, end, estimator):
        if estimator is None:
            row = zscore[start:end]
        else:
            row = neuron_mean_std_corr(zscore[start: end], estimator)
        return row

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