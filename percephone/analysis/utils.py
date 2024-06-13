"""
Théo Gauvrit 22/01/2024
Utility functions for analysis
"""


import matplotlib
import numpy as np


matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Qt5Agg")


def kernel_biexp(sf):
    """
    Generate kernel of a biexponential function for mlr analysis or onset delay analysis
    Parameters
    ----------
    sf: float
        sampling frequency of the recording

    Returns
    -------
    kernel_bi: array
        kernel of the biexponential function

    """
    tau_r = 0.07  # s0.014
    tau_d = 0.236  # s
    kernel_size = 10  # size of the kernel in units of tau
    a = 5  # scaling factor for biexpo kernel
    dt = 1 / sf  # spacing between successive timepoints
    n_points = int(kernel_size * tau_d / dt)
    kernel_times = np.linspace(-n_points * dt,
                               n_points * dt,
                               2 * n_points + 1)  # linearly spaced array from -n_pts*dt to n_pts*dt with spacing dt
    kernel_bi = a * (1 - np.exp(-kernel_times / tau_r)) * np.exp(-kernel_times / tau_d)
    kernel_bi[kernel_times < 0] = 0  # set to zero for negative times
    # fig, ax = plt.subplots()
    # ax.plot(kernel_times, kernel_rise)
    # ax.set_xlabel('time (s)')
    # plt.show()
    return kernel_bi


def get_iter_range(rec, time_span):
    """
    Returns the iterator range corresponding to the provided time span.

    Parameters
    ----------
    rec : object of the Recording class
        The recording for which we want to get the iter range
    time_span : str
        The time period for which we want to get the iter range

    Returns
    -------
    The iter range corresponding to the provided time span.
    """
    if time_span == "stim" or time_span == "pre_stim" or time_span == "spaced_pre_stim":
        iter_range = rec.stim_time.shape[0]
    elif time_span == "reward":
        iter_range = rec.reward_time.shape[0]
    elif time_span == "timeout":
        iter_range = rec.timeout_time.shape[0]
    return iter_range


def get_timepoints(rec, i, time_span, window=0.5):
    """
    Returns the time points corresponding to the start and the end of i-th provided time span.

    Parameters
    ----------
    rec : object of the Recording class
        The recording for which we want to get the timepoints
    time_span : str
        The time period for which we want to get the timepoints
    i : int
        The index of the time span for which we want to get the timepoints
    window : float (optional, default 0.5)
        The window size of the time span in seconds. For stimulation, the stimulation duration is taken.
    """
    if time_span == "stim":
        start = rec.stim_time[i]
        end = rec.stim_time[i] + int(rec.stim_durations[i])
    elif time_span == "pre_stim":
        start = rec.stim_time[i] - int(window * rec.sf)
        end = rec.stim_time[i]
    elif time_span == "spaced_pre_stim":
        start = rec.stim_time[i] - int(window * rec.sf) - int(0.25 * rec.sf)
        end = rec.stim_time[i] - int(0.25 * rec.sf)
    elif time_span == "reward":
        start = rec.reward_time[i]
        end = rec.reward_time[i] + int(window * rec.sf)
    elif time_span == "timeout":
        start = rec.timeout_time[i]
        end = rec.timeout_time[i] + int(window * rec.sf)
    return start, end


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
    elif estimator == "Mean":
        return np.mean(array, axis=0)
    elif estimator == "Std":
        return np.std(array, axis=0)
    elif estimator == "Max":
        return np.max(array, axis=0)
    elif estimator == "Min":
        return np.min(array, axis=0)


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


def idx_resp_neur(rec, n_type="EXC"):
    if n_type=="EXC":
        signals= rec.zscore_exc
        resp = rec.matrices['EXC']["Responsivity"][:,rec.detected_stim]
    elif n_type=="INH":
        signals= rec.zscore_inh
        resp = rec.matrices['INH']["Responsivity"][:,rec.detected_stim]
    indices_resp = np.argwhere(np.count_nonzero(resp == 1, axis=1) >5)
    indices_inhibited = np.argwhere(np.count_nonzero(resp ==-1, axis=1)>5)
    return np.ravel(indices_resp), np.ravel(indices_inhibited)