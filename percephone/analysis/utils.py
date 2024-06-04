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
    if estimator == "Mean":
        return np.mean(array, axis=0)
    if estimator == "Std":
        return np.std(array, axis=0)

