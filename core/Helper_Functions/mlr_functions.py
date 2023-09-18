"""Théo Gauvrit, 15/09/2023
Functions for multiple linear regression linear"""

import numpy as np
from scipy.interpolate import interp1d


def bootstrap(signal):
    """resample a time series with replacement.

    Parameters :
    ------------
    signal : array
        original time series, the first dimension corresponds to time, shape (n_timepoints,...)

    Return :
    --------
    resampled_signal : array
        bootstrapped time series
    """
    l = len(signal)
    random_indices = np.random.randint(l, size=l)
    resampled_signal = signal[random_indices]
    return resampled_signal


def calculate_covariance(signal, dff):
    """Calculate covariance values between signal and rescaled fluorescence for all neurons.

    Parameters :
    ------------
    signal : 1d array
        external signal, shape (n_timepoints)

    dff : 2d array
        rescaled fluorescence traces for all neurons, shape (n_neurons,n_timepoints)

    Return :
    --------
    covariance : 1d array
        covariance values
    """
    n_neurons = len(dff)
    signal_centered = signal - np.mean(signal)  # subtract the average from the signal
    dff_centered = dff - np.mean(dff, axis=1)[:, np.newaxis]  # subtract the average from every fluorescence trace
    covariance = np.zeros(n_neurons)
    for i in range(n_neurons):  # loop over all neurons
        covariance[i] = np.mean(signal_centered * dff_centered[i])  # evaluate the covariance

    return covariance


def calculate_pvalue(statistic, statistic_bootstrap):
    """Calculate p-values for a set of statistics, based on a null distribution estimated with a set of bootstrapped statistics.

    Parameters :
    ------------
    statistic : 1d array
        values of a statistic for which we want to calculate the p-value

    statistic_bootstrap : 1d array
        bootstrapped values of a statistic from which we estimated the null distribution

    Return :
    --------
    pvalue : 1d array
        p-values
    """
    # ⌨️⬇️
    statistic_bootstrap = statistic_bootstrap.ravel()  # flatten array in case it's not already one-dimensional
    statistic_bootstrap_sorted = np.sort(statistic_bootstrap)  # sort the values in increasing order
    statistic_bootstrap_sorted = np.append(statistic_bootstrap_sorted, statistic_bootstrap_sorted[-1])
    probability_of_smaller_values = np.arange(len(statistic_bootstrap_sorted)) / (
                len(statistic_bootstrap_sorted) - 1)  # express the probability of finding a smaller value of the statistic
    empirical_distribution_function = interp1d(statistic_bootstrap_sorted, probability_of_smaller_values, kind='next',
                                               fill_value='extrapolate')  # interpolate to get the eCDF
    pvalue = np.zeros(len(statistic))
    for i in range(len(statistic)):
        pvalue[i] = 1 - empirical_distribution_function(
            statistic[i])  # probability of finding a larger value according to the null distribution

    return pvalue


def stationary_bootstrap(signal, average_block_size=100):
    """resample a time series using stationary bootstrap, it samples with replacement random blocks of sizes following a geometric distribution.

    Parameters :
    ------------
    signal : array
        original time series, the first dimension corresponds to time, shape (n_timepoints,...)

    average_block_size : integer
        average number of values in a block during resampling
        default : 240

    Return :
    --------
    resampled_signal : array
        bootstrapped time series
    """
    l = len(signal)
    p = 1 / average_block_size  # probability of choosing a new random value
    resampled_signal = np.zeros_like(signal)
    j = np.random.randint(l)  # index of the first random value
    resampled_signal[0] = signal[j]  # assign the first random value
    for i in range(1, l):  # loop over the next positions in the new array
        if np.random.random() < 1 - p:  # sample a random number to choose in between the two options
            j = (j + 1) % l  # take the next index
        else:
            j = np.random.randint(l)  # take a random index
        resampled_signal[i] = signal[j]  # assign the value to the new array
    return resampled_signal


def calculate_covariance_bootstrap(signal, dff, n_resamples=50, average_block_size=240):
    """Calculate covariance values between resampled signal using stationary bootstrap and rescaled fluorescence for all neurons.

    Parameters :
    ------------
    signal : 1d array
        external signal, shape (n_timepoints)

    dff : 2d array
        rescaled fluorescence traces for all neurons, shape (n_neurons,n_timepoints)

    n_resamples: integer
        number of resampled signals used for calculating covariance
        default : 50

    average_block_size : integer
        average number of values in a block during resampling
        default : 240

    Return :
    --------
    covariance_bootstrap : 1d array
        bootstrapped covariance values, shape (n_resamples*n_neurons)
    """
    # ⌨️⬇️
    n_neurons = len(dff)
    covariance_bootstrap = np.zeros(n_resamples * n_neurons)
    from tqdm import tqdm
    for i in tqdm(range(n_resamples)):
        signal_resampled = stationary_bootstrap(signal, average_block_size)  # sample a bootstrapped time series
        covariance_bootstrap[i * n_neurons:(i + 1) * n_neurons] = calculate_covariance(signal_resampled,
                                                                                       dff)  # calculate covariances with neural traces

    return covariance_bootstrap


def find_significant_neurons(statistic, statistic_bootstrap, max_neurons, alpha_bh=0.05, ):
    """Calculate p-values and test if they are significant using the Benjamini-Hochberg procedure.

    Parameters :
    ------------
    statistic : 1d array
        values of the statistic for which we obtained the p-values

    statistic_bootstrap : 1d array
        bootstrapped values of a statistic from which we estimated the null distribution

    alpha_bh : float
        threshold on the false discovery rate
        default : 0.05

    Return :
    --------
    indices_thr : 1d array
        indices of neurons for which we reject the null hypothesis sorted with decreasing value of the statistic

    mask : 1d array
        boolean array which gives True on the indices of neurons for which we reject the null hypothesis
    """
    # ⌨️⬇️
    pvalue = calculate_pvalue(statistic, statistic_bootstrap)

    n_neurons = len(pvalue)
    indices = np.arange(n_neurons) + 1
    sorted_pvalues = np.sort(pvalue)
    line_bh = indices / n_neurons * alpha_bh

    indices_inequality = np.where(sorted_pvalues <= line_bh)[
        0]  # find all indices for which the p-value is below the line
    if len(indices_inequality) == 0:
        n_significant_neurons = 0
    else:
        n_significant_neurons = np.max(indices_inequality) + 1  # largest index

    indices_sorted = np.argsort(-statistic)  # sort indices with decreasing value of the statistic
    indices_thr = indices_sorted[
                  :n_significant_neurons]  # only take the ones for which the statistic is significantly larger than zero
    mask = np.zeros(max_neurons)  # create a mask for the neurons for which we reject the null hypothesis
    mask[indices_thr] = 1
    mask = mask.astype(bool)

    return indices_thr, mask

