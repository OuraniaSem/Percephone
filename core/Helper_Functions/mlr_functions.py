"""Théo Gauvrit, 15/09/2023
Functions for multiple linear regression linear"""

from scipy.interpolate import interp1d
import itertools
import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Helper_Functions.Utils_core import kernel_biexp
from tqdm import tqdm
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")


def regressor_labels(timings, duration, sf, len_signal, amplitude=100):
    """
    Convolve label events to obtain regressor for Multiple Linear Regression
    Parameters
    ----------
    timings: array
        timings in index of the label events
    duration: array
        duration of the label events
    amplitude: int
        amplitude of the label events
    sf: float
        sampling frequency
    len_signal: int
        length of the df/f trace in index

    Returns
    -------
    regressor: array
        convolved behavior label events
    """
    vector = np.zeros(len_signal)
    index = [list(range(stim, int(stim + duration[i]))) for i, stim in enumerate(timings)]
    vector[np.concatenate(index)] = amplitude
    regressor = np.convolve(vector, kernel_biexp(sf), mode='same') * (1/sf)
    return regressor


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


def mlr(dff, regressors, sf):
    """
    Compute Multiple Linear Regression (MLR) between behavior events and neurons activity
    Parameters
    ----------
    dff: array
        delta f/f array for each neurons
    regressors: list
        list of regressors array
    sf: float
        sampling frequency

    Returns
    -------
        text_labels: array
            differents possible between the regressors
        n_neurons_per_label:array
            number of neurons for each labels
    """
    n_regressors = len(regressors)
    df_times = np.linspace(0, len(dff[0]) / sf, len(dff[0]))
    # Linear regression
    reg = LinearRegression().fit(regressors.T, dff.T)
    coef = reg.coef_
    r2 = r2_score(dff.T, reg.predict(regressors.T), multioutput='raw_values')

    n_neurons = len(dff)  # Not sure, problem in the notebooks, don't understand where it came from
    n_resamples = 50
    r2_sbs = np.zeros(n_resamples * n_neurons)

    for i in tqdm(range(n_resamples)):
        dff_resampled = stationary_bootstrap(dff.T).T
        reg = LinearRegression().fit(regressors.T, dff_resampled.T)
        r2_sbs[i * n_neurons:(i + 1) * n_neurons] = r2_score(dff_resampled.T, reg.predict(regressors.T),
                                                             multioutput='raw_values')
    indices_r2, mask_r2 = find_significant_neurons(r2, r2_sbs, max_neurons=len(dff))
    len(indices_r2)
    n_resamples = 100
    coef_bs = np.zeros((n_resamples, n_neurons, n_regressors))
    samples = np.concatenate([regressors, dff])

    for n in tqdm(range(n_resamples)):
        resampled = bootstrap(samples.T).T
        reg = LinearRegression().fit(resampled[:n_regressors].T, resampled[n_regressors:].T)
        coef_bs[n] = reg.coef_

    coef_se = np.zeros_like(coef)
    for i in range(n_neurons):
        coef_se[i] = np.sqrt(n_resamples / (n_resamples - 1)) * np.std(coef_bs[:, i, :], axis=0, ddof=1)

    neuron_labels = np.zeros_like(coef)
    for i in range(n_regressors):
        neuron_labels[
            coef[:, i] > 2 * coef_se[:, i], i] = 1  # if the coefs are larger than 2*SE we set the sign to 1
        neuron_labels[coef[:, i] < -2 * coef_se[:, i], i] = -1  # if they are smaller than -2*SE we set it to -1
    neur_labels = neuron_labels[indices_r2]  # we only consider the neurons for which the fit is good

    all_possible_labels = list(
        itertools.product([0, 1, -1], repeat=n_regressors))  # get all possible combinations of 0, 1 and -1
    n_neurons_per_label = []
    text_labels = []
    for label in all_possible_labels:
        n_neurons_per_label.append(np.sum(np.all(neur_labels == label, axis=1)))  # nb of nrs with a certain label
        text_labels.append(str(label))  # each label saved as a string
    n_neurons_per_label = np.array(n_neurons_per_label)
    text_labels = np.array(text_labels)
    return text_labels, n_neurons_per_label, neuron_labels, indices_r2
