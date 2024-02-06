"""
Théo Gauvrit 18/01/2024
analysis of neuronal response to stimulus
-response
-onset
-AUC
"""

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random as rnd
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from percephone.analysis.utils import kernel_biexp
matplotlib.use("Qt5Agg")
plt.switch_backend("Qt5Agg")

sf = 30.9609  # Hz
pre_boundary = int(0.25 * sf)  # index
post_boundary = int(0.5 * sf)  # index


def resp_single_neuron(neuron_df_data_, random_timing, stim_idx):
    """
    Compute if the neurons are
    Parameters
    ----------
    neuron_df_data_: array
        df over f data inh or exc
    random_timing: array
    stim_idx: array

    Returns
    -------
     resp: array
    """
    resp = []
    bootstrap_responses = []
    for rnd_idx in random_timing:
        bootstrap_responses.append(ss.iqr(neuron_df_data_[rnd_idx - pre_boundary:rnd_idx], nan_policy='omit'))
    threshold = np.nanpercentile(bootstrap_responses, 95)  # 2 * np.nanper... before 11-09-2023
    for y, stim_i in enumerate(stim_idx):
        # bsl_activity = np.subtract(*np.nanpercentile((neuron_df[(int(stim_timing * sf) - pre_boundary):int(stim_timing * sf)]), [75, 25]))
        bsl_activity = np.mean(neuron_df_data_[(stim_i - pre_boundary):stim_i])
        peak_high = np.max(neuron_df_data_[stim_i:(stim_i + post_boundary)])
        true_response = peak_high - bsl_activity
        if true_response > threshold:
            resp.append(True)
        else:
            resp.append(False)
    return resp


def resp_matrice(rec, df_data):
    """ Method with interquartile measure of the baseline

    Parameters
    ----------
    rec: Recording that will be processed
    df_data :  numpy array
        delta f over f (neurons,time) can be exc or inh
    """
    pre_boundary = int(0.25 * rec.sf)  # index
    post_boundary = int(0.5 * rec.sf)
    exclude_windows = [list(range(t, t + post_boundary)) for t in rec.stim_time]
    exclude_windows.append(list(range(0, pre_boundary)))  # to not have edge problem
    exclude_windows.append(
        list(range(len(df_data[0]) - post_boundary, len(df_data[0]))))  # to not have edge problem
    range_iti = set(range(len(df_data[0]))).difference(set(np.concatenate(exclude_windows)))
    random_timing = rnd.sample(list(range_iti), k=1999)

    from multiprocessing import Pool, cpu_count
    workers = cpu_count()
    pool = Pool(processes=workers)
    async_results = [pool.apply_async(resp_single_neuron, args=(i, random_timing, rec.stim_time)) for i in df_data]
    resp_mat = [ar.get() for ar in async_results]
    return resp_mat


def auc_matrice(rec, df_data, resp_mask):
    """
    Perform Area Under the Curve for evey responses trace to the stimulations
    Parameters
    ----------
    rec : Recording
        recording to analyze
    df_data: array
        df over f data inh or exc
    resp_mask: array
        boolean matrice (n neurones x n stims) indicating if there a response or not to the stim

    Returns
    -------
    auc_mat: array
        AUC matrice (n neurones x n stims) for each response in every neurons


    """

    def auc(signal, responsive):
        if not responsive:
            return False
        signal[signal < 0] = 0
        return simps(signal, dx=1 / rec.sf)

    timings = rec.stim_time
    if rec.stim_time[-1] + int(0.5 * rec.sf) >= len(df_data[0]):
        timings = rec.stim_time[0:-1]
    data = df_data[:, np.linspace(timings,
                                  timings + int(0.5 * rec.sf),
                                  num=int(0.5 * rec.sf) + 1, dtype=int)]
    data1 = np.swapaxes(data, 1, 2)
    auc_mat = [list(map(auc, x, responsive)) for x, responsive in zip(data1, resp_mask)]
    return auc_mat


def delay_response(signal, responsive):
    # 0-10(stim)-45
    sf = 30.9609  # Hz
    if not responsive:
        return False
    smooth_sig = savgol_filter(signal, 5, 1)
    baseline = np.mean(smooth_sig[0:10])
    st_vector = np.zeros(len(smooth_sig))
    st_vector[10] = np.abs(np.max(smooth_sig[10:25])) + 0.50 * np.abs(np.max(smooth_sig[10:25]))
    conv_st = np.convolve(st_vector, kernel_biexp(sf), mode='same') * 1 / sf
    conv_st = conv_st + baseline
    rs_ = []
    for y in range(15):
        conv_ = range(5, 10 + int(0.5 * sf))
        portion_sig = smooth_sig[10 + y: 30 + y]
        conv_ = np.array(conv_)[np.array(conv_) < len(smooth_sig)]
        reg_fit = LinearRegression().fit(conv_st[conv_][:, np.newaxis], portion_sig[:, np.newaxis])
        r_ = reg_fit.score(conv_st[conv_][:, np.newaxis], portion_sig[:, np.newaxis])
        rs_.append(r_)

    delay = np.argmax(rs_)
    return delay


def delay_matrice(rec, df_data, stims, resp_mask):
    ar_stims = np.linspace(stims - 10, stims+45, 56, dtype=int)  # linspace frames
    signals = np.array([signal[ar_stims] for signal in df_data])
    signals = np.swapaxes(signals, 1, 2)
    delay_mat = [list(map(delay_response, df_n, resp_n)) for df_n, resp_n in tqdm(zip(signals, resp_mask))]
    return delay_mat


def group_matrices(recs, savename, no_cache=False):
    output = pd.DataFrame()
    for rec in recs:
        rec.responsivity()
        rec.delay_onset()
        rec.auc()
        onset_exc = np.nanmean(rec.matrices["EXC"]["Delay_onset"])
        onset_inh = np.nanmean(rec.matrices["INH"]["Delay_onset"])
        auc_exc = np.nanmean(rec.matrices["EXC"]["AUC"])
        auc_inh = np.nanmean(rec.matrices["INH"]["AUC"])
        output = output.append({"Filename": rec.filename,
                                "Group": rec.genotype,
                                "Delay EXC": onset_exc,
                                "Delay INH": onset_inh,
                                "AUC EXC": auc_exc,
                                "AUC INH": auc_inh
                                }, ignore_index=True)
    output.to_csv(savename)
