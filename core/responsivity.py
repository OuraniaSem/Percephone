"""Théo Gauvrit 18/04/2023
Charecterization of the responsivity for all neurons.
"""
import random as rnd

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as ss

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Qt5Agg")

sf = 30.9609  # Hz
pre_boundary = int(0.25 * sf)  # index
post_boundary = int(1 * sf)  # index


def responsive_prsa_et_al_method(df, stim_times):
    """ Method with max minus baseline """
    responsive = np.zeros((len(df), len(stim_times)))
    for i, neuron_df in enumerate(df):
        print("Neuron " + str(i))
        exclude_windows = [range(int(t * sf), int(t * sf) + post_boundary) for t in stim_times]
        exclude_windows.append(range(0, pre_boundary))
        exclude_windows.append(range(len(df[0]) - post_boundary, len(df[0])))
        range_iti = set(range(len(neuron_df))).difference(set(np.array(exclude_windows).flatten()))
        random_timing = rnd.sample(range_iti, k=1999)
        bootstrap_responses = []
        for rnd_idx in random_timing:
            bsl_activity = np.mean(neuron_df[(rnd_idx - pre_boundary):rnd_idx])
            peak_high = np.max(neuron_df[rnd_idx:(rnd_idx + post_boundary)])
            bootstrap_responses.append(peak_high - bsl_activity)
        for y, stim_timing in enumerate(stim_times):
            bsl_activity = np.mean(neuron_df[(int(stim_timing * sf) - pre_boundary):int(stim_timing * sf)])
            peak_high = np.max(neuron_df[int(stim_timing * sf):(int(stim_timing * sf) + post_boundary)])
            true_response = peak_high - bsl_activity
            if true_response > np.nanpercentile(bootstrap_responses, 95):
                responsive[i, y] = True
            else:
                responsive[i, y] = False
    return responsive


def resp_single_neuron(neuron_df_data_, random_timing, stim_idx):
    resp = []
    bootstrap_responses = []
    for rnd_idx in random_timing:
        bootstrap_responses.append(ss.iqr(neuron_df_data_[rnd_idx - pre_boundary:rnd_idx], nan_policy='omit'))
    threshold = 2 * np.nanpercentile(bootstrap_responses, 95)
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


def responsive_iq_method(df_data, stim_idx):
    """ Method with interquartile measure of the baseline

    Parameters
    ----------
    df_data :  numpy array
        delta f over f (neurons,time)
    stim_idx :  numpy array
        stim indexes

    """
    exclude_windows = [list(range(t, t + post_boundary)) for t in stim_idx]
    exclude_windows.append(list(range(0, pre_boundary)))
    exclude_windows.append(list(range(len(df_data[0]) - post_boundary, len(df_data[0]))))
    range_iti = set(range(len(df_data[0]))).difference(set(np.concatenate(np.array(exclude_windows))))
    random_timing = rnd.sample(range_iti, k=1999)

    from multiprocessing import Pool, cpu_count
    workers = cpu_count()
    pool = Pool(processes=workers)
    async_results = [pool.apply_async(resp_single_neuron, args=(i, random_timing, stim_idx)) for i in df_data]
    results = [ar.get() for ar in async_results]
    return results


def responsivity(record, row_metadata):
    """
    Compute different responsivity parameters for each neurons and return a summary for the whole recording as dataframe

    Parameters
    -------
    record: Recording object
    row_metadata:   pandas.Series()
                    Contains metadatas of the recording

    Returns
    -------
    summary_resp:   pandas.DataFrame()
                    Contains different responsivity parameters for each amplitude of stimulation(rows)
    resp_neurons:   pandas.DataFrame()
                    Contains different responsivity parameters for each neurons

    """
    print("Calculation responsivity")
    filename = row_metadata["Number"].values[0]
    group = row_metadata["Genotype"].values[0]
    date = str(row_metadata["Date"].values[0])

    stim_timings, stim_ampl = record.stim_time, record.stim_ampl
    stims = np.unique(stim_ampl)
    resp_overall_exc = {}
    resp_overall_inh = {}
    summary_resp = pd.DataFrame()
    resp_neurons = pd.DataFrame()
    for amp in stims:
        print("Amplitude: " + str(amp))
        stim_times = stim_timings[stim_ampl == amp]

        responsiveness_exc = responsive_iq_method(record.df_f_exc, stim_times)
        responsiveness_inh = responsive_iq_method(record.df_f_inh, stim_times)
        print("Generating dataframes")
        resp_overall_exc[str(amp)] = responsiveness_exc
        resp_overall_inh[str(amp)] = responsiveness_inh
        total_responsivity_exc = np.sum(responsiveness_exc, axis=1)
        total_responsivity_inh = np.sum(responsiveness_inh, axis=1)
        response_per_trials_exc = np.sum(responsiveness_exc, axis=0)
        response_per_trials_inh = np.sum(responsiveness_inh, axis=0)
        resp = {"Filename": filename, "Genotype": group, "Date": date, "Amplitude": amp,
                "total exc": len(record.df_f_exc), "total inh": len(record.df_f_inh),
                "mean_nb_exc_per_trials": np.mean(response_per_trials_exc),
                "mean_nbinh_per_trials": np.mean(response_per_trials_inh),
                "tbt_var_exc": np.std(response_per_trials_exc),
                "tbt_var_inh": np.std(response_per_trials_inh),
                "nb responsive exc units >2": (total_responsivity_exc > 1).sum(),
                "nb responsive inh units >2": (total_responsivity_inh > 1).sum(),
                "responsivity of neurons exc >2": np.mean(
                    total_responsivity_exc[total_responsivity_exc > 1]) / len(
                    stim_times),
                "responsivity of neurons inh >2": np.mean(
                    total_responsivity_inh[total_responsivity_inh > 1]) / len(
                    stim_times)}
        # for percent in [0.10, 0.30, 0.5, 0.60, 0.9]:
        summary_resp = summary_resp.append(resp, ignore_index=True)
        resp_neurons[str(amp)] = np.concatenate(
            [total_responsivity_exc / len(stim_times), total_responsivity_inh / len(stim_times)])
    resp_neurons["Filename"] = [filename] * (len(total_responsivity_exc) + len(total_responsivity_inh))
    resp_neurons["Date"] = [date] * (len(total_responsivity_exc) + len(total_responsivity_inh))
    resp_neurons["Genotype"] = [group] * (len(total_responsivity_exc) + len(total_responsivity_inh))
    resp_neurons["Type"] = np.concatenate(
        [["exc"] * len(total_responsivity_exc), ["inh"] * len(total_responsivity_inh)])
    resp_neurons["STD baseline"] = np.concatenate(
        [np.std(record.df_f_exc, axis=1), np.std(record.df_f_inh, axis=1)])
    resp_neurons["Mean baseline"] = np.concatenate(
        [np.mean(record.df_f_exc, axis=1), np.mean(record.df_f_inh, axis=1)])
    record.sum_resp = summary_resp
    record.resp_neurons = resp_neurons
    record.name = filename
    record.group = group
    record.date = date
    return summary_resp, resp_neurons


if __name__ == '__main__':
    # directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format/"
    # roi_info = pd.read_excel(
    #     "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format/FmKO_ROIs&inhibitory.xlsx")
    # summary_resp_glob = pd.DataFrame()
    # output_neurons_resp = pd.DataFrame()
    # for folder in os.listdir(directory):
    #     if os.path.isdir(directory + folder):
    #         path = directory + folder + '/'
    #         row = roi_info[roi_info["Number"] == int(folder[9:13])]
    #         inhib_ids = np.array(list(list(row["Inhibitory neurons: ROIs"])[0].split(", ")))
    #         recording = pc.RecordingStimulusOnly(path, inhibitory_ids=inhib_ids.astype(int))
    #         sum_resp, neurons_resp = recording.responsivity(row)
    #         summary_resp_glob = pd.concat([summary_resp_glob, sum_resp])
    #         output_neurons_resp = pd.concat([output_neurons_resp, neurons_resp])
    # summary_resp_glob.to_csv("global_responsiveness.csv")
    # output_neurons_resp.to_csv("neurons_resp.csv")

    """Plot to verify manually the quality of responsivity detection"""
    # df, stim_timings, stim_ampl = recording.df_f_exc, recording.stim_time, recording.stim_ampl
    # choosen_amp = 12
    # stim_times = stim_timings[stim_ampl == choosen_amp]
    # for i in range(len(stim_times)):
    #     # i = 29
    #     response = [3, i]
    #     df_to_plot = df[response[0], int(stim_times[response[1]]*sf-(pre_boundary*3)):int(stim_times[response[1]]*sf+(post_boundary*3))]
    #     fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [0.2, 3]})
    #     axs[0].spines['right'].set_visible(False)
    #     axs[0].spines['top'].set_visible(False)
    #     axs[0].spines['left'].set_visible(False)
    #     axs[0].spines['bottom'].set_visible(False)
    #     axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #     axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    #     axs[1].plot(df_to_plot)
    #     plt.ylabel('DF/F0', fontsize=30)
    #     plt.tick_params(axis="x", which="both", width=0)
    #     plt.tick_params(axis="y", which="both", width=0)
    #     plt.title("Stim n°" + str(i) + " considered " + str(output_neurons_df[str(choosen_amp)][response[0], response[1]]))
    #     axs[0].vlines((pre_boundary*3), ymin=0, ymax=12, color='black', lw=2)
    #     plt.ylim([0, 400])
    #     fig.tight_layout()
    #     plt.show()
