"""Théo Gauvrit 14/04/2023
Functions for plottings. Most of them use the Recordings class.
"""

import time

import matplotlib
import numpy as np
import scipy.signal as ss
from scipy.cluster.hierarchy import dendrogram, linkage

import core as pc
from scalebars import add_scalebar

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import scipy.interpolate as si

plt.switch_backend("Qt5Agg")
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['svg.fonttype'] = 'none'

sampling_rate = 30.9609  # Hz
wt_color = "#326993"
ko_color = "#CC0000"


def interpolation_frames(df):
    """
    Fill the missing frames with interpolation if the sampling rate
    is 21.2209 Hz for some files for the 3.5s window range
        int(3.5*30.9609) = 108
        int(3.5*21.2209) = 74
        dif= 34"""
    index_init = np.arange(0, 108)
    index = np.delete(index_init, (index_init[::3])[1:-1])
    f = si.interp1d(index, df)
    df_extended = f(index_init)
    return df_extended


def peristimulus(record, stim, inh=False):
    stim_times, stim_ampl = record.stim_time, record.stim_ampl
    if inh:
        df = record.df_f_inh
    else:
        df = record.df_f_exc
    stim_timings = stim_times[stim_ampl == stim]
    stim_timings = stim_timings[stim_timings < (len(df[0])-int(record.sf * 3.5))]
    stim_ranges = [np.arange(stim_timing, stim_timing + int(record.sf * 3.5))
                   for stim_timing in stim_timings]
    output = np.zeros((len(df), int(record.sf * 3.5)))
    print(stim)
    for i, r in enumerate(df):
        try:
            output[i] = np.mean(r[np.array(stim_ranges)], axis=0)
        except IndexError as e:
            if stim in np.unique(record.stim_ampl):
                print("IndexError for the neurone n°" + str(i))
    return output


def perirevent(times, df, sf):
    event_times = times[times < (len(df[0]) - int(sf * 3.5))]
    ranges = [np.arange(timing - int(sf*1), timing + int(sf * 3.5))for timing in event_times]
    output = np.zeros((len(df), int(sf * 4.48)))  # 4.48 and not 4.5 because int(sf*4.5)>int(sf)+int(sf*3.5)
    for i, r in enumerate(df):
        output[i] = np.mean(r[np.array(ranges)], axis=0)
    return output


def heat_map(df_f, stim_times, stim_ampl):
    print("Plotting heatmap.")
    session_length = len(df_f[0])
    session_duration = session_length / sampling_rate
    time_range = np.linspace(0, session_duration, session_length)
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex="all", gridspec_kw={'height_ratios': [0.2, 3]})
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    df_f_clustered = df_f
    axs[1].pcolor(time_range, np.arange(len(df_f_clustered)), df_f_clustered, vmin=0, vmax=50, cmap="Reds")
    plt.xlabel('Time (seconds)', fontsize=30)
    plt.ylabel('Neurons', fontsize=30)
    plt.tick_params(axis="x", which="both", width=0)
    plt.tick_params(axis="y", which="both", width=0)
    axs[0].vlines(stim_times[stim_times < session_duration], ymin=0.5, ymax=stim_ampl, color='black', lw=2)
    fig.tight_layout()
    plt.show()


def heat_map_per_stim(df_f, stim_times, stim_ampl):
    print("Plotting heat map per stimulation.")
    start_time = time.time()
    stims = np.unique(stim_ampl)
    window_response = int(3.5 * sampling_rate)
    for stim in stims[::-1]:
        stim_timings = stim_times[stim_ampl == stim]
        stim_ranges = [np.arange(stim_timing, stim_timing + window_response)
                       for stim_timing in stim_timings]
        output = np.zeros((len(df_f), window_response))
        for i, row in enumerate(df_f):
            output[i] = np.mean(row[np.array(stim_ranges)], axis=0)
        print("Plotting heatmap.")
        time_range = np.linspace(0, window_response / sampling_rate, window_response)
        if stim == 12:
            Z = linkage(output, 'ward')
            dn = dendrogram(Z, no_plot=True)
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        df_f_clustered = output[dn['leaves']]
        ax.pcolor(time_range, np.arange(len(df_f_clustered)), df_f_clustered, vmin=0, vmax=50, cmap="Reds")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Neurons')
        plt.title("Stimulus " + str(stim) + " amp")
        fig.tight_layout()
        fig.savefig("heat_map_4939_04_" + str(stim) + "amp.png")
        plt.show()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))


def group_heat_map_per_stim(df_f_exc, df_f_inh, dn_exc, dn_inh, name, filename):
    start_time = time.time()
    print("Plotting heatmap.")
    time_range = np.linspace(-1, (len(df_f_exc[0]) / sampling_rate)-1, len(df_f_exc[0]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex="all", gridspec_kw={'height_ratios': [3, 1]})
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    df_f_clustered = df_f_exc[dn_exc['leaves']]
    ax1.pcolor(time_range, np.arange(len(df_f_clustered)), df_f_clustered, vmin=0, vmax=50, cmap="Reds")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    df_f_clustered = df_f_inh[dn_inh['leaves']]
    ax2.pcolor(time_range, np.arange(len(df_f_clustered)), df_f_clustered, vmin=0, vmax=50, cmap="Reds")
    ax2.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Exc neurons')
    ax2.set_ylabel('Inh neurons')
    ax1.set_title(str(name))
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    fig.tight_layout()
    fig.savefig(filename)
    plt.show()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))


def df_f_per_stim(recording, stim, exc_ids, inh_ids, color):
    print("Plotting heat map per stimulation.")
    start_time = time.time()
    window_response = int(4 * sampling_rate)
    stim_timings = np.array(recording.stim_time)[recording.stim_ampl == stim]
    stim_ranges = [np.arange(stim_timing, stim_timing + window_response)
                   for stim_timing in stim_timings]
    output_exc = np.zeros((len(exc_ids), window_response))
    output_inh = np.zeros((len(inh_ids), window_response))
    for i, row in enumerate(recording.df_f_exc[exc_ids]):
        output_exc[i] = np.mean(row[np.array(stim_ranges)], axis=0)
    for i, row in enumerate(recording.df_f_inh[inh_ids]):
        output_inh[i] = np.mean(row[np.array(stim_ranges)], axis=0)
    time_range = np.linspace(0, window_response / sampling_rate, window_response)
    fig, axs = plt.subplots(2, 1, sharex="all", figsize=(7, 12), gridspec_kw={'height_ratios': [3, 1]})
    for i, trace in enumerate(output_exc):
        axs[0].plot(time_range, ss.savgol_filter(np.add(trace, i * 100), 3, 1), lw=3, color=color)
    for i, trace in enumerate(output_inh):
        axs[1].plot(time_range, ss.savgol_filter(np.add(trace, i * 100), 3, 1), lw=3, color=color)
    axs[1].tick_params(which='both', width=4)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs[0].set_title("Stimulus " + str(stim) + " amp", fontsize=40)
    axs[1].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Exc')
    axs[1].set_ylabel('Inh')
    axs[1].set_ylim([-50, 150])
    axs[0].set_ylim([0, 600])
    add_scalebar(axs[0], matchy=False, matchx=False, sizey=100, sizex=0, hidey=False, labelx="", labely="", barwidth=3,
                 sep=4)
    fig.tight_layout()
    plt.show()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    fig.savefig(recording.input_path + "ex_df_resp_" + str(stim) + ".pdf")


def df_f_graph(recording, neuron_id):
    """ Work only for one excitatory neuron id for the moment"""
    print("Plotting delta f over f trace for neuron " + str(neuron_id))
    session_length = len(recording.df_f_exc[0])
    session_duration = session_length / sampling_rate
    time_range = np.linspace(0, session_duration, session_length)
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex="all", gridspec_kw={'height_ratios': [0.2, 3]})
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs[1].plot(time_range, recording.df_f_exc[neuron_id])
    plt.xlabel('Time (seconds)', fontsize=30)
    plt.ylabel('Delta_f_over_f', fontsize=30)
    plt.tick_params(axis="x", which="both", width=0)
    plt.tick_params(axis="y", which="both", width=0)
    axs[0].vlines(recording.stim_time[recording.stim_time / 1000 < session_duration] / 1000, ymin=0.5,
                  ymax=recording.stim_ampl, color='black', lw=2)
    fig.tight_layout()
    plt.show()


def df_f_graph_trials(recording, stim, neuron_id, ntrials=5, color=wt_color):
    print("Plotting example trials.")
    start_time = time.time()
    window_response = int(4 * sampling_rate)
    stim_timings = np.array(recording.stim_time)[recording.stim_ampl == stim]
    time_range = np.linspace(0, window_response / sampling_rate, window_response)
    fig, axs = plt.subplots(1, 1, figsize=(7, 12))
    for i in range(ntrials):
        trace = recording.df_f_exc[neuron_id, stim_timings[i]:stim_timings[i] + window_response]
        axs.plot(time_range, ss.savgol_filter(np.add(trace, i * 500), 3, 1), lw=3, color=color)
    axs.tick_params(which='both', width=4)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs.set_title("Stimulus " + str(stim) + " amp", fontsize=40)
    axs.set_xlabel('Time (seconds)')
    axs.set_ylabel('Exc')
    add_scalebar(axs, matchy=False, matchx=False, sizey=100, sizex=0, hidey=False, labelx="", labely="", barwidth=3,
                 sep=4)
    fig.tight_layout()
    plt.show()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    fig.savefig(recording.input_path + "example_trials_df_resp_" + str(stim) + ".pdf")


if __name__ == '__main__':
    path = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/4445/20220710_4445_00_synchro/"
    test_rec = pc.RecordingAmplDet(path, starting_trial=0, inhibitory_ids=[7, 24, 34, 73, 89, 103, 683],
                                        sf=30.9609)

    heat_map_per_stim(test_rec.df_f_inh, test_rec.stim_time, test_rec.stim_ampl)

    # df_f_graph(test_rec, neuron_id=0)
    # for i in range(20):
    #     df_f_graph_trials(recording=test_rec, stim=12, ntrials=5, neuron_id=i, color=wt_color)
    # df_f_per_stim(recording=test_rec, stim=12, exc_ids=[23, 5, 9, 11, 32, 10], inh_ids=[1, 2], color=ko_color)

    analog_trace = test_rec.analog.iloc[:, 1]
    # # analog_trace = test_rec.analog["stimulus"].to_numpy()

    trace = test_rec.df_f_exc[1]
    # print(len(analog_trace)/10000)
    # print(len(trace) / sampling_rate)

    # # SHARE X percentage
    # fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex="all")
    # time_range = np.linspace(0, 100, len(analog_trace))
    # axs[0].plot(time_range, analog_trace)
    # time_range = np.linspace(0, 100, len(trace))
    # axs[1].plot(time_range, trace)
    # axs[1].plot((stim_onset_idx1/len(trace))*100, [200] * len(stim_onset_idx1), 'x')
    # plt.show()

    # SHARE X seconds
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex="all")
    time_range = np.linspace(0, len(analog_trace) / 10000, len(analog_trace))
    axs[0].plot(time_range, analog_trace)
    time_range = np.linspace(0, len(trace) / sampling_rate, len(trace))
    axs[1].plot(time_range, trace)
    axs[1].plot((test_rec.stim_time / sampling_rate), [200] * len(test_rec.stim_time), 'x')
    plt.show()

    # # # NO share x
    # # fig, axs = plt.subplots(2, 1, figsize=(18, 10))
    # # axs[0].plot(analog_trace)
    # # axs[1].plot(trace)
    # # axs[1].plot(stim_onset_idx1, [200]*len(stim_onset_idx1), 'x')
    # # plt.show()

    # #diff percentage:
    # diff_stim_analog = np.subtract(test_rec.stim_time.flatten(), len(analog_trace))/len(analog_trace)
    # diff_df = np.subtract(stim_onset_idx1.flatten(), len(test_rec.df_f_inh[0]))/len(test_rec.df_f_inh[0])
    # fig, axs = plt.subplots(1, 1, figsize=(18, 10))
    # axs.plot(diff_df - diff_stim_analog)
    # plt.show()
