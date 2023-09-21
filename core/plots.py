"""Théo Gauvrit 14/04/2023
Functions for plottings. Most of them use the Recordings class.
"""

import os
import time
import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.cluster.hierarchy import dendrogram, linkage
import core as pc
from scalebars import add_scalebar
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import scipy.interpolate as si
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.switch_backend("Qt5Agg")
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['svg.fonttype'] = 'none'

sampling_rate = 30.9609  # Hz
wt_color = "#326993"
ko_color = "#CC0000"

# Kernel for convolution
tau_r = 0.07  # s0.014
tau_d = 0.236  # s
kernel_size = 10  # size of the kernel in units of tau
a = 5  # scaling factor for biexpo kernel
dt = 1 / sampling_rate  # spacing between successive timepoints
n_points = int(kernel_size * tau_d / dt)
kernel_times = np.linspace(-n_points * dt, n_points * dt,
                           2 * n_points + 1)  # linearly spaced array from -n_points*dt to n_points*dt with spacing dt
kernel_bi = a * (1 - np.exp(-kernel_times / tau_r)) * np.exp(-kernel_times / tau_d)
kernel_bi[kernel_times < 0] = 0


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
    stim_timings = stim_timings[stim_timings < (len(df[0]) - int(record.sf * 3.5))]
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
    ranges = [np.arange(timing - int(sf * 1), timing + int(sf * 3.5)) for timing in event_times]
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


def plot_dff_stim(rec, filename):
    df_inh = rec.df_f_inh
    df_exc = rec.df_f_exc
    cmap = 'inferno'
    time_range = np.linspace(0, (len(df_exc[0]) / sampling_rate) - 1, len(df_exc[0]))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)
    # convolution of the stims
    stim_vector = np.zeros(len(df_exc[0]))
    stim_index = [list(range(stim, stim + int(0.5 * rec.sf))) for i, stim in
                  enumerate(rec.stim_time[rec.stim_ampl != 0])]
    for stim_range, stim_amp in zip(stim_index, rec.stim_ampl[rec.stim_ampl != 0]):
        stim_vector[stim_range] = stim_amp * 100
    conv_stim_det = np.convolve(stim_vector, kernel_bi, mode='same') * dt
    extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(df_exc) - 0.5, -0.5]
    tax.imshow(conv_stim_det.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    Z = linkage(df_exc, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(df_exc[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(df_exc), 1),
                   vmax=np.nanpercentile(np.ravel(df_exc), 99), extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'$\Delta F/F$')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Neurons')
    tax.set_title(filename)
    plt.show()


def plot_dff_stim_detected(rec, filename):
    df_inh = rec.df_f_inh
    df_exc = rec.df_f_exc
    cmap = 'inferno'
    time_range = np.linspace(0, (len(df_exc[0]) / sampling_rate) - 1, len(df_exc[0]))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax2 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)
    # convolution of the stims
    stim_vector = np.zeros(len(df_exc[0]))
    stim_vector_det =  np.zeros(len(df_exc[0]))
    stim_index = [list(range(stim, stim + int(0.5 * rec.sf))) for i, stim in
                  enumerate(rec.stim_time[rec.stim_ampl != 0])]
    for stim_range, stim_amp in zip(stim_index, rec.stim_ampl[rec.stim_ampl != 0]):
        stim_vector[stim_range] = stim_amp * 100
    conv_stim = np.convolve(stim_vector, kernel_bi, mode='same') * dt
    for stim_range, stim_amp in zip(np.array(stim_index)[rec.detected_stim[rec.stim_ampl != 0]], rec.stim_ampl[(rec.detected_stim) & (rec.stim_ampl != 0)]):
        stim_vector_det[stim_range] = stim_amp * 100
    conv_stim_det = np.convolve(stim_vector_det, kernel_bi, mode='same') * dt
    extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(df_exc) - 0.5, -0.5]
    tax2.imshow(conv_stim_det.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    tax1.imshow(conv_stim.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    Z = linkage(df_exc, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(df_exc[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(df_exc), 1),
                   vmax=np.nanpercentile(np.ravel(df_exc), 99), extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'$\Delta F/F$')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Neurons')
    tax1.set_title(filename)
    plt.show()


def group_heat_map_per_stim(df_f_exc, df_f_inh, dn_exc, dn_inh, name, filename):
    start_time = time.time()
    print("Plotting heatmap.")
    time_range = np.linspace(-1, (len(df_f_exc[0]) / sampling_rate) - 1, len(df_f_exc[0]))
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
    ax1.tick_params(which='both', width=4)
    ax2.tick_params(which='both', width=4)
    ax1.set_title(str(name))
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(filename)
    plt.show()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))


def group_heat_map_per_stim_split(df_f, dn, name, filename,legend):
    start_time = time.time()
    print("Plotting heatmap.")
    time_range = np.linspace(-1, (len(df_f[0]) / sampling_rate) - 1, len(df_f[0]))
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 10))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    df_f_clustered = df_f[dn['leaves']]
    ax1.pcolor(time_range, np.arange(len(df_f_clustered)), df_f_clustered, vmin=0, vmax=50, cmap="Reds")
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(legend)
    ax1.tick_params(which='both', width=4)
    ax1.set_title(str(name))
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(filename)
    plt.show()
    print("--- %s seconds ---" % round(time.time() - start_time, 2))


def df_f_per_stim(rec, stim, exc_ids, inh_ids, color):
    print("Plotting heat map per stimulation.")
    start_time = time.time()
    window_response = int(4 * sampling_rate)
    stim_timings = np.array(rec.stim_time)[rec.stim_ampl == stim]
    stim_ranges = [np.arange(stim_timing, stim_timing + window_response)
                   for stim_timing in stim_timings]
    output_exc = np.zeros((len(exc_ids), window_response))
    output_inh = np.zeros((len(inh_ids), window_response))
    for i, row in enumerate(rec.df_f_exc[exc_ids]):
        output_exc[i] = np.mean(row[np.array(stim_ranges)], axis=0)
    for i, row in enumerate(rec.df_f_inh[inh_ids]):
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
    fig.savefig(rec.input_path + "ex_df_resp_" + str(stim) + ".pdf")


def df_f_graph(rec, neuron_id):
    """ Work only for one excitatory neuron id for the moment"""
    print("Plotting delta f over f trace for neuron " + str(neuron_id))
    session_length = len(rec.df_f_exc[0])
    session_duration = session_length / sampling_rate
    time_range = np.linspace(0, session_duration, session_length)
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex="all", gridspec_kw={'height_ratios': [0.2, 3]})
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs[1].plot(time_range, rec.df_f_exc[neuron_id])
    plt.xlabel('Time (seconds)', fontsize=30)
    plt.ylabel('Delta_f_over_f', fontsize=30)
    plt.tick_params(axis="x", which="both", width=0)
    plt.tick_params(axis="y", which="both", width=0)
    axs[0].vlines(rec.stim_time[rec.stim_time / 1000 < session_duration] / 1000, ymin=0.5,
                  ymax=rec.stim_ampl, color='black', lw=2)
    fig.tight_layout()
    plt.show()


def df_f_graph_trials(rec, stim, neuron_id, ntrials=5, color=wt_color):
    print("Plotting example trials.")
    start_time = time.time()
    window_response = int(4 * sampling_rate)
    stim_timings = np.array(rec.stim_time)[rec.stim_ampl == stim]
    time_range = np.linspace(0, window_response / sampling_rate, window_response)
    fig, axs = plt.subplots(1, 1, figsize=(7, 12))
    for i in range(ntrials):
        trace = rec.df_f_exc[neuron_id, stim_timings[i]:stim_timings[i] + window_response]
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
    fig.savefig(rec.input_path + "example_trials_df_resp_" + str(stim) + ".pdf")


if __name__ == '__main__':
    # directory = "/datas/Théo/Projects/Percephone/data/StimulusOnlyWT/"
    directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
    roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
    files = ["20220715_4456_00_synchro"]
    for folder in files:
        if os.path.isdir(directory + folder):
            path = directory + folder + '/'
            recording = pc.RecordingAmplDet(path, 0, folder, roi_info, correction=False, no_cache=True)
            plot_dff_stim_detected(recording, folder)
