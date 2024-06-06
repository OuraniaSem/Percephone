"""
Théo Gauvrit 18/01/2024
New inferno traces colored heatmaps
"""
import os

import matplotlib
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from percephone.core.recording import RecordingAmplDet
from percephone.plts.utils import get_zscore

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import math

plt.switch_backend("Qt5Agg")
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams["xtick.major.width"] = 3
plt.rcParams["ytick.major.width"] = 3

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


def plot_dff_stim_detected(rec, dff):
    cmap = 'inferno'
    time_range = np.linspace(0, (len(dff[0]) / sampling_rate) - 1, len(dff[0]))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax2 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)
    # convolution of the stims
    stim_vector = np.zeros(len(dff[0]))
    stim_vector_det = np.zeros(len(dff[0]))
    stim_index = [list(range(stim, stim + int(0.5 * rec.sf))) for i, stim in
                  enumerate(rec.stim_time[rec.stim_ampl != 0])]
    for stim_range, stim_amp in zip(stim_index, rec.stim_ampl[rec.stim_ampl != 0]):
        stim_vector[stim_range] = stim_amp * 100
    conv_stim = np.convolve(stim_vector, kernel_bi, mode='same') * dt
    for stim_range, stim_amp in zip(np.array(stim_index)[rec.detected_stim[rec.stim_ampl != 0]], rec.stim_ampl[(rec.detected_stim) & (rec.stim_ampl != 0)]):
        stim_vector_det[stim_range] = stim_amp * 100
    conv_stim_det = np.convolve(stim_vector_det, kernel_bi, mode='same') * dt
    extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(dff) - 0.5, -0.5]
    tax2.imshow(conv_stim_det.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    tax1.imshow(conv_stim.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    Z = linkage(dff, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(dff[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(dff), 1),
                   vmax=np.nanpercentile(np.ravel(dff), 99), extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'Z-score')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    tax1.set_title(rec.filename)
    plt.show()


def plot_dff_stim_detected_timeout(rec, dff):
    """

    Parameters
    ----------
    rec
    dff
    filename

    Returns
    -------

    """
    cmap = 'inferno'
    time_range = np.linspace(0, (len(dff[0]) / sampling_rate) - 1, len(dff[0]))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax2 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)
    # convolution of the stims
    stim_vector = np.zeros(len(dff[0]))
    stim_vector_det = np.zeros(len(dff[0]))
    stim_index = [list(range(stim, stim + int(0.5 * rec.sf))) for i, stim in
                  enumerate(rec.stim_time[rec.stim_ampl != 0])]
    for stim_range, stim_amp in zip(stim_index, rec.stim_ampl[rec.stim_ampl != 0]):
        stim_vector[stim_range] = stim_amp * 100
    conv_stim = np.convolve(stim_vector, kernel_bi, mode='same') * dt
    timeout_index = [list(range(to, to + int(0.5 * rec.sf))) for i, to in
                  enumerate(rec.timeout_time)]
    for to_range in timeout_index:
        stim_vector_det[to_range] = 1000
    conv_stim_det = np.convolve(stim_vector_det, kernel_bi, mode='same') * dt
    extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(dff) - 0.5, -0.5]
    tax2.imshow(conv_stim_det.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    tax1.imshow(conv_stim.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    Z = linkage(dff, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(dff[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(dff), 1),
                   vmax=np.nanpercentile(np.ravel(dff), 99), extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'Z-score')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    tax1.set_title(rec.filename)
    plt.show()
    return fig


def plot_dff_stim_detected_lick(rec, dff):
    """

    Parameters
    ----------
    rec
    dff
    filename

    Returns
    -------

    """
    cmap = 'inferno'
    time_range = np.linspace(0, (len(dff[0]) / sampling_rate) - 1, len(dff[0]))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax2 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)
    # convolution of the stims
    stim_vector = np.zeros(len(dff[0]))
    stim_vector_lick = np.zeros(len(dff[0]))
    stim_index = [list(range(stim, stim + int(0.5 * rec.sf))) for i, stim in
                  enumerate(rec.stim_time[rec.stim_ampl != 0])]
    for stim_range, stim_amp in zip(stim_index, rec.stim_ampl[rec.stim_ampl != 0]):
        stim_vector[stim_range] = stim_amp * 100
    conv_stim = np.convolve(stim_vector, kernel_bi, mode='same') * dt
    lick_index = [list(range(to, to + int(0.5 * rec.sf))) for i, to in
                  enumerate(rec.lick_time[rec.lick_time<len(stim_vector_lick)-16])]
    for to_range in lick_index:
        stim_vector_lick[to_range] = 1000
    conv_licks = np.convolve(stim_vector_lick, kernel_bi, mode='same') * dt
    extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(dff) - 0.5, -0.5]
    tax2.imshow(conv_licks.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    tax1.imshow(conv_stim.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    Z = linkage(dff, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(dff[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(dff), 1),
                   vmax=np.nanpercentile(np.ravel(dff), 99), extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'Z-score')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    tax1.set_title(rec.filename)
    plt.show()
    return fig


def interactive_heatmap(rec, activity):
    # rec.responsivity()
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    cmap = 'inferno'
    time_range = np.linspace(0, (len(activity[0]) / rec.sf) - 1, len(activity[0]))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax2 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)
    # convolution of the stims
    stim_vector = np.zeros(len(activity[0]))
    stim_vector_lick = np.zeros(len(activity[0]))
    stim_index = [list(range(stim, stim + int(0.5 * rec.sf))) for i, stim in
                  enumerate(rec.stim_time[rec.stim_ampl != 0])]
    for stim_range, stim_amp in zip(stim_index, rec.stim_ampl[rec.stim_ampl != 0]):
        stim_vector[stim_range] = stim_amp
    conv_stim = stim_vector
    lick_index = [list(range(to, to + int(0.5 * rec.sf))) for i, to in
                  enumerate(rec.lick_time[rec.lick_time < len(stim_vector_lick) - 16])]
    for to_range in lick_index:
        stim_vector_lick[to_range] = 2
    conv_licks = stim_vector_lick
    extent = [time_range[0] - dt / 2, time_range[-1] + dt / 2, len(activity) - 0.5, -0.5]
    tax2.imshow(conv_licks.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    tax1.imshow(conv_stim.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    Z = linkage(activity, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(activity[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(activity), 1),
                   vmax=np.nanpercentile(np.ravel(activity), 99), extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'Z-score')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    tax1.set_title(rec.filename)

    durations = np.zeros(len(rec.stim_time), dtype=int)
    for i, timing in enumerate(rec.stim_time):
        if rec.detected_stim[i]:
            durations[i] = np.min(np.array(rec.lick_time - timing)[(rec.lick_time - timing) > 0])
        else:
            durations[i] = (int(0.5 * rec.sf))
    durations[durations > int(0.5 * rec.sf)] = int(0.5 * rec.sf)

    def on_click(event):
        if event.button is MouseButton.RIGHT:
            fig, ax = plt.subplots(1, 1, figsize=(18, 10))
            pre_b = 1
            post_b = 1
            time = np.linspace(-pre_b, post_b, int((pre_b + post_b) * rec.sf))
            # time = np.arange(-pre_b + 1/rec.sf, post_b - 1/rec.sf, 1/rec.sf)
            rand_stim = find_nearest(rec.stim_time, int(event.xdata * rec.sf))
            trace = rec.zscore_exc[dn_exc['leaves'][math.ceil(event.ydata)],
                    rec.stim_time[rand_stim] - round(pre_b * rec.sf):rec.stim_time[rand_stim] + round(post_b * rec.sf)-1]
            ax.plot(time, trace)

            # Get the values of various parameters
            lick_time = (durations[rand_stim]) / rec.sf
            time_lick = np.linspace(0,0.5,16)
            lick_time = time_lick[durations[rand_stim]]
            value_resp = rec.matrices["EXC"]["Responsivity"][dn_exc['leaves'][math.ceil(event.ydata)]][rand_stim]
            value_peak_ind = rec.matrices["EXC"]["Peak_delay"][dn_exc['leaves'][math.ceil(event.ydata)]][rand_stim]
            value_peak_amp = rec.matrices["EXC"]["Peak_amplitude"][dn_exc['leaves'][math.ceil(event.ydata)]][rand_stim]
            value_auc = rec.matrices["EXC"]["AUC"][dn_exc['leaves'][math.ceil(event.ydata)]][rand_stim]

            # Plot the lines
            ax.hlines(0, -1, 2, color="green")
            ax.vlines(0, min(trace), max(trace), lw=2, color="red")
            ax.vlines(lick_time, min(trace), max(trace), lw=3, color="black", linestyles="--")
            ax.vlines(0.5, min(trace), max(trace), lw=2, color="red")

            # Interpolation of values for AUC
            time_inter = np.array([np.linspace(start, end, num=10)[:-1] for start, end in zip(time[:-1], time[1:])]).flatten()
            trace_inter = np.array([np.linspace(start, end, num=10)[:-1] for start, end in zip(trace[:-1], trace[1:])]).flatten()

            # Fill the AUC
            if value_resp > 0:
                ax.fill_between(time_inter, 0, trace_inter, where=(trace_inter > 0) & (time_inter >= 0) & (time_inter <= lick_time), color='red', alpha=0.3)
            elif value_resp < 0:
                ax.fill_between(time_inter, 0, trace_inter, where=(trace_inter < 0) & (time_inter >= 0) & (time_inter <= lick_time), color='red', alpha=0.3)

            # Set the title
            ax.set_title(f"Resp:{value_resp} - Peak n°{value_peak_ind} - amp: {value_peak_amp:.4} - AUC: {value_auc:.4}")

    plt.connect('button_press_event', on_click)
    plt.show()


def amp_tuning_heatmap(ax, rec, activity, title=""):
    cmap = 'inferno'
    amps_reponses = []
    for amp in [2, 4, 6, 8, 10, 12]:
        stims = rec.stim_time[rec.stim_ampl == amp]
        response = activity[:, np.linspace(stims, stims + int(1 * rec.sf), int(1 * rec.sf), dtype=int)]
        responses = response.reshape(len(activity), len(stims) * int(1 * rec.sf))
        response_ = np.mean(responses, axis=1)
        amps_reponses.append(response_)

    tune_act = np.transpose(amps_reponses)
    inter_response = np.array([np.interp(np.linspace(2, 12, 100), [2, 4, 6, 8, 10, 12], resp) for resp in tune_act])
    Z = linkage(inter_response, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(inter_response[dn_exc["leaves"]], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(activity), 1),
                   vmax=np.nanpercentile(np.ravel(activity), 99))
    ax.set_xticks([0, 20, 40, 60, 80, 99])
    ax.set_xticklabels(["2", "4", "6", "8", "10", "12"])
    ax.set_xlabel("Amplitude Stim")
    ax.set_ylabel("Neurons")
    ax.set_title(title)


def ordered_heatmap(rec, exc_neurons=True, inh_neurons=False,
                    time_span="stim", window=0.5, estimator=None,
                    det_sorted=False, amp_sorted=False):

    data, stim_dur = get_zscore(rec, exc_neurons=exc_neurons, inh_neurons=inh_neurons,
                                time_span=time_span, window=window, estimator=estimator,
                                sort=det_sorted, amp_sort=amp_sorted)

    # figure global parameters
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax) if (
                time_span == "stim" or time_span == "pre_stim") else None
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cmap = "inferno"
    extent = [0, data.shape[1], data.shape[0] - 0.5, -0.5]

    # plotting the stimulation amplitudes
    if det_sorted:
        stim_det = []
        stim_undet = []
        for i in range(rec.stim_time.shape[0]):
            if time_span == "stim":
                ampl_vec = [rec.stim_ampl[i]] * int(rec.stim_durations[i])
            elif time_span == "pre_stim":
                ampl_vec = [rec.stim_ampl[i]] * int(window * rec.sf)
            (stim_det if rec.detected_stim[i] else stim_undet).extend(ampl_vec)
        if amp_sorted:
            stim_det.sort()
            stim_undet.sort()
        stim_array = np.array(stim_det + stim_undet)
    else:
        stim_bar = []
        for i in range(rec.stim_time.shape[0]):
            if time_span == "stim":
                stim_bar.extend([rec.stim_ampl[i]] * int(rec.stim_durations[i]))
            elif time_span == "pre_stim":
                stim_bar.extend([rec.stim_ampl[i]] * int(window * rec.sf))
        stim_array = np.array(stim_bar)

    if (time_span == "stim" or time_span == "pre_stim"):
        tax1.imshow(stim_array.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
        tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # neurons clustering and data display
    Z = linkage(data, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(data[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=np.nanpercentile(np.ravel(data), 1),
                   vmax=np.nanpercentile(np.ravel(data), 99), extent=extent)

    # plotting lines to separate stimulation
    if time_span == "stim":
        cumulative_stim_duration = 0
        for stim in stim_dur:
            cumulative_stim_duration += stim
            ax.vlines(cumulative_stim_duration, ymin=-0.5, ymax=len(data) - 0.5, color='w', linewidth=0.5)
        if det_sorted:
            det_stim_duration = rec.stim_durations[rec.detected_stim]
            ax.vlines(det_stim_duration.sum(), ymin=-0.5, ymax=len(data) - 0.5, color='b', linewidth=1)
    elif time_span == "pre_stim":
        for i in range(len(rec.detected_stim)):
            ax.vlines(i * int(window * rec.sf), ymin=-0.5, ymax=len(data) - 0.5, color='w', linewidth=0.5)
        if det_sorted:
            ax.vlines(rec.detected_stim.sum() * int(window * rec.sf), ymin=-0.5, ymax=len(data) - 0.5, color='b',
                      linewidth=1)
    else:
        iterator = len(rec.reward_time) if time_span == "reward" else (
            len(rec.timeout_time) if time_span == "timeout" else 0)
        for i in range(iterator):
            ax.vlines(i * int(window * rec.sf), ymin=-0.5, ymax=len(data) - 0.5, color='w', linewidth=0.5)

    # color scale parameters
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label("Z-score") if estimator is None else cbar.set_label(f"Z-score ({estimator})")

    ax.set_ylabel('Neurons')
    ax.set_xlabel(f"Frames ({time_span})")
    tax1.set_title(f"{rec.filename} ({rec.genotype}) - {rec.threshold}") if (
                time_span == "stim" or time_span == "pre_stim") else ax.set_title(
        f"{rec.filename} ({rec.genotype}) - {rec.threshold}")
    plt.tight_layout()
    plt.show()


def resp_heatmap(rec, n_type="EXC"):

    data = rec.matrices[n_type]["Responsivity"]

    det_ampl = rec.stim_ampl[rec.detected_stim]
    undet_ampl = rec.stim_ampl[np.invert(rec.detected_stim)]
    det_resp = data[:, rec.detected_stim]
    undet_resp = data[:, np.invert(rec.detected_stim)]

    stim_array = np.array(sorted(det_ampl) + sorted(undet_ampl))

    ordered_data = np.empty((data.shape[0], 0))
    for amp in sorted(set(det_ampl)):
        for index, stim_ampl in enumerate(det_ampl):
            if stim_ampl == amp:
                ordered_data = np.column_stack((ordered_data, det_resp[:, index]))
    for amp in sorted(set(undet_ampl)):
        for index, stim_ampl in enumerate(undet_ampl):
            if stim_ampl == amp:
                ordered_data = np.column_stack((ordered_data, undet_resp[:, index]))

    # figure global parameters
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    divider = make_axes_locatable(ax)
    tax1 = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cmap = "inferno"
    extent = [0, data.shape[1], data.shape[0] - 0.5, -0.5]


    tax1.imshow(stim_array.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


    Z = linkage(ordered_data, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    im = ax.imshow(ordered_data[dn_exc['leaves']], cmap=cmap, interpolation='none', aspect='auto',
                   vmin=-1,
                   vmax=1, extent=extent)

    # color scale parameters
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label("Responsivity")
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Trials")
    tax1.set_title(f"{rec.filename} ({rec.genotype}) - {rec.threshold}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Record import
    plt.ion()
    roi_path = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/FmKO_ROIs&inhibitory.xlsx"

    plot_all_records = True
    plot_ordered_heatmap = True
    plot_responsivity_heatmap = True

    if plot_all_records:
        directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
        files = os.listdir(directory)
        files_ = [file for file in files if file.endswith("synchro")]
        for file in files_:
            folder = f"/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/{file}/"
            rec = RecordingAmplDet(folder, 0, roi_path, cache=True)
            if plot_ordered_heatmap:
                ordered_heatmap(rec, exc_neurons=True, inh_neurons=False,
                                time_span="pre_stim", window=0.5, estimator=None,
                                det_sorted=True, amp_sorted=True)
            if plot_responsivity_heatmap:
                resp_heatmap(rec, n_type="EXC")

    else:
        directory = "C:/Users/cvandromme/Desktop/Data/20220715_4456_00_synchro/"
        rec = RecordingAmplDet(directory, 0, roi_path, cache=True)
        if plot_ordered_heatmap:
            ordered_heatmap(rec, exc_neurons=True, inh_neurons=False,
                            time_span="stim", window=0.5, estimator=None,
                            det_sorted=True, amp_sorted=True)
        if plot_responsivity_heatmap:
            resp_heatmap(rec, n_type="EXC")


    # hm.plot_dff_stim_detected(rec, rec.df_f_exc)
    # hm.plot_dff_stim_detected(rec, rec.df_f_inh)
    # hm.intereactive_heatmap(rec, rec.zscore_exc)
    # hm.intereactive_heatmap(rec, rec.zscore_inh)