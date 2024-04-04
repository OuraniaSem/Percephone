"""
Théo Gauvrit 18/01/2024
New inferno traces colored heatmaps
"""

import matplotlib
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def plot_dff_stim_detected(rec, dff, filename):
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
    tax1.set_title(filename)
    plt.show()


def plot_dff_stim_detected_timeout(rec, dff, filename):
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
    tax1.set_title(filename)
    plt.show()
    return fig


def plot_dff_stim_detected_lick(rec, dff, filename):
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
    tax1.set_title(filename)
    plt.show()
    return fig


def intereactive_heatmap(rec, activity):
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
            pre_b = 5
            post_b = 5
            time = np.linspace(-pre_b, post_b, int((pre_b + post_b) * rec.sf) - 1)
            rand_stim = find_nearest(rec.stim_time, int(event.xdata * rec.sf))
            trace = rec.zscore_exc[dn_exc['leaves'][math.ceil(event.ydata)],
                    rec.stim_time[rand_stim] - int(pre_b * rec.sf):rec.stim_time[rand_stim] + int(
                        post_b * rec.sf)]
            ax.plot(time, trace)
            # onset_line = rec.matrices["EXC"]["Delay_onset"][rand_neu][rand_stim]
            ax.hlines(0, -1, 2)
            ax.set_title("Resp: " + str(
                rec.matrices["EXC"]["Responsivity"][dn_exc['leaves'][math.ceil(event.ydata)]][
                    rand_stim]) + " - Peak: " + str(
                rec.matrices["EXC"]["Peak_delay"][dn_exc['leaves'][math.ceil(event.ydata)]][
                    rand_stim]))  # +      "Onset: " + str(rec.matrices["EXC"]["Delay_onset"][rand_neu][rand_stim]) + "  " + "AUC: " + str(rec.matrices["EXC"]["AUC"][rand_neu][rand_stim]
            ax.vlines(0, min(trace), max(trace), lw=2, color="red")
            ax.vlines(durations[rand_stim] / rec.sf, min(trace), max(trace), lw=2, color="black", linestyles="--")
            ax.vlines(0.5, min(trace), max(trace), lw=2, color="red")

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

def responsivity(heatmap):

    ax