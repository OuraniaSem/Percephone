"""
Théo Gauvrit 18/01/2024
New inferno traces colored heatmaps
"""

import matplotlib
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.switch_backend("Qt5Agg")
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
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