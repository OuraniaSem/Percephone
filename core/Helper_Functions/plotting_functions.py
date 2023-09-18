import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_dff_raster(signal, times, indices, dff):
    """Plot dff of neurons with certain indices in a raster plot, also plot signal for comparison.

    Parameters :
    ------------
    signal : 1d array
        external signal, shape (n_timepoints)
        
    times : 1d array
        timepoints of signal and dff, shape (n_timepoints)

    dff : 2d array
        rescaled fluorescence traces for all neurons, shape (n_neurons,n_timepoints)
    
    indices : 1d array
        indices of the neurons that can be selected with the slider
    """
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    tax = divider.append_axes('top', size='10%', pad=0.1, sharex=ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    ax.tick_params(which='both', width=4)

    dt = np.mean(np.diff(times))
    extent = [times[0] - dt / 2, times[-1] + dt / 2, len(indices) - 0.5, -0.5]
    cmap = 'inferno'

    tax.imshow(signal.reshape(1, -1), cmap=cmap, aspect='auto', interpolation='none', extent=extent)
    tax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    # tax.set_title('signal')

    im = ax.imshow(dff[indices], cmap=cmap, interpolation='none', aspect='auto', extent=extent)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(which='both', width=4)
    cbar.set_label(r'$\Delta F/F$')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('neurons')
    plt.show()
