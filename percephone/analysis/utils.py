"""
Théo Gauvrit 22/01/2024
Utility functions for analysis
"""


import matplotlib
import numpy as np


matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Qt5Agg")


def kernel_biexp(sf):
    """
    Generate kernel of a biexponential function for mlr analysis or onset delay analysis
    Parameters
    ----------
    sf: float
        sampling frequency of the recording

    Returns
    -------
    kernel_bi: array
        kernel of the biexponential function

    """
    tau_r = 0.07  # s0.014
    tau_d = 0.236  # s
    kernel_size = 10  # size of the kernel in units of tau
    a = 5  # scaling factor for biexpo kernel
    dt = 1 / sf  # spacing between successive timepoints
    n_points = int(kernel_size * tau_d / dt)
    kernel_times = np.linspace(-n_points * dt,
                               n_points * dt,
                               2 * n_points + 1)  # linearly spaced array from -n_pts*dt to n_pts*dt with spacing dt
    kernel_bi = a * (1 - np.exp(-kernel_times / tau_r)) * np.exp(-kernel_times / tau_d)
    kernel_bi[kernel_times < 0] = 0  # set to zero for negative times
    # fig, ax = plt.subplots()
    # ax.plot(kernel_times, kernel_rise)
    # ax.set_xlabel('time (s)')
    # plt.show()
    return kernel_bi


