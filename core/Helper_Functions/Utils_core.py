"""Théo Gauvrit, 15/09/2023
Regroup small utils functions used in the cores function of Percephone
"""
import numpy as np


def read_info(foldername, rois):
    """ Extract inhbitory ids and frame rate from rois_info excel sheet
    with the foldername

    Parameters
    ----------
    foldername :  str
        name of the folder (ex:"20220728_4454_00_synchro")
    rois: pd.Dataframe
        metadata for each file. Need a manually added column "Recording number"
    """
    name = int(foldername[9:13])
    n_record = foldername[14:16]
    row = rois[(rois["Number"] == name) & (rois["Recording number"] == int(n_record))]
    inhib_ids = np.array(list(list(row["Inhibitory neurons: ROIs"])[0].split(", ")))
    return row["Number"].values[0], inhib_ids.astype(int), row["Frame Rate (Hz)"].values[0], row["Genotype"].values[0]


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
