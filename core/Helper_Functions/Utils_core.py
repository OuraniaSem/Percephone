"""Théo Gauvrit, 15/09/2023
Regroup small utils functions used in the cores function of Percephone
"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

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
    date = str(foldername[:4]) + "-" + str(foldername[4:6]) + "-" + str(foldername[6:8])
    row = rois[(rois["Number"] == name) & (rois["Recording number"] == int(n_record)) & (rois["Date"] == pd.to_datetime(date))]
    inhib_ids = np.array(list(list(row["Inhibitory neurons: ROIs"])[0].split(", ")))
    return row["Number"].values[0], inhib_ids.astype(int), row["Frame Rate (Hz)"].values[0], row["Genotype"].values[0]


def extract_analog_from_mesc(path_mesc, tuple_mesc, frame_rate, savepath=""):
    """
    Extract analog from mesc file for ITI curve. Save it as analog.txt in order to be use by percephone
    Parameters
    ----------
    path_mesc:
        path to mesc file
    tuple_mesc: tuple
        (a,b) where "a" is the session number and "b" is the unit number from femtonics software
    savepath:
        path where to save the analog.txt
    """
    file = h5py.File(path_mesc)
    dset = file['MSession_' + str(tuple_mesc[0])]
    unit = dset['MUnit_' + str(tuple_mesc[1])]
    iti = unit['Curve_2']
    iti_curve = np.array(iti['CurveDataYRawData'])
    timings = unit['Curve_0']
    timing_curve = np.array(timings['CurveDataYRawData'])

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(iti_curve)
    ax.set_title("Check if it look like ITI curve!")
    plt.show()
    end_timings = timing_curve[-1] * np.array(timings.attrs.get("CurveDataYConversionConversionLinearScale"))
    print(end_timings)
    end_timings_frames = len(timing_curve)*frame_rate
    print(end_timings_frames)
    end_timings_iti = len(iti_curve[::2])/10
    print(end_timings_iti)
    nb_points = int(len(iti_curve[::2]))  # int(end_timings_frames*10)  #int(end_timings*10)
    timings = np.linspace(0,  end_timings_frames, nb_points)
    analog_np = np.zeros((4, nb_points))
    analog_np[0] = timings
    analog_np[1] = analog_np[1]  # no stim analog in the new format
    analog_np[2] = timings
    iti_curve_ = iti_curve[::2]
    analog_np[3] = iti_curve_[:nb_points]
    analog_t = np.transpose(analog_np)
    np.savetxt(savepath + 'analog.txt', analog_t, fmt='%.8g', delimiter="\t")
    print("Analog saved.")


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


if __name__ == '__main__':
    path = 'D:\\Ca imaging\\Analysis_Dec2023\\'
    mesc = '20231007_5879_det.mesc'
    extract_analog_from_mesc(path + '\\'+mesc, (0 ,0), 31.2663, savepath=path +'20231007_5879_det_00_synchro\\' )