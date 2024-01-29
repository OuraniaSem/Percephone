"""Théo Gauvrit, 15/09/2023
utility functions for input/output management
"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def read_info(folder_name, rois):
    """ Extract inhibitory ids and frame rate from rois_info Excel sheet
    with the folder name

    Parameters
    ----------
    folder_name :  str
        name of the folder (ex:"20220728_4454_00_synchro")
    rois: pd.Dataframe
        metadata for each file. Need a manually added column "Recording number"
    """
    name = int(folder_name[9:13])
    n_record = folder_name[14:16]
    date = str(folder_name[:4]) + "-" + str(folder_name[4:6]) + "-" + str(folder_name[6:8])
    row = rois[(rois["Number"] == name) &
               (rois["Recording number"] == int(n_record)) & (rois["Date"] == pd.to_datetime(date))]
    inhibitory_ids = np.array(list(list(row["Inhibitory neurons: ROIs"])[0].split(", ")))
    return (row["Number"].values[0],
            inhibitory_ids.astype(int),
            row["Frame Rate (Hz)"].values[0], row["Genotype"].values[0])


def extract_analog_from_mesc(path_mesc, tuple_mesc, frame_rate, savepath=""):
    """
    Extract analog from mesc file for ITI curve. Save it as analog.txt in order to be used by percephone
    Parameters
    ----------
    path_mesc:
        path to mesc file
    tuple_mesc: tuple
        (a,b) where "a" is the session number and "b" is the unit number from femtonics software
    savepath:
        path where to save the analog.txt
    """
    print("Analog signal extraction from .mesc file.")
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

