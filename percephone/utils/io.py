"""Théo Gauvrit, 15/09/2023
utility functions for input/output management
"""
import numpy as np
import pandas as pd


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
