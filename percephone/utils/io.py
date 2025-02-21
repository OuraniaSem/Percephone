"""Théo Gauvrit, 15/09/2023
utility functions for input/output management
"""
import os

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, pool
# import percephone.core.recording as pc


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
    inhibitory_ids = eval(f"[{str(row["Inhibitory neurons: ROIs"].values[0])}]")
    hit_rates = [float(x) for x in row["Stimulus detection"].values[0].split(',')]
    return (row["Number"].values[0],
            inhibitory_ids,
            row["Frame Rate (Hz)"].values[0], row["Genotype"].values[0], row["Threshold"].values[0], row["ITI1 ONLY"].values[0], hit_rates)


def extract_analog_from_mesc(path_mesc, tuple_mesc, frame_rate, analog_fs=20000, savepath=""):
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
    factor = int(analog_fs /10000)
    print("Analog signal extraction from .mesc file.")
    file = h5py.File(path_mesc)
    dset = file['MSession_' + str(tuple_mesc[0])]
    unit = dset['MUnit_' + str(tuple_mesc[1])]
    iti = unit['Curve_3']  # 3 in general   # 4 for after 01-2024
    iti_curve = np.array(iti['CurveDataYRawData'])
    timings = unit['Curve_1']  # 1 or 0
    timing_curve = np.array(timings['CurveDataYRawData'])
    # if timings.attrs.get("CurveDataYConversionType") == 2:
    #     refvalues = np.array(timings.attrs.get("CurveDataYConversionReferenceValues"))
    #     xp = refvalues[::2]
    #     yp = refvalues[1::2]
    #     timing_curve = np.interp(timing_curve, xp, yp)
    #     print(timing_curve)
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(iti_curve )
    ax.set_title("Check if its look like ITI curve!")
    plt.show()
    end_timings = timing_curve[-1] * np.array(timings.attrs.get("CurveDataYConversionConversionLinearScale"))
    start_timings = timing_curve[0] * np.array(timings.attrs.get("CurveDataYConversionConversionLinearScale"))
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    diff_frames = np.diff(timing_curve * np.array(timings.attrs.get("CurveDataYConversionConversionLinearScale")))
    ax.plot(diff_frames)
    latency = np.sum(np.subtract(diff_frames, (1/frame_rate)*1000))
    ax.set_title(f"Check lost frames!  Total latency: {latency:.2f}")
    plt.show()
    print(f"Start timings: {start_timings }")
    print(f"end_timings {end_timings}")
    end_timings_frames = len(timing_curve)*((1/frame_rate)*1000)
    print(f"nb frames in mesc: {len(timing_curve)}")
    print(f"end timing frames: {end_timings_frames}")
    end_timings_iti = len(iti_curve[::factor])/10
    print(f"end timing iti {end_timings_iti}")
    nb_points = int(len(iti_curve[::factor]))  # int(end_timings_frames*10)  #  #int(end_timings*10)
    timings_c = np.linspace(0, end_timings_frames, nb_points)
    analog_np = np.zeros((4, nb_points))
    analog_np[0] = timings_c
    analog_np[1] = analog_np[1]  # no stim analog in the new format
    analog_np[2] = timings_c
    iti_curve_ = iti_curve[::factor]
    analog_np[3] = iti_curve_[:nb_points]
    analog_t = np.transpose(analog_np)
    np.savetxt(savepath + 'analog.txt', analog_t, fmt='%.8g', delimiter="\t")
    np.save(savepath + 'timestamps_frames.npy', timing_curve * np.array(timings.attrs.get("CurveDataYConversionConversionLinearScale")))
    print(f"len analog : {analog_np.shape}")
    print(f"last analog : {analog_np[:,-1]}")
    print("Analog saved.")


def correction_drift_fluo(df_f, path):
    corrected_df_f = np.zeros(df_f.shape)
    for i,neuron_trace in enumerate(df_f):
        start_fluo = np.mean(neuron_trace[:30])
        end_fluo = np.mean(neuron_trace[-30:])
        drift = np.linspace(0, 1.5*(start_fluo - end_fluo), len(neuron_trace))
        corrected_df_f[i] = neuron_trace + drift
        np.save(path, corrected_df_f)
    return corrected_df_f


def get_idx_frame_mesc(time_ms, timestamps):
    idx_frame = (np.abs(timestamps - (time_ms/10))).argmin()
    return idx_frame


# def get_recs_dict(user, cache=True, BMS=False):
#
#     workers = cpu_count()
#
#     if user == "Célien":
#         pool_ = pool.ThreadPool(processes=workers)
#         if BMS:
#             directory = "C:/Users/cvandromme/Desktop/Data_DMSO_BMS/"
#             roi_path = "C:/Users/cvandromme/Desktop/Fmko_bms&dmso_info.xlsx"
#         else:
#             directory = "C:/Users/cvandromme/Desktop/Data/"
#             roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
#
#     elif user == "Théo":
#         pool_ = Pool(processes=workers)
#         directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
#         roi_path = directory + "/FmKO_ROIs&inhibitory.xlsx"
#
#     files = os.listdir(directory)
#     files_ = [file for file in files if file.endswith("synchro")]
#
#     def opening_rec(fil, i):
#         print(fil)
#         rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path, cache=cache)
#         return rec
#
#     async_results = [pool_.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
#     if BMS:
#         recs = {str(ar.get().filename) + ar.get().genotype: ar.get() for ar in async_results}
#     else:
#         recs = {str(ar.get().filename): ar.get() for ar in async_results}
#
#     return recs


def get_server_address(user):
    if user == "Célien":
        server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper/"
    elif user == "Théo":
        server_address = "/run/user/1004/gvfs/smb-share:server=engram.local,share=data/Current_members/Ourania_Semelidou/2p/Figures_paper/"
    return server_address




if __name__ == '__main__':
    import percephone.plts.heatmap as hm

    path = "C:/Users/cvandromme/Desktop/Data/"
    roi_info = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    # folder = "20240404_6601_04_synchro_temp"
    # folder = "20240404_6602_01_synchro_temp"
    # path_to_mesc = path + "/20240404_6602_det.mesc"
    folder = "20240501_6611_01_detBMS_synchro"
    path_to_mesc = path + "20231009_5896_det.mesc"

    # extract_analog_from_mesc(path_to_mesc, (0, 4), 30.9609, 20000, path + folder + "/")
    rec = pc.RecordingAmplDet(path + folder + "/", 0, roi_info,  cache=False, correction=False)
    hm.interactive_heatmap(rec, rec.zscore_exc)

