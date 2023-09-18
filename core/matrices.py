"""Théo Gauvrit, 05/09/2023
Matrices neurones x events(stim, reward, timeout)"""

import matplotlib
import numpy as np
import pandas as pd
import core as pc
import os
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")


directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format/"
roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
matrices = {"EXC": {"Responsivity": [], "AUC": [], "Delay_onset": []},
            "INH": {"Responsivity": [], "AUC": [], "Delay_onset": []}}

delay_per_mouse = pd.DataFrame({"Filename": [], "Response delay EXC (frames)": [], "Response delay INH (frames)": []})
for folder in os.listdir(directory):
    if os.path.isdir(directory + folder):
        path = directory + folder + '/'
        print(folder)
        recording = pc.RecordingAmplDet(path, 0, folder, roi_info)
        matrice_resp_exc = recording.resp_matrice(recording.df_f_exc)
        matrice_resp_inh = recording.resp_matrice(recording.df_f_inh)
        matrices["EXC"]["Responsivity"].append(matrice_resp_exc)
        matrices["INH"]["Responsivity"].append(matrice_resp_inh)
        matrice_delay_exc = recording.delay_matrice(recording.df_f_exc, matrice_resp_exc)
        matrice_delay_inh = recording.delay_matrice(recording.df_f_inh, matrice_resp_inh)
        matrices["EXC"]["Delay_onset"].append(matrice_delay_exc)
        matrices["INH"]["Delay_onset"].append(matrice_delay_inh)
        delay_per_mouse = delay_per_mouse.append({"Filename": folder,
                                                  "Response delay EXC (frames)": np.nanmean(matrice_delay_exc),
                                                  "Response delay INH (frames)": np.nanmean(matrice_delay_inh)},
                                                 ignore_index=True)
        # matrices["AUC"].append(recording.auc_matrice(timings()))

delay_per_mouse.to_csv("delay_output.csv")

