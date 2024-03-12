"""Théo Gauvrit 01/03/2024
Main script to test changes in functions or classe
"""
import numpy as np
import pandas as pd
import percephone.core.recording as pc
import os
import percephone.plts.behavior as pbh
import matplotlib
import percephone.plts.stats as ppt
import matplotlib.pyplot as plt
import percephone.analysis.mlr_models as mlr_m
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")


directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
files = os.listdir(directory)
amp = 10
files_ = [file for file in files if file.endswith("synchro")]
y, i = 0, 0
amps = [2, 6, 4, 4, 4, 8, 4, 4, 12, 8, 6, 12, 12]
recs = []
# fig, ax = plt.subplots(4, 7, figsize=(25, 14))
for file, amp in zip(files_, amps):
    if os.path.isdir(directory + file):
        print(file)
        rec = pc.RecordingAmplDet(directory + file + "/", 0, file, roi_info)
        print("MLR")
        # mlr_model, model_name = mlr_m.precise_stim_model(rec)
        # rec.mlr(mlr_model, model_name)
        print("baseline analysis")
        recs.append(rec)
        rec.responsivity()

#         neurons_activated = rec.matrices["EXC"]["Responsivity"]
#         trace = rec.df_f_exc[neurons_activated]
#         #  get all the bsl before stim of the corresponding amp stim
#         stims_det = rec.stim_time[rec.detected_stim & (rec.stim_ampl == amp)]
#         bsl = trace[:, np.linspace(stims_det - int(1 * rec.sf), stims_det, int(1 * rec.sf), dtype=int)]
#         bsl_ = bsl.reshape(len(trace), len(stims_det) * int(1 * rec.sf))
#         det_bsl = np.mean(bsl_, axis=1)
#         stims_undet = rec.stim_time[~rec.detected_stim & (rec.stim_ampl == amp)]
#         bsl = trace[:, np.linspace(stims_undet - int(1 * rec.sf), stims_undet, int(1 * rec.sf), dtype=int)]
#         bsl_ = bsl.reshape(len(trace), len(stims_undet) * int(1 * rec.sf))
#         undet_bsl = np.mean(bsl_, axis=1)
#         if rec.genotype == "WT":
#             pbh.psycho_like_plot(rec, roi_info, ax[0, i])
#             ppt.paired_boxplot(ax[1, i], det_bsl, undet_bsl, " std baseline", "AMP: " + str(amp) + " " + str(rec.filename))
#             i = i + 1
#         else:
#             pbh.psycho_like_plot(rec, roi_info, ax[2, y])
#             ppt.paired_boxplot(ax[3, y], det_bsl, undet_bsl, "std baseline", "AMP: " + str(amp) + " " + str(rec.filename))
#             y = y + 1
#
# np.ravel(ax)[-1].set_axis_off()
# ax[2, 6].set_axis_off()
