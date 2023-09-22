"""Théo Gauvrit 09/06/2023
Main script to use the functions and classes defined in core and plots
"""
import os
import time as time
import matplotlib
import numpy as np
import pandas as pd
import core as pc
import plots as p
from scipy.cluster.hierarchy import dendrogram, linkage

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Qt5Agg")
directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")

start_time = time.time()
summary_resp = pd.DataFrame()
neurons_resp = pd.DataFrame()
traces_wt = {"EXC": {"Amp 2µm": [], "Amp 4µm": [], "Amp 6µm": [], "Amp 8µm": [], "Amp 10µm": [], "Amp 12µm": [],
                     "Reward event": [], "Timeout event": []},
             "INH": {"Amp 2µm": [], "Amp 4µm": [], "Amp 6µm": [], "Amp 8µm": [], "Amp 10µm": [], "Amp 12µm": [],
                     "Reward event": [], "Timeout event": []}}
traces_ko = {"EXC": {"Amp 2µm": [], "Amp 4µm": [], "Amp 6µm": [], "Amp 8µm": [], "Amp 10µm": [], "Amp 12µm": [],
                     "Reward event": [], "Timeout event": []},
             "INH": {"Amp 2µm": [], "Amp 4µm": [], "Amp 6µm": [], "Amp 8µm": [], "Amp 10µm": [], "Amp 12µm": [],
                     "Reward event": [], "Timeout event": []}}
for folder in os.listdir(directory):
    if os.path.isdir(directory + folder):
        path = directory + folder + '/'
        name = int(folder[9:13])
        n_record = folder[14:16]
        print(name)
        row = roi_info[(roi_info["Number"] == name) & (roi_info["Recording number"] == int(n_record))]
        recording = pc.RecordingAmplDet(path, 0, folder, roi_info)
        #  responsivity
        # sum_resp, neur_resp = recording.compute_responsivity(row)
        # summary_resp = pd.concat([summary_resp, sum_resp])
        # neurons_resp = pd.concat([neurons_resp, neur_resp])
        # summary_resp.to_csv("global_responsiveness_test.csv")
        # neurons_resp.to_csv("neurons_resp_test.csv")
        #  heatmaps
        events_timings = [recording.stim_time[recording.stim_ampl == amp] for amp in [2, 4, 6, 8, 10, 12]]
        events_timings.append(recording.reward_time)
        events_timings.append(recording.timeout_time)

        for timings, name in zip(events_timings,
                                 ["Amp 2µm", "Amp 4µm", "Amp 6µm", "Amp 8µm", "Amp 10µm", "Amp 12µm", "Reward event",
                                  "Timeout event"]):
            peri_exc = p.perirevent(timings, recording.df_f_exc, recording.sf)
            peri_inh = p.perirevent(timings, recording.df_f_inh, recording.sf)
            if not np.any(peri_exc):
                continue
            if row["Genotype"].values == "WT":
                traces_wt['EXC'][name].append(peri_exc)
                traces_wt['INH'][name].append(peri_inh)
            if row["Genotype"].values == "KO":
                traces_ko['EXC'][name].append(peri_exc)
                traces_ko['INH'][name].append(peri_inh)


def cluster_heatmap(exc, inh):
    Z = linkage(exc, 'ward', optimal_ordering=True)
    dn_exc = dendrogram(Z, no_plot=True, count_sort="ascending")
    Z = linkage(inh, 'ward', optimal_ordering=True)
    dn_inh = dendrogram(Z, no_plot=True, count_sort="ascending")
    return dn_exc, dn_inh


dn_ex_wt, dn_in_wt = cluster_heatmap(np.concatenate(traces_wt['EXC']["Amp 12µm"]),
                                     np.concatenate(traces_wt['INH']["Amp 12µm"]))
dn_ex_ko, dn_in_ko = cluster_heatmap(np.concatenate(traces_ko['EXC']["Amp 12µm"]),
                                     np.concatenate(traces_ko['INH']["Amp 12µm"]))
for event_type in traces_wt['EXC'].keys():
    # p.group_heat_map_per_stim(np.concatenate(traces_wt['EXC'][event_type]),
    #                           np.concatenate(traces_wt['INH'][event_type]),
    #                           dn_ex_wt, dn_in_wt, str(event_type),
    #                           filename="../output/Heat_map_" + str(event_type) + "_WT.png")
    # p.group_heat_map_per_stim(np.concatenate(traces_ko['EXC'][event_type]),
    #                           np.concatenate(traces_ko['INH'][event_type]),
    #                           dn_ex_ko, dn_in_ko, str(event_type),
    #                           filename="../output/Heat_map_" + str(event_type) + "_KO.png")
    p.group_heat_map_per_stim_split(np.concatenate(traces_wt['EXC'][event_type]),
                                    dn_ex_wt, str(event_type),
                                    filename="../output/Heat_map_" + str(event_type) + "_WT.png",
                                    legend="Exc neurons")
    p.group_heat_map_per_stim_split(np.concatenate(traces_ko['EXC'][event_type]),
                                    dn_ex_ko, str(event_type),
                                    filename="../output/Heat_map_" + str(event_type) + "_KO.png",
                                    legend="Exc neurons")
    p.group_heat_map_per_stim_split(np.concatenate(traces_wt['INH'][event_type]), dn_in_wt, str(event_type),
                                    filename="../output/Heat_map_" + str(event_type) + "INH_WT.png",
                                    legend="Inh neurons")
    p.group_heat_map_per_stim_split(np.concatenate(traces_ko['INH'][event_type]), dn_in_ko, str(event_type),
                                    filename="../output/Heat_map_" + str(event_type) + "INH_KO.png",
                                    legend="Inh neurons")

print("--- %s seconds ---" % (time.time() - start_time))

##### TEMP check heatmap for each file independantly

# directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
# roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
# files = os.listdir(directory)
# files = ["20221008_4746_00_synchro_sigma2_acuFR_tau02"]
# for folder in files:
#     if os.path.isdir(directory + folder):
#         print(folder)
#         path = directory + folder + '/'
#         recording = pc.RecordingAmplDet(path, 0, folder, roi_info)
#         # recording = pc.RecordingStimulusOnly(path, 0, folder, roi_info, correction=False)
#         events_timings = [recording.stim_time[recording.stim_ampl == amp] for amp in [2, 4, 6, 8, 10, 12]]
#         events_timings.append(recording.reward_time)
#         events_timings.append(recording.timeout_time)
#         from scipy.cluster.hierarchy import dendrogram, linkage
#
#         peri_exc = p.perirevent(recording.stim_time[recording.stim_ampl == 10], recording.df_f_exc, recording.sf)
#         peri_inh = p.perirevent(recording.stim_time[recording.stim_ampl == 10], recording.df_f_inh, recording.sf)
#         Z = linkage(peri_exc, 'ward', optimal_ordering=True)
#         dn_exc = dendrogram(Z, no_plot=True)
#         Z = linkage(peri_inh, 'ward', optimal_ordering=True)
#         dn_inh = dendrogram(Z, no_plot=True)
#         for timings, name in zip(events_timings, [2, 4, 6, 8, 10, 12, "reward", "timeout"]):
#             if len(timings) == 0:
#                 continue
#             peri_exc = p.perirevent(timings, recording.df_f_exc, recording.sf)
#             peri_inh = p.perirevent(timings, recording.df_f_inh, recording.sf)
#
#             p.group_heat_map_per_stim(peri_exc,
#                                       peri_inh,
#                                       dn_exc,  dn_inh, str(name) + " " + str(folder),
#                                       filename="../output/tau_02/Heat_map_" + str(name) + "_" + str(folder) + "_KO.png")
#
