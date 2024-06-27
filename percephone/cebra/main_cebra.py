"""" Theo Gauvrit
05/06/2023
First test of Cebra library"""
import os

import cebra
import json
import numpy as np
import pandas as pd
import percephone.core.recording as pc
import os
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, pool
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.switch_backend("Qt5Agg")
matplotlib.use("Qt5Agg")
matplotlib.use('Qt5Agg')
plt.switch_backend('Qt5Agg')

#rnd.seed(20552)
directory = "C:\\Users\\acorniere\\Desktop\\percephone_data\\"
roi_info = pd.read_excel(directory + "\\FmKO_ROIs&inhibitory.xlsx")
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]
def opening_rec(fil,i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0, fil, roi_info)
    return rec

workers = cpu_count()
pool = pool.ThreadPool(processes=workers)
async_results = [pool.apply_async(opening_rec, args=(file,i)) for i,file in enumerate(files_)]
recs = [ar.get() for ar in async_results]

for rec in recs:
        neural_data = np.transpose(rec.zscore_exc)
        stim_time = rec.stim_time
        stim_ampl = rec.stim_ampl
        reward_time = rec.reward_time
        timeout_time = rec.timeout_time
        detected_stim = rec.detected_stim
        lick_time = rec.lick_time
        discrete_label = np.zeros(len(neural_data))
        index_stim = np.concatenate([list(np.arange(i, i+17)) for i in stim_time[detected_stim]])
        index_stim_undet = np.concatenate([list(np.arange(i, i+17)) for i in stim_time[~detected_stim]])
        # index_rew = np.concatenate([list(np.arange(i, i+10)) for i in reward_time])
        # index_timeout = np.concatenate([list(np.arange(i, i+40)) for i in timeout_time])
        index_lick = np.concatenate([list(np.arange(i, i+15)) for i in lick_time])
        discrete_label[index_stim[index_stim < len(discrete_label)]] = 1
        discrete_label[index_stim_undet[index_stim_undet < len(discrete_label)]] = 2
        # discrete_label[index_rew[index_rew < len(discrete_label)]] = 3
        # discrete_label[index_timeout[index_timeout < len(discrete_label)]] = 4
        discrete_label[index_lick[index_lick < len(discrete_label)]] = 5
        cebra_model = cebra.CEBRA(
            model_architecture="offset1-model",
            batch_size=1024,
            temperature_mode="auto",
            learning_rate=0.001,
            max_iterations=10000,
            time_offsets=10,
            output_dimension=3,
            device="cuda_if_available",  
            verbose=True
        )
        cebra_model.fit(neural_data, discrete_label)
        embedding = cebra_model.transform(neural_data)
        color_labels = np.full(len(discrete_label), 'c')
        color_labels[index_stim[index_stim < len(discrete_label)]] = 'b'
        color_labels[index_stim_undet[index_stim_undet < len(discrete_label)]] = 'k'
        # color_labels[index_rew[index_rew < len(discrete_label)]] = 'r'
        # color_labels[index_timeout[index_timeout < len(discrete_label)]] = 'y'
        color_labels[index_lick[index_lick < len(discrete_label)]] = "grey"
        cebra.plot_embedding(embedding, embedding_labels=color_labels, markersize=5, idx_order=(0, 1, 2), title=rec.filename)
        ax = cebra.plot_loss(cebra_model)
        ax.set_title(rec.filename)
