import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA
from multiprocessing import Pool, cpu_count, pool
import percephone.core.recording as pc
from percephone.analysis.utils import get_zscore


def get_recs_dict(user, cache=True, BMS=False):

    workers = cpu_count()

    if user == "Célien":
        pool_ = pool.ThreadPool(processes=workers)
        if BMS:
            directory = "C:/Users/cvandromme/Desktop/Data_DMSO_BMS/"
            roi_path = "C:/Users/cvandromme/Desktop/Fmko_bms&dmso_info.xlsx"
        else:
            directory = "C:/Users/cvandromme/Desktop/Data/"
            roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"

    elif user == "Théo":
        pool_ = Pool(processes=workers)
        directory = "/datas/Théo/Projects/Percephone/data/Amplitude_Detection/loop_format_tau_02/"
        roi_path = directory + "/FmKO_ROIs&inhibitory.xlsx"

    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]

    def opening_rec(fil, i):
        print(fil)
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path, cache=cache)
        return rec

    async_results = [pool_.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    if BMS:
        recs = {str(ar.get().filename) + ar.get().genotype: ar.get() for ar in async_results}
    else:
        recs = {int(ar.get().filename): ar.get() for ar in async_results}

    return recs


if __name__ == '__main__':
    recs = get_recs_dict("Célien", cache=True, BMS=False)
    rec = recs[4456]
    frame_per_stim = 30
    zscore = get_zscore(rec, exc_neurons=True, inh_neurons=True, time_span="prestim_fixed_stim", window=0.5, estimator=None, sort=False,
               amp_sort=False, single_frame_estimator=False)[0].T
    nb_stim = rec.detected_stim.shape
    nb_hit = rec.detected_stim.sum()
    nb_miss = rec.detected_stim.sum()
    label = np.repeat(rec.detected_stim, frame_per_stim).astype(int)
    assert zscore.shape[0] == label.shape[0]
    print(zscore.shape)
    print(label.shape)
    print(set(label))

    monkey_target = cebra.datasets.init('area2-bump-target-active')
    max_iterations = 1000

    cebra_target_model = CEBRA(model_architecture='offset10-model',
                               batch_size=512,
                               learning_rate=0.0001,
                               temperature=1,
                               output_dimension=3,
                               max_iterations=max_iterations,
                               distance='cosine',
                               conditional='time_delta',
                               device='cuda_if_available',
                               verbose=True,
                               time_offsets=10)

    # cebra_target_model.fit(monkey_target.neural, monkey_target.discrete_index.numpy())
    # cebra_target = cebra_target_model.transform(monkey_target.neural)
    # print([list(monkey_target.discrete_index.numpy()).count(i) for i in range(8)])
    cebra_target_model.fit(zscore, label)
    cebra_target = cebra_target_model.transform(zscore)

    fig = plt.figure(figsize=(4, 2), dpi=300)
    plt.suptitle('CEBRA-behavior trained with target label', fontsize=5)

    ax = plt.subplot(121, projection = '3d')
    ax.set_title('All trials embedding', fontsize=5, y=-0.1)
    x = ax.scatter(cebra_target[:, 0],
                   cebra_target[:, 1],
                   cebra_target[:, 2],
                   # c=monkey_target.discrete_index,
                   c=label,
                   cmap=plt.cm.RdYlBu,
                   s=0.01)
    ax.axis('off')

    ax = plt.subplot(122, projection='3d')
    ax.set_title('direction-averaged embedding', fontsize=5, y=-0.1)
    # for i in range(8):
    for i, cmap in enumerate([plt.cm.Reds, plt.cm.Blues]):
        # Defining a boolean aray to select the neural data according to the current trial
        direction_trial = (label == i)
        # trial_avg = cebra_target[direction_trial, :].reshape(-1, 600, 3).mean(axis=0)
        trial_avg = cebra_target[direction_trial, :].reshape(-1, frame_per_stim, 3).mean(axis=0)
        trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:, None]
        ax.scatter(trial_avg_normed[:, 0],
                   trial_avg_normed[:, 1],
                   trial_avg_normed[:, 2],
                   # color=plt.cm.hsv(1 / 8 * i),
                   cmap=cmap,
                   c=np.flip(np.linspace(0, 1, frame_per_stim)),
                   s=0.1)
    ax.axis('off')
    plt.show()