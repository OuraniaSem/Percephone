"""" Adrien Corniere
14/05/2024
test multi trials"""
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
roi_info = directory + "\\FmKO_ROIs&inhibitory.xlsx"
files = os.listdir(directory)
files_ = [file for file in files if file.endswith("synchro")]


def opening_rec(fil,i):
    rec = pc.RecordingAmplDet(directory + fil + "/", 0,  roi_info)
    return rec

workers = cpu_count()
pool = pool.ThreadPool(processes=workers)
async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
recs_ = [ar.get() for ar in async_results]

for group in ["WT", "KO"]:
    recs = [ar for ar in recs_ if ar.genotype == group]
    multi_cebra_model = cebra.CEBRA(
        model_architecture="offset1-model",
        batch_size=1024,
        temperature_mode="auto",
        learning_rate=0.001,
        max_iterations=10000,
        time_offsets=10,
        output_dimension=3,
        conditional='time_delta',
        device="cuda_if_available",
        verbose=True
    )
    labels, datas, colors = [], [], []
    for rec in recs:
            neural_data = np.transpose(rec.zscore_exc)

            discrete_label = np.zeros(len(neural_data))
            index_stim = np.concatenate([list(np.arange(i, i+17)) for i in rec.stim_time[rec.detected_stim]])
            index_stim_undet = np.concatenate([list(np.arange(i, i+17)) for i in rec.stim_time[~rec.detected_stim]])
            # index_rew = np.concatenate([list(np.arange(i, i+10)) for i in rec.reward_time])
            # index_timeout = np.concatenate([list(np.arange(i, i+40)) for i in rec.timeout_time])
            index_lick = np.concatenate([list(np.arange(i, i+15)) for i in rec.lick_time])
            discrete_label[index_stim[index_stim < len(discrete_label)]] = 1
            discrete_label[index_stim_undet[index_stim_undet < len(discrete_label)]] = 2
            # discrete_label[index_rew[index_rew < len(discrete_label)]] = 3
            # discrete_label[index_timeout[index_timeout < len(discrete_label)]] = 4
            discrete_label[index_lick[index_lick < len(discrete_label)]] = 5
            color_labels = np.full(len(discrete_label), 'c')
            color_labels[index_stim[index_stim < len(discrete_label)]] = 'b'
            color_labels[index_stim_undet[index_stim_undet < len(discrete_label)]] = 'k'
            color_labels[index_lick[index_lick < len(discrete_label)]] = "green"
            labels.append(discrete_label)
            datas.append(neural_data)
            colors.append(color_labels)

    multi_cebra_model.fit(datas, labels)
    multi_embeddings = dict()
    for i, (rec, X) in enumerate(zip(recs, datas)):
        multi_embeddings[rec.filename] = multi_cebra_model.transform(X, session_id=i)
    for rec, name, color in zip(recs, multi_embeddings.keys(),colors):
        cebra.plot_embedding(multi_embeddings[name], embedding_labels=color, markersize=5, idx_order=(0, 1, 2), title=f"{rec.filename}  {rec.genotype}")
        # cebra.plot_embedding(multi_embeddings[name], embedding_labels=color, markersize=5, idx_order=(0, 1), title=f"{rec.filename}  {rec.genotype}")
        # ax = cebra.plot_loss(multi_cebra_model)
    plt.show()

    """" Adrien Corniere
    14/05/2024
    test multi trials"""
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

    # rnd.seed(20552)
    directory = "C:\\Users\\acorniere\\Desktop\\percephone_data\\"
    roi_info = directory + "\\FmKO_ROIs&inhibitory.xlsx"
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]


    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_info)
        return rec


    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs_ = [ar.get() for ar in async_results]

    for group in ["WT", "KO"]:
        recs = [ar for ar in recs_ if ar.genotype == group]
        multi_cebra_model = cebra.CEBRA(
            model_architecture="offset1-model",
            batch_size=1024,
            temperature_mode="auto",
            learning_rate=0.001,
            max_iterations=10000,
            time_offsets=10,
            output_dimension=3,
            conditional='time_delta',
            device="cuda_if_available",
            verbose=True
        )
        labels, datas, colors = [], [], []
        for rec in recs:
            neural_data = np.transpose(
                np.mean(rec.zscore_exc[:, np.linspace(rec.stim_time - 15, rec.stim_time + 15, 45, dtype=int)], axis=2))
            discrete_label = np.zeros(len(neural_data))

            discrete_label[0:15] = 1  # prestim
            discrete_label[15:30] = 2  # stim
            discrete_label[30:45] = 3  # poststim
            color_labels = np.full(45, 'b')
            color_labels[15:30] = 'k'
            color_labels[30:45] = "green"

            labels.append(discrete_label)
            datas.append(neural_data)
            colors.append(color_labels)

        multi_cebra_model.fit(datas, labels)
        multi_embeddings = dict()
        for i, (rec, X) in enumerate(zip(recs, datas)):
            multi_embeddings[rec.filename] = multi_cebra_model.transform(X, session_id=i)
        for rec, name, color in zip(recs, multi_embeddings.keys(), colors):
            cebra.plot_embedding(multi_embeddings[name], embedding_labels=color, markersize=5, idx_order=(0, 1, 2),
                                 title=f"{rec.filename}  {rec.genotype}")
            cebra.plot_embedding(multi_embeddings[name], embedding_labels=color, markersize=5, idx_order=(0, 1),
                                 title=f"{rec.filename}  {rec.genotype}")
            # ax = cebra.plot_loss(multi_cebra_model)
        plt.show()