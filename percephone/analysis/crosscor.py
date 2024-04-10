"""
Théo Gauvrit 18/01/2024
Cross correlation analysis to get share variability of neurons
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")
plt.switch_backend("Qt5Agg")

# get the clusters of neurons from MLR
# compute the mean activity trace for each cluster
# compute the var from the mean trace for each neuron
# compute the correlation matrix for every neuron against every neuron


def cross_cor(rec, fig, ax):
    order = []
    for label in np.unique(rec.mlr_labels_exc['neuron_labels'], axis=0):
        indexes = np.all(rec.mlr_labels_exc['neuron_labels'] == label, axis=1)
        if len(indexes) == 0:
            continue
        else:
            # avg_signal = np.average(rec.zscore_exc[indexes], axis=0)
            order.append(np.where(indexes)[0])

    order_s = np.concatenate(np.array(sorted(order, key=len)))
    order_s = [x for x in order_s if x in rec.mlr_labels_exc["indices_r2"]]
    corr = np.corrcoef(rec.zscore_exc)

    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.filename) + " " + rec.genotype)
    fig.suptitle('Cross cor for whole recording', fontsize=16)


def cross_cor_prestim(rec, ax, title):
    order = []
    for label in np.unique(rec.mlr_labels_exc['neuron_labels'], axis=0):
        indexes = np.all(rec.mlr_labels_exc['neuron_labels'] == label, axis=1)
        if len(indexes) == 0:
            continue
        else:
            # avg_signal = np.average(rec.zscore_exc[indexes], axis=0)
            order.append(np.where(indexes)[0])

    order_s = np.concatenate(np.array(sorted(order, key=len)))
    order_s = [x for x in order_s if x in rec.mlr_labels_exc["indices_r2"]]
    exc = rec.df_f_exc[:, np.linspace(rec.stim_time - int(1 * rec.sf), rec.stim_time, int(1 * rec.sf), dtype=int)]
    exc_ = exc.reshape(len(rec.df_f_exc), len(rec.stim_time) * int(1 * rec.sf))
    corr = np.corrcoef(exc_)
    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(title)



def cross_cor_stim(rec, fig, ax):
    order = []
    for label in np.unique(rec.mlr_labels_exc['neuron_labels'], axis=0):
        indexes = np.all(rec.mlr_labels_exc['neuron_labels'] == label, axis=1)
        if len(indexes) == 0:
            continue
        else:
            # avg_signal = np.average(rec.zscore_exc[indexes], axis=0)
            order.append(np.where(indexes)[0])

    order_s = np.concatenate(np.array(sorted(order, key=len)))
    order_s = [x for x in order_s if x in rec.mlr_labels_exc["indices_r2"]]
    exc = rec.df_f_exc[:, np.linspace(rec.stim_time, rec.stim_time + int(0.5 * rec.sf), int(0.5 * rec.sf),dtype=int)]
    exc_ = exc.reshape(len(rec.df_f_exc), len(rec.stim_time) * int(0.5 * rec.sf))
    corr = np.corrcoef(exc_)
    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.filename) + " " + rec.genotype)
    fig.suptitle('Cross cor during stim', fontsize=16)


def cross_cor_stim_sub(rec, fig, ax):
    order = []
    for label in np.unique(rec.mlr_labels_exc['neuron_labels'], axis=0):
        indexes = np.all(rec.mlr_labels_exc['neuron_labels'] == label, axis=1)
        if len(indexes) == 0:
            continue
        else:
            # avg_signal = np.average(rec.zscore_exc[indexes], axis=0)
            order.append(np.where(indexes)[0])

    order_s = np.concatenate(np.array(sorted(order, key=len)))
    order_s = [x for x in order_s if x in rec.mlr_labels_exc["indices_r2"]]
    bsl = rec.df_f_exc[:, np.linspace(rec.stim_time - int(3 * rec.sf), rec.stim_time,int(3 * rec.sf), dtype=int)]
    bsl_ = bsl.reshape(len(rec.df_f_exc), len(rec.stim_time) * int(3 * rec.sf))
    exc = rec.df_f_exc[:, np.linspace(rec.stim_time, rec.stim_time + int(0.5 * rec.sf), int(0.5 * rec.sf),dtype=int)]
    exc_ = exc.reshape(len(rec.df_f_exc), len(rec.stim_time) * int(0.5 * rec.sf))
    corr = np.corrcoef(exc_) - np.corrcoef(bsl_)

    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.filename) + " " + rec.genotype)
    fig.suptitle('Cross cor during stim with subtracted baseline', fontsize=16)


def cross_cor_stim_det(rec, fig, ax):
    order = []
    for label in np.unique(rec.mlr_labels_exc['neuron_labels'], axis=0):
        indexes = np.all(rec.mlr_labels_exc['neuron_labels'] == label, axis=1)
        if len(indexes) == 0:
            continue
        else:
            # avg_signal = np.average(rec.zscore_exc[indexes], axis=0)
            order.append(np.where(indexes)[0])

    order_s = np.concatenate(np.array(sorted(order, key=len)))
    order_s = [x for x in order_s if x in rec.mlr_labels_exc["indices_r2"]]
    bsl = rec.df_f_exc[:, np.linspace(rec.stim_time - int(3 * rec.sf), rec.stim_time,int(3 * rec.sf), dtype=int)]
    bsl_ = bsl.reshape(len(rec.df_f_exc), len(rec.stim_time) * int(3 * rec.sf))
    exc = rec.df_f_exc[:, np.linspace(rec.stim_time[rec.detected_stim], rec.stim_time[rec.detected_stim] + int(0.5 * rec.sf), int(0.5 * rec.sf),dtype=int)]
    exc_ = exc.reshape(len(rec.df_f_exc), len(rec.stim_time[rec.detected_stim]) * int(0.5 * rec.sf))
    corr = np.corrcoef(exc_) - np.corrcoef(bsl_)

    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.filename) + " " + rec.genotype)
    fig.suptitle('Cross cor during  DET stim with subtracted baseline', fontsize=16)


def cross_cor_stim_undet(rec, fig, ax):
    order = []
    for label in np.unique(rec.mlr_labels_exc['neuron_labels'], axis=0):
        indexes = np.all(rec.mlr_labels_exc['neuron_labels'] == label, axis=1)
        if len(indexes) == 0:
            continue
        else:
            # avg_signal = np.average(rec.zscore_exc[indexes], axis=0)
            order.append(np.where(indexes)[0])

    order_s = np.concatenate(np.array(sorted(order, key=len)))
    order_s = [x for x in order_s if x in rec.mlr_labels_exc["indices_r2"]]
    bsl = rec.df_f_exc[:, np.linspace(rec.stim_time - int(3 * rec.sf), rec.stim_time,int(3 * rec.sf), dtype=int)]
    bsl_ = bsl.reshape(len(rec.df_f_exc), len(rec.stim_time) * int(3 * rec.sf))
    exc = rec.df_f_exc[:, np.linspace(rec.stim_time[~rec.detected_stim], rec.stim_time[~rec.detected_stim] + int(0.5 * rec.sf), int(0.5 * rec.sf),dtype=int)]
    exc_ = exc.reshape(len(rec.df_f_exc), len(rec.stim_time[~rec.detected_stim]) * int(0.5 * rec.sf))
    corr = np.corrcoef(exc_) - np.corrcoef(bsl_)

    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.filename) + " " + rec.genotype)
    fig.suptitle('Cross cor during UNDET stim with subtracted baseline', fontsize=16)
