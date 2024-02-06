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


def cross_cor(rec):
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
    fig, ax = plt.subplots()
    h = ax.imshow(corr[order_s][:, order_s], cmap="seismic", vmin=-1, vmax=+1, interpolation="none")
    fig.colorbar(h, ax=ax, label="Correlation Coefficient $C_{ij}$")
    ax.set_xlabel("Neuron i")
    ax.set_ylabel("Neuron j")
    ax.set_title(str(rec.filename) + " " + rec.genotype)
