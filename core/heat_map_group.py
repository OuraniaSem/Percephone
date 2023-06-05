"""Théo Gauvrit 24/04/2023
"""

import numpy as np
import pandas as pd
import matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from scalebars import add_scalebar


resp_data = pd.read_excel("/datas/Théo/Projects/Percephone/core/Responsivness_neurons.xlsx", sheet_name="neurons_resp")

WT_data = resp_data[resp_data["Genotype"] == "WT"]
to_plot_WT = np.array(WT_data.iloc[:, 1:7])
print("Plotting heatmap.")


Z = linkage(to_plot_WT, 'ward')
dn = dendrogram(Z)
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
c = ax.pcolor(np.arange(6), np.arange(len(to_plot_WT)), to_plot_WT[dn['leaves']], vmin=0, vmax=1, cmap="Reds")
plt.xlabel('2 4 6 8 10 12', fontsize=30)
ax.tick_params(which='both', width=4)
plt.ylabel('Neurons', fontsize=30)
co = fig.colorbar(c, ax=ax)
co.ax.tick_params(which='both', width=4)
fig.tight_layout()
plt.show()
