"""Théo Gauvrit, 17/01/2024
Cross correlation of the activity of neurons to decipher if there are sharing variability"""

import json
import os
import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
from responsivity import responsivity, resp_single_neuron
from response import resp_matrice, auc_matrice, delay_matrice
from Helper_Functions.Utils_core import read_info
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")

# get the clusters of neurons from MLR
# compute the mean activity trace for each clusters
# compute the var from the mean trace for each neurons
# compute the correlation matrix for every neurons against every neurons
