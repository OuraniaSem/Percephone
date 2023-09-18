"""Théo Gauvrit 30/06/2023
Get analog trace from the mesc file (hdf5 type file) version 2022
Need Msession that indicates wich recording it is"""

import h5py
import matplotlib
import numpy as np
import pandas as pd
import core as pc
import scipy.interpolate as si
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# analog = pd.read_csv("/datas/Théo/Projects/Percephone/data/StimulusOnlyWT/20221128_4939_00_synchro/analog.txt" , sep="\t")

# file = h5py.File('/run/user/1004/gvfs/smb-share:server=engram,share=data/Current_members/Ourania_Semelidou/2p/fmkoB6_4938_4939_4942_4943/4939'
#               '/20221128_4939_StimOnly.mesc', 'r')
#
# dset = file ['MSession_0']
# list(dset .keys())
# unit = dset['MUnit_0']
# list(unit.keys())
# c0 = unit['Curve_0']
# c1 = unit['Curve_1']
# list(c0.keys())
# curve0 = np.array(c0['CurveDataYRawData'])
# curve1 = np.array(c1['CurveDataYRawData'])
# ref_values = c0.attrs.get("CurveDataYConversionReferenceValues")
# f = si.interp1d(ref_values[::2], ref_values[1::2])
# analog_interp = f(curve0)
# fig, axs = plt.subplots(3, 1, figsize=(18, 10))
# axs[0].plot(analog.iloc[:, 1])
# axs[1].plot(analog_interp)
# axs[2].plot(curve1)
# plt.show()
#
# df = pd.DataFrame({0: np.linspace(0, len(analog_interp)/10, len(analog_interp)), 1: analog_interp})
# df.to_csv("analog.txt", sep="\t", index=False)
#
#

"""  Linear transfromation for 2023 files """
file = h5py.File("/run/user/1004/gvfs/smb-share:server=engram.local,share=data/Current_members/Ourania_Semelidou/2p/fmkoB6_5137_5140_5141_5143_5148_5149/5140/20230301_5140_StimOnly.mesc")
dset = file['MSession_0']
unit = dset['MUnit_5']  # change according to number
c0 = unit['Curve_2']
curve0 = np.array(c0['CurveDataYRawData'])
offset = c0.attrs.get("CurveDataYConversionConversionLinearOffset")
scale = c0.attrs.get("CurveDataYConversionConversionLinearScale")
analog_linear = curve0*scale
fig, axs = plt.subplots(2, 1, figsize=(18, 10))
# axs[0].plot(analog.iloc[:, 1])
axs[1].plot(curve0)
plt.show()
