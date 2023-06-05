"""Théo Gauvrit
16 May 2022
Delta F over F and corrections-> Alexandre Cornier in spiflash
Recordings class to centralize all the signal and information about behaviour"""
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

matplotlib.use('Qt5Agg')
plt.switch_backend('Qt5Agg')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3


class Recordings:
    def __init__(self, f, fneu, behaviour, sf):
        self.sampling_rate = sf
        self.delta_f_over_f = None
        self.add_delta_f_over_f(f, fneu)
        self.behaviour_timings = behaviour
        self.n_unit, self.n_frame = f.shape
        self.time_range = np.linspace(0, self.n_frame / self.sampling_rate, self.n_frame)

    def add_delta_f_over_f(self, f_, f_neu):
        print("Computating Delta F over F.")
        first_percentile = np.nanpercentile(f_, 1, axis=1)
        F_bkg_corrected = f_ - first_percentile[:, np.newaxis]
        Fneu_bkg_corrected = f_neu - first_percentile[:, np.newaxis]
        factor = 0.7
        F_NN_corrected = F_bkg_corrected - Fneu_bkg_corrected * factor
        sigma = 0.1 * self.sampling_rate
        F_smoothed = gaussian_filter1d(F_NN_corrected, sigma=sigma)
        window_size = int(5 * self.sampling_rate)
        half_window = int(window_size / 2)
        start = half_window
        end = len(f_[0]) - half_window - 1
        F0 = np.zeros_like(f_)
        for i in tqdm(range(start, end + 1)):
            F_window = F_smoothed[:, i - half_window:i + half_window]
            F0[:, i] = np.median(F_window, axis=1)
        F0[:, :start] = F0[:, start][:, np.newaxis]
        F0[:, end:] = F0[:, end][:, np.newaxis]
        self.delta_f_over_f = np.divide(np.subtract(F_smoothed, abs(F0)), abs(F0))
        # self.delta_f_over_f = abs(F0)

    def get_trials(self):
        pass
        # trial = Trials()
        # return trial

    def plot_trials(self, units, timing=None):
        if timing is None:
            timing = [0, 2000]
        timing = np.multiply(timing, self.sampling_rate)
        print("Plotting units traces.")
        fig, ax = plt.subplots(1, 1, figsize=(13, 8))
        for i, unit in tqdm(enumerate(units)):
            to_plot = np.subtract(self.delta_f_over_f[unit, timing[0]:timing[1]].T, np.mean(self.delta_f_over_f[unit,
                                                                                            timing[0]:timing[1]].T))
            ax.plot(self.time_range[timing[0]:timing[1]], to_plot + i * 10)
        plt.xlabel('Time [s]')
        plt.ylabel(r'$\Delta F/F$ [fold]')
        plt.show()

    def plot_heat_map(self):
        print("Plotting heatmap.")
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.pcolormesh(self.time_range, np.arange(len(self.delta_f_over_f)), self.delta_f_over_f, vmin=0, vmax=7,
                      cmap="Blues")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Neuronal  Units')
        plt.tick_params(axis="x", which="both", width=2)
        plt.tick_params(axis="y", which="both", width=2)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    start_time = time.time()
    F = np.load('/datas/Théo/Projects/Percephone/data/20220728_4445_02_synchro/F.npy', allow_pickle=True)
    F_neu = np.load('/datas/Théo/Projects/Percephone/data/20220728_4445_02_synchro/Fneu.npy', allow_pickle=True)
    recording = Recordings(F, F_neu, None, 30)
    recording.plot_heat_map()
    recording.plot_trials([0, 2, 21, 15, 12, 1, 5, 8, 9])
    print("--- End ---")
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
