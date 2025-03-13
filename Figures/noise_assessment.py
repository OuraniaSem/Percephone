# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool

import percephone.core.recording as pc
# endregion
# region ======================================== TBT variability ======================================================




# endregion ============================================================================================================
# region ======================================== Pre-stimulus =========================================================





# endregion ============================================================================================================
# region ======================================== E/I Ratio ============================================================





# endregion ============================================================================================================


if __name__ == '__main__':
    ### Initialisation of recs instances ###
    directory = "C:/Users/cvandromme/Desktop/Data/"
    roi_path = "C:/Users/cvandromme/Desktop/FmKO_ROIs&inhibitory.xlsx"
    server_address = "Z:/Current_members/Ourania_Semelidou/2p/Figures_paper/"
    roi_info = pd.read_excel(roi_path)
    files = os.listdir(directory)
    files_ = [file for file in files if file.endswith("synchro")]
    def opening_rec(fil, i):
        rec = pc.RecordingAmplDet(directory + fil + "/", 0, roi_path)
        return rec
    workers = cpu_count()
    pool = pool.ThreadPool(processes=workers)
    async_results = [pool.apply_async(opening_rec, args=(file, i)) for i, file in enumerate(files_)]
    recs = {ar.get().filename: ar.get() for ar in async_results}

    # Dropping 5886 from the noise assessment analysis because its computed threshold is 3 (10% hit rate for 2µm and 90% for 4µm)
    excluded_rec = recs.pop(5886)

    # ====== Comparison of threshold to session threshold ======
    rows = []
    for rec in recs.values():
        rows.append({"ID": rec.filename, "Genotype": rec.genotype, "threshold": rec.threshold, "session_threshold": rec.session_threshold, "session_x0": rec.x0_psy})
    session_threshold = pd.DataFrame(rows)

    from percephone.utils.math_formulas import sigmoid_fit
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(20, 12), constrained_layout=True)
    axs = ax.flatten()
    for i, rec in enumerate(recs.values()):
        axs[i].set_title(f"{rec.filename} - {rec.threshold}/{rec.session_threshold}({rec.x0_psy:.2f})")
        axs[i].set_ylim(0, 1)
        axs[i].scatter(np.arange(start=2, stop=13, step=2), rec.hit_rates[1:])
        x, y, x0, k = sigmoid_fit(np.arange(start=0, stop=13, step=2), rec.hit_rates)
        axs[i].plot(x, y, color='red')
    plt.show()