# region ======================================== Imports ==============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
from multiprocessing import cpu_count, pool
from scipy.stats import pointbiserialr

import percephone.core.recording as pc
# endregion
# region ======================================== Correlation ==========================================================

def perceptual_magnitude_behavior_corr(recs):
    """
    Correlates a single parameter witht he behavioral outcome for threshold trials using point biserial corrrelation.
    Can not be used to correlate all trials because need to include the amplitude as a covariate.

    Parameters
    ----------
    recs

    Returns
    -------

    """
    rows = []
    for rec in recs:
        # Keeping only the stimulation at threshold amplitude to limit bias
        threshold_trials_mask = rec.stim_ampl_filter(stim_ampl="all")
        # Getting the vector of the parameter to correlate with the behavior
        act_exc_vector = rec.get_perc_resp(pattern=1, n_type="EXC")[threshold_trials_mask]
        # Getting the vector of behavioral outcome
        behavior_vector = rec.detected_stim[threshold_trials_mask]
        if len(behavior_vector) > 2:
            # Point biserial correlation of the vectors
            r, p_val = pointbiserialr(act_exc_vector, behavior_vector)
            rows.append({"ID": rec.filename, "Genotype": rec.genotype, "session_threshold": rec.session_threshold,
                         "nb_threshold_trials": len(behavior_vector), "R2": r**2, "p_val": p_val})
        else:
            print(f"{rec.filename} {rec.genotype} excluded → only {len(behavior_vector)} threshold trial(s)")
    return pd.DataFrame(rows)





# endregion ============================================================================================================
# region ======================================== Modelling ============================================================

def model_behavior(recs):
    """
    Aims to model the behavioral outcome using a list of predictors defined as important for stimulus encoding (GLMM).

    Parameters
    ----------
    rec

    Returns
    -------

    """
    # === === Building the DataFrame === ===
    rows = []
    for rec in recs:
        feature_vectors = {
            # --- Retrieving the target and covariate ---
            "behavior": rec.detected_stim,
            "amplitude": rec.stim_ampl,
            # --- Retrieving the different predictors ---
            # Percentage of recruited neurons
            "act_EXC_perc": rec.get_perc_resp(pattern=1, n_type="EXC"),
            "inh_EXC_perc": rec.get_perc_resp(pattern=-1, n_type="EXC"),
            "act_INH_perc": rec.get_perc_resp(pattern=1, n_type="INH"),
            "inh_INH_perc": rec.get_perc_resp(pattern=-1, n_type="INH"),
            # Mean peak amplitude for responsive neurons
            "act_EXC_amp": rec.get_mean_param(pattern=1, n_type="EXC", parameter="Peak_amplitude"),
            "inh_EXC_amp": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="Peak_amplitude"),
            "act_INH_amp": rec.get_mean_param(pattern=1, n_type="INH", parameter="Peak_amplitude"),
            "inh_INH_amp": rec.get_mean_param(pattern=-1, n_type="INH", parameter="Peak_amplitude"),
            # Mean peak delay for responsive neurons
            "act_EXC_delay": rec.get_mean_param(pattern=1, n_type="EXC", parameter="Peak_delay"),
            "inh_EXC_delay": rec.get_mean_param(pattern=-1, n_type="EXC", parameter="Peak_delay"),
            "act_INH_delay": rec.get_mean_param(pattern=1, n_type="INH", parameter="Peak_delay"),
            "inh_INH_delay": rec.get_mean_param(pattern=-1, n_type="INH", parameter="Peak_delay"),
        }
        nb_trials = len(feature_vectors["behavior"])
        for trial_id in range(nb_trials):
            row = {"ID": rec.filename, "Genotype": rec.genotype}
            for feature, vector in feature_vectors.items():
                row[feature] = vector[trial_id]
            rows.append(row)
    data = pd.DataFrame(rows)
    data["Genotype"] = pd.Categorical(data["Genotype"], categories=["WT", "KO", "KO-Hypo"], ordered=True)
    # data["behavior"] = pd.Categorical(data["behavior"], categories=["False", "True"], ordered=True)
    # data["behavior"] = data["behavior"].astype(int)
    # === === Fitting the model === ===
    # --- GEE ---
    formula = ("behavior ~ amplitude*Genotype + amplitude:act_EXC_perc*Genotype + amplitude:inh_EXC_perc*Genotype + "
               "amplitude:act_INH_perc*Genotype + amplitude:inh_INH_perc*Genotype")# + "
    #            # "act_EXC_amp + inh_EXC_amp + act_INH_amp + inh_INH_amp + "
    #            # "act_EXC_delay + inh_EXC_delay + act_INH_delay + inh_INH_delay + "
    #            # "Genotype + amplitude:Genotype")
    # gee_model = smf.gee(formula, groups="ID", data=data[data["Genotype"] == "WT"], family=sm.families.Binomial())
    # gee_result = gee_model.fit()
    # print(gee_result.summary())
    # --- GLMM ---
    vc_formulas = {"ID": "0 + C(ID)"}
    model = sm.genmod.BinomialBayesMixedGLM.from_formula(formula, vc_formulas, data=data)#, family=sm.families.Binomial())
    result = model.fit_vb()
    print(result.summary())
    return data


# endregion ============================================================================================================



if __name__ == '__main__':
    # region ====== Initialisation of recs instances ======
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
    for rec in recs.values():
        rec.peak_delay_amp()
    # endregion
    # test = perceptual_magnitude_behavior_corr(recs.values())
    # mean_corr = test.groupby("Genotype").mean()
    data = model_behavior(recs.values())
