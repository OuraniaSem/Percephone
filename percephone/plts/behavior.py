"""
Théo Gauvrit 18/01/2024
plotting psychometric graph-like for behavior and detection
"""
font_s = 10


def psycho_like_plot(rec, roi_info, ax):
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    converted_list = eval("[" + seq[0] + "]")
    # converted_list = [float(x) for x in seq[0].split(',')]
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[['right', 'top', 'bottom', 'left']].set_color("black")
    ax.tick_params(axis='both', labelsize=font_s)


def psycho_like_plot_and_synchro(rec, roi_info, ax):
    seq = roi_info["Stimulus detection"][roi_info["Number"] == rec.filename].values
    # converted_list = eval("[" + seq[0] + "]")
    converted_list = [float(x) for x in seq[0].split(',')]
    to_plot = []
    for amp in [0, 2, 4, 6, 8, 10, 12]:
        if len(rec.detected_stim[rec.stim_ampl == amp]) == 0:
            to_plot.append(0)
        else:
            res = sum(rec.detected_stim[rec.stim_ampl == amp]) / len(rec.detected_stim[rec.stim_ampl == amp])
            to_plot.append(res)
    ax.plot([0, 2, 4, 6, 8, 10, 12], converted_list)
    ax.plot([0, 2, 4, 6, 8, 10, 12], to_plot, linestyle='--')
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=font_s)

