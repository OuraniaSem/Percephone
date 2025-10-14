"""
Théo Gauvrit 18/01/2024
Style and aesthetics for plots
"""
import matplotlib as mpl


mpl.rcParams["axes.grid"] = False
mpl.rcParams['font.size'] = 35
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['lines.linewidth'] = 5
font_signif = mpl.rcParams['font.size'] / 2

mpl.rcParams["boxplot.whiskerprops.linewidth"] = 5
mpl.rcParams["boxplot.boxprops.linewidth"] = 5
mpl.rcParams["boxplot.capprops.linewidth"] = 5
mpl.rcParams["boxplot.medianprops.linewidth"] = 5
mpl.rcParams["boxplot.meanprops.linewidth"] = 5
mpl.rcParams["boxplot.flierprops.linewidth"] = 5


mpl.rcParams["xtick.labelsize"] = mpl.rcParams['font.size']
mpl.rcParams["ytick.labelsize"] = mpl.rcParams['font.size']
mpl.rcParams["axes.labelsize"] = mpl.rcParams['font.size']
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["lines.markersize"] = 28

mpl.rcParams['svg.fonttype'] = 'none'

mpl.rcParams["xtick.major.width"] = 3
mpl.rcParams["xtick.minor.width"] = 2
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.width"] = 3
mpl.rcParams["ytick.minor.width"] = 2
mpl.rcParams["ytick.major.size"] = 6
mpl.rcParams["ytick.left"] = True

# region === === === Tactile Detection === === ===
# ========== For genotypes ==========
wt_color = "#3d6993"
wt_light_color = "#7aabd2"
wt_bms_color = "#2bd0f1"
wt_bms_light_color = "#95e7f8"

all_ko_color = "#CC0000"
all_ko_light_color = "#ff8080"
all_ko_bms_color = "#c74375"
all_ko_bms_light_color = "#e3a1ba"

hypo_color = "firebrick"
hypo_light_color = "#e18282"
hypo_bms_color = "#da004a"
hypo_bms_light_color = "#f17581"

ko_color = "#c57c9a"
ko_light_color = "#fda7ca"  # I defined this one

color_dict = {
    # WT
    "WT": [wt_color, wt_light_color],
    "WT-DMSO": [wt_color, wt_light_color],
    "WT-BMS": [wt_bms_color, wt_bms_light_color],
    # KO-Hypo
    "KO-Hypo": [hypo_color, hypo_light_color],
    "KO-DMSO": [hypo_color, hypo_light_color],
    "KO-BMS": [hypo_bms_color, hypo_bms_light_color],
    # KO
    "KO": [ko_color, ko_light_color]}

# ========== For neurons ==========
exc_color = "#229708"
inh_color = "#cba61b"
exc_inh_color = "#859717"

# endregion
# region === === === Learning and Motivation === === ===
# ========== For conditions ==========
naive_color = "#4cc9f0"
trained_color = "#480ca8"
motivated_color = "#f72585"
# endregion