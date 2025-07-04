"""
01/11/2024
Adrien Corniere

Stats related plot functions like boxplot, barplot...
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import math
import warnings

import mplcursors
import numpy as np
from scipy.stats import levene, shapiro, mannwhitneyu, yeojohnson, boxcox
import pingouin as pg
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from percephone.plts.style import *
from percephone.plts.utils import *
from percephone.utils.math_formulas import inv_transform, arcsin_transform

mpl.rcParams["axes.grid"] = False
mpl.rcParams['font.size'] = 35
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['lines.linewidth'] = 5
# mpl.rcParams['font.size'] = 10
# mpl.rcParams['axes.linewidth'] = 1
# mpl.rcParams['lines.linewidth'] = 1
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
# mpl.rcParams["axes.titlesize"] = 12
# mpl.rcParams["lines.markersize"] = 8

mpl.rcParams['svg.fonttype'] = 'none'

mpl.rcParams["xtick.major.width"] = 3
mpl.rcParams["xtick.minor.width"] = 2
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.width"] = 3
mpl.rcParams["ytick.minor.width"] = 2
mpl.rcParams["ytick.major.size"] = 6
mpl.rcParams["ytick.left"] = True

mpl.use("Qt5Agg")

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

ko_color = "#c57c9a"
ko_light_color = "#fda7ca"  # I defined this one




@boxplot_style
def boxplot(ax, gp1, gp2, ylabel, paired=False, title="", ylim=[], colors=[wt_color, hypo_color],
            det_marker=True, force_markers_identity=False, force_normality=False):
    """
    create boxplot for two data groups.

    Parameters
    ----------
    wt : numpy.ndarray, series, list
        data of the wt group
    ko : numpy.ndarray, series, list
        data of the ko group
    ylabel : string
        columns names

    """
    if paired:
        gp1_nan = np.array(gp1)[~np.isnan(gp1) & ~np.isnan(gp2)]
        gp2_nan = np.array(gp2)[~np.isnan(gp1) & ~np.isnan(gp2)]
    else:
        gp1_nan = np.array(gp1)[~np.isnan(gp1)]
        gp2_nan = np.array(gp2)[~np.isnan(gp2)]

    groups = [gp1_nan, gp2_nan]
    x = [0.15, 0.40]
    if not paired or force_markers_identity:
        markers = ["o", "o"] if det_marker else ["v", "v"]
    else:
        markers = ["o", "v"]

    for index in range(2):
        # Plot the boxplots
        ax.boxplot(groups[index], positions=[x[index]], patch_artist=True, showfliers=False, widths=0.2,
                   meanprops=dict(marker=markers[index], markerfacecolor=colors[index], markeredgecolor='black'),
                   boxprops=dict(facecolor='white', color=colors[index]),
                   capprops=dict(color=colors[index]),
                   whiskerprops=dict(color=colors[index]),
                   medianprops=dict(color=colors[index]))
        # Plot the data points
        if paired:
            x_random = [x[index]] * len(groups[index])
        else:
            x_random = np.random.normal(x[index], 0.02, size=len(groups[index]))
        ax.plot(x_random, groups[index], marker=markers[index], alpha=0.5, ms=14, markerfacecolor="None", linestyle="None", markeredgecolor=colors[index], markeredgewidth=4)
    # Plot the connecting lines between data points if paired
    if paired:
        for i in range(len(gp1)):
            ax.plot([x[0], x[1]], [gp1[i], gp2[i]], marker=None, color=colors[1], alpha=0.5, linewidth=2.5,
                    markersize=14, markeredgewidth=4, markeredgecolor=colors[0], markerfacecolor=colors[1])

    # Retrieving the maximum of the data for the ylim and significance
    max_y_0 = max(np.nanmax(gp1), np.nanmax(gp2))
    min_y_0 = min(np.nanmin(gp1), np.nanmin(gp2))
    max_y = max_y_0 if np.isfinite(max_y_0) else 1
    min_y = min_y_0 if np.isfinite(min_y_0) else 0

    # Setting the ylim if specified
    if len(ylim) != 0:
        if not (ylim[0] <= min_y and ylim[1] >= max_y):
            warnings.warn("The ylim you have set don't cover the data range.")
        ax.set_ylim(ylim)
    else:
        y_lim_sup = ax.get_ylim()[1]
        ax.set_ylim(top=y_lim_sup * (1.2 if y_lim_sup > 0 else 0.8))

    # Plotting the significance bar
    bar_color = "black"
    y = max_y + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x[0], x[1]], [y, y], lw=mpl.rcParams['axes.linewidth'], c=bar_color, clip_on=False)
    # Computing and plotting the significance symbol
    if len(gp1_nan) > 2 and len(gp2_nan) > 2:
        pval = stat_boxplot(gp1_nan, gp2_nan, ylabel, title=title, paired=paired, force_normality=force_normality)
        sig_symbol = symbol_pval(pval)
    else:
        sig_symbol = "n.a"
    ax.text((x[0] + x[1]) * 0.5, y, sig_symbol, ha='center', va='bottom', c=bar_color, fontsize=font_signif)

    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.set_xticks([])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(None)


def barplot(wt, ko, ylabel):
    """
    create barplot for two data groups.

    Parameters
    ----------
    wt : numpy.ndarray, series, list
        data of the wt group,
    ko : numpy.ndarray, series, list
        data of the ko group,
    ylabel : string
        columns names,

    -------

    """
    print('barplot plotting')
    var_wt = np.var(wt)
    var_ko = np.var(ko)
    print(np.var(wt))
    print(np.var(ko))
    y = [np.var(wt), np.var(ko)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([1.7, 1.8], width=0.05, height=y, color=[wt_light_color, all_ko_light_color], edgecolor=[wt_color, all_ko_color],
           linewidth=3)
    ax.set_xlim([1.65, 1.9])

    ax.set_ylabel(ylabel + " Var ")
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.spines["bottom"].set_visible(True)
    ax.set_xticks([])
    fig.tight_layout()
    x1, x2 = 1.7, 1.8
    max_data = max([var_wt, var_ko])
    y, col = max_data + 0.15 * abs(max_data), 'k'
    ax.plot([x1, x2], [y, y], lw=3, c=col)

    pval = stat_varplot(wt, ko, ylabel)
    sig_symbol = symbol_pval(pval)

    ax.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col)

# kept for backwards compatibility
@boxplot_style
def paired_boxplot(ax, det, undet, ylabel, title, ylim=[], colors=[all_ko_color, all_ko_light_color], allow_stats_skip=False, variant = False):
    """
    create boxplot for two data groups.

    Parameters
    ----------
    det : numpy.ndarray, series, list
        data of the detected group
    undet : numpy.ndarray, series, list
        data of the undetected group
    ylabel : string
        columns names
    allow_stats_skip : bool


    """
    print("Boxplot plotting.")
    marker1, marker2 = "o",  "o"
    if variant:
        marker2 = "v"
    det_nan = np.array(det)[~np.isnan(det)]
    undet_nan = np.array(undet)[~np.isnan(undet)]
    ax.set_ylabel(ylabel)
    ax.boxplot(det_nan, positions=[0.15], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker="o", markerfacecolor=colors[0], markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=colors[0]),
               capprops=dict(color=colors[0]),
               whiskerprops=dict(color=colors[0]),
               medianprops=dict(color=colors[0]))
    ax.boxplot(undet_nan, positions=[0.40], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker="o", markerfacecolor=colors[1], markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=colors[1]),
               capprops=dict(color=colors[1]),
               whiskerprops=dict(color=colors[1]),
               medianprops=dict(color=colors[1]))
    for i in range(len(det)):
        ax.plot([0.15], [det[i]], marker=marker1, alpha=0.5,markersize=14, markeredgewidth=4,
                markeredgecolor=colors[0], markerfacecolor=(1,1,1,0))
        ax.plot([0.40], [undet[i]], marker=marker2, alpha=0.5, markersize=14, markeredgewidth=4,
                markeredgecolor=colors[1], markerfacecolor=(1,1,1,0))
        ax.plot([0.15, 0.40], [det[i], undet[i]], marker="", color=colors[1], linewidth=2.5)

    ax.set_xlabel(None)
    max_y = max(np.nanmax(det), np.nanmax(undet))
    if len(ylim) != 0:
        ax.set_ylim(ylim)
    else:
        max_y = max(np.nanmax(det), np.nanmax(undet))
        lim_max = max(int(max_y * 0.15 + max_y), int(math.ceil(max_y / 2)) * 2)
        min_y = min(np.nanmin(det), np.nanmin(undet))
        lim_inf = min(0, min_y + 0.15 * min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.set_xticks([])

    x_1, x_2 = [0.15, 0.40]
    max_data = max([np.nanmax(det), np.nanmax(undet)])
    y, col = max_data + 0.10 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col)

    if allow_stats_skip:
        try:
            pval = stat_boxplot(det, undet, ylabel, paired=True)
            sig_symbol = symbol_pval(pval)
            ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_signif)
        except ValueError:
            pass
    else:
        pval = stat_boxplot(det, undet, ylabel, paired=True)
        sig_symbol = symbol_pval(pval)
        ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_signif)

    ax.set_xticks([0.15, 0.40], ['', ""])
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.set_title(title)
    ax.tick_params(axis='y')


@boxplot_style
def dmso_bms(ax, wt_dmso, wt_bms, ko_dmso, ko_bms, ylabel, title="", ylim=[], colors=[wt_color, wt_bms_color, all_ko_color, all_ko_bms_color], marker="o"):
    print("Boxplot plotting.")
    wt_dmso_nan = np.array(wt_dmso)[~np.isnan(wt_dmso)]
    wt_bms_nan = np.array(wt_bms)[~np.isnan(wt_bms)]
    ko_dmso_nan = np.array(ko_dmso)[~np.isnan(ko_dmso)]
    ko_bms_nan = np.array(ko_bms)[~np.isnan(ko_bms)]
    x_1, x_2, x_3, x_4 = [0.05, 0.30, 0.60, 0.85]

    for position, group, color in zip([x_1, x_2, x_3, x_4], [wt_dmso_nan, wt_bms_nan, ko_dmso_nan, ko_bms_nan], colors):
        ax.boxplot(group, positions=[position], patch_artist=True, showfliers=False, widths=0.2,
                   meanprops=dict(marker='o', markerfacecolor=color, markeredgecolor='black'),
                   boxprops=dict(facecolor='white', color=color),
                   capprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   medianprops=dict(color=color))
        ax.plot([position]*len(group), group, marker=marker, alpha=0.5, ms=14, markerfacecolor="None",
                linestyle="None", markeredgecolor=color, markeredgewidth=4)
    for i in range(len(wt_dmso)):
        ax.plot([x_1, x_2], [wt_dmso[i], wt_bms[i]], marker=None, color=colors[1], alpha=0.5, linewidth=2.5,
                markersize=14, markeredgewidth=4, markeredgecolor=colors[0], markerfacecolor=colors[1])

    for j in range(len(ko_dmso)):
        ax.plot([x_3, x_4], [ko_dmso[j], ko_bms[j]], marker=None, color=colors[3], alpha=0.5, linewidth=2.5,
                markersize=14, markeredgewidth=4, markeredgecolor=colors[2], markerfacecolor=colors[3])

    max_y = max(np.nanmax(wt_dmso), np.nanmax(wt_bms), np.nanmax(ko_dmso), np.nanmax(ko_bms))
    min_y = min(np.nanmin(wt_dmso), np.nanmin(wt_bms), np.nanmin(ko_dmso), np.nanmin(ko_bms))
    max_y_wt = max(np.nanmax(wt_dmso), np.nanmax(wt_bms))
    max_y_ko = max(np.nanmax(ko_dmso), np.nanmax(ko_bms))

    if len(ylim) != 0:
        if not (ylim[0] <= min_y and ylim[1] >= max_y):
            warnings.warn("The ylim you have set don't cover the data range.")
        ax.set_ylim(ylim)
    else:
        lim_max = max(int(max_y * 0.20 + max_y), int(math.ceil(max_y / 2)) * 2.1)
        lim_inf = min(0, min_y + 0.15 * min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)

    bottom, top = ax.get_ylim()
    color = "black"
    y_wt = max_y_wt + 0.05 * (top - bottom)
    y_ko = max_y_ko + 0.05 * (top - bottom)
    y_dmso = max(max_y_wt, max_y_ko) + 0.10 * (top - bottom)
    y_bms = y_dmso + 0.05 * (top - bottom)

    for positions, groups, nan_groups, y in zip([[x_1, x_2], [x_3, x_4]], [[wt_dmso, wt_bms], [ko_dmso, ko_bms]], [[wt_dmso_nan, wt_bms_nan], [ko_dmso_nan, ko_bms_nan]], [[y_wt, y_wt], [y_ko, y_ko]]):
        ax.plot(positions, y, lw=mpl.rcParams['axes.linewidth'], c=color)
        if len(nan_groups[0]) > 2 and len(nan_groups[1]) > 2:
            pval = stat_boxplot(groups[0], groups[1], ylabel, title=f"{title} ({groups[0]}/{groups[1]})", paired=True) * 2  # multiplied by 2 because of Bonferroni correction
            sig_symbol = symbol_pval(pval)
        else:
            sig_symbol = "n.a"
        ax.text((positions[0] + positions[1]) * 0.5, y[0], sig_symbol, ha='center', va='bottom', c=color, fontsize=font_signif)

    for positions, groups, nan_groups, y in zip([[x_1, x_3], [x_2, x_4]], [[wt_dmso, ko_dmso], [wt_bms, ko_bms]], [[wt_dmso_nan, ko_dmso_nan], [wt_bms_nan, ko_bms_nan]], [[y_dmso, y_dmso], [y_bms, y_bms]]):
        ax.plot(positions, y, lw=mpl.rcParams['axes.linewidth'], c=color)
        if len(nan_groups[0]) > 2 and len(nan_groups[1]) > 2:
            pval = stat_boxplot(groups[0], groups[1], ylabel, title=f"{title} ({groups[0]}/{groups[1]})", paired=False) * 2
            sig_symbol = symbol_pval(pval)
        else:
            sig_symbol = "n.a."
        ax.text((positions[0] + positions[1]) * 0.5, y[0], sig_symbol, ha='center', va='bottom', c=color, fontsize=font_signif)

    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.set_xticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    # ax.set_xticks([x_1, x_2, x_3, x_4], ["WT-DMSO", "WT-BMS", "KO-DMSO", "KO-BMS"])
    ax.tick_params(axis="x", which="both", bottom=False, labelrotation=45)
    ax.set_title(title)


def boxplot_anova(groups_data, lim_y, label_y, filename, colors, annot_text=[],
                  title="", thickformater=True, show_only_significant=False):
    fig = plt.figure(figsize=(3 * len(groups_data), 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.Axes(fig, [0., 0.5, 1., 1.])
    ax1 = fig.add_subplot(1, 1, 1, title=title)
    band = [0, 0.33 * len(groups_data)]
    plt.xlim(band)
    linewidth = 5
    positions = np.linspace(0.20, band[1] - 0.20, len(groups_data))
    box_props = dict(linewidth=linewidth, color=colors[1])
    whisker_props = dict(color=colors[1], linewidth=linewidth)
    cap_props = dict(color=colors[1], linewidth=linewidth)
    median_props = dict(color=colors[1], linewidth=linewidth)
    a = ax1.boxplot(groups_data,
                    positions=positions,
                    showfliers=False,
                    widths=0.2,
                    boxprops=box_props,
                    whiskerprops=whisker_props,
                    capprops=cap_props,
                    medianprops=median_props,
                    meanline=True,
                    showmeans=True)

    for i, group_data in enumerate(groups_data):
        a["boxes"][i].set(color=colors[i], linewidth=linewidth)
        a["whiskers"][i * 2].set(color=colors[i], linewidth=linewidth)
        a["whiskers"][i * 2 + 1].set(color=colors[i], linewidth=linewidth)
        a["caps"][i * 2].set(color=colors[i], linewidth=linewidth)
        a["caps"][i * 2 + 1].set(color=colors[i], linewidth=linewidth)
        a["medians"][i].set(color=colors[i], linewidth=linewidth)
        a["means"][i].set(linewidth=linewidth)

    plt.xticks(positions, [''] * len(groups_data))
    plt.ylim(lim_y)
    plt.ylabel(label_y)
    ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    for i, group_data in enumerate(groups_data):
        x = np.random.normal(positions[i], 0.02, size=len(group_data))
        ax1.plot(x, group_data, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=colors[i],
                 markeredgewidth=4)

    ax1.tick_params(axis='both', labelsize=35)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(which='both', width=3)
    ax1.tick_params(which='major', length=8)
    if thickformater:
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    plt.tick_params(axis="x", which="both", bottom=False, top=False)
    ax1.yaxis.labelpad = 10
    plt.subplots_adjust(left=None, bottom=0.2, right=0.99, top=0.9, wspace=None, hspace=None)

    # Calculate significance stars using stats_anova
    stars = stats_anova(*groups_data)
    max_d = max([max(data) for data in groups_data])
    y, h, col = max_d + abs(0.10 * max_d), 0.026 * abs(max_d), 'k'
    fixed_star_distance = 0.005 * h

    # Calculate positions of group pairs and sort them by the mean position
    positions_pairs = sorted(combinations(positions, 2), key=lambda x: np.mean(x))
    offset = 15 * h  # Adjust this offset factor for more space between comparisons

    num_stars = len(stars)
    significant_pairs = []

    # Filter pairs based on significance if needed
    for i, (pos1, pos2) in enumerate(positions_pairs):
        if not show_only_significant or stars[i] != "n.s":
            significant_pairs.append((pos1, pos2, stars[i]))
        print(f"Position pair: ({pos1}, {pos2}), Star: {stars[i]}")  # Debugging print

    # Plot only the filtered significant pairs
    for i, (pos1, pos2, star) in enumerate(significant_pairs):
        y_offset = i * h * offset  # Apply offset
        plt.plot([pos1, pos1, pos2, pos2], [y + y_offset, y + h + y_offset, y + h + y_offset, y + y_offset], lw=3,
                 c=col)
        plt.text((pos1 + pos2) * .5, y + y_offset + fixed_star_distance, star, ha='center', va='bottom', color=col)

    fig.tight_layout()


def boxplot_3_conditions(group1_data, group2_data, cond_labels=["A", "B", "C"],
                         title="",
                         lim_y="auto",
                         label_y=None,
                         y_percent=False,
                         color1=wt_color,
                         color2=all_ko_color,
                         legend_labels=None,
                         filename=None):
    """
    group1_data: list of 3 lists, one list per condition
    group2_data:  list of 3 lists, one list per condition
    Returns
    -------
    object
    """
    group1_data_nan = []
    group2_data_nan = []
    for conditions_idx in range(len(cond_labels)):
        group1_data_nan.append(np.array(group1_data[conditions_idx])[np.isfinite(group1_data[conditions_idx])])
        group2_data_nan.append(np.array(group2_data[conditions_idx])[np.isfinite(group2_data[conditions_idx])])
    print(group1_data_nan)
    print(group2_data_nan)
    fig, axs = plt.subplots(1, 3, figsize=(14, 8), sharey="all")
    for i, ax in enumerate(axs.flat):
        bx = ax.boxplot([group1_data_nan[i], group2_data_nan[i]],
                        positions=[0.15, 0.40],
                        showfliers=False,
                        widths=0.2,
                        boxprops=dict(color=color2),
                        whiskerprops=dict(color=color2),
                        capprops=dict(color=color2),
                        medianprops=dict(color=color2),
                        meanline=True,
                        showmeans=True)
        bx["boxes"][0].set(color=color1)
        bx["whiskers"][0].set(color=color1)
        bx["whiskers"][1].set(color=color1)
        bx["caps"][0].set(color=color1)
        bx["caps"][1].set(color=color1)
        bx["medians"][0].set(color=color1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.set_xticks([0.15, 0.40], ['', ""])
        ax.set_xlabel(cond_labels[i])
        ax.set_facecolor("white")
        y = group1_data_nan[i]
        x = np.random.normal(0.15, 0.02, size=len(y))
        y1 = group2_data_nan[i]
        x1 = np.random.normal(0.40, 0.02, size=len(y1))
        ax.plot(x1, y1, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color2, markeredgewidth=4)
        ax.plot(x, y, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color1, markeredgewidth=4)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis="y", which="both", left=False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, left=True)
        if len(group1_data_nan[i]) > 2 and len(group2_data_nan[i]) > 2:
            pval = stat_boxplot(group1_data_nan[i], group2_data_nan[i], f"{cond_labels[i]} group comp")
            sig_symbol = symbol_pval(pval)
        else:
            sig_symbol = "n.a"

        x1, x2, = 0.15, 0.40
        max_d = np.concatenate([np.concatenate(group1_data), np.concatenate(group2_data)]).max()
        y, h, col = max_d + abs(0.10 * max_d), 0.025 * abs(max_d), 'k'
        axs[i].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=3, c=col)
        axs[i].text((x1 + x2) * .5, y + h, sig_symbol, ha='center', va='bottom', color=col, fontsize=font_signif)

    axs[0].set_ylabel(label_y)
    axs[0].tick_params(axis='y')
    axs[0].yaxis.set_visible(True)
    if y_percent:
        axs[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axs[0].tick_params(which='major', length=10, width=4)

    if legend_labels is not None:
        hB, = ax.plot([1, 1], wt_color)
        hR, = ax.plot([1, 1], ko_color)
        ax.legend((hB, hR), legend_labels)
        hB.set_visible(False)
        hR.set_visible(False)

    if lim_y != "auto":
        plt.ylim(lim_y)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.2, hspace=None)
    plt.suptitle(title)

    if filename is not None:
        fig.savefig(filename)
    plt.show()
    # fig.tight_layout(pad=0.1)


def curveplot(ax, data, between="Genotype", within="Trial", variable=None,
              data_points=True,
              title=None, ylabel=None, xlabel=None, ylim=None, colors=None, id_display=True, legend_display=True, qq_show=True,
              transformation=None, consider_normality=False, consider_homogeneity=False):
    """
    Plot the evolution of a variable across different measurements for different groups, and perform the
    statistical analysis for the data provided in a long format Pandas DataFrame.

    Parameters
    ----------
    ax : axis
        The Matplotlib axis to use to plot the graph.
    data = Pandas DataFrame
        The Pandas DataFrame containing the data in long format (With an ID column).
    between : str
        The name of the column representing groups between which the variable is compared across measurements (the different curves).
    within : str
        The name of the column representing the different measurements of the variable (the different points on the x axis).
    variable : str
        The name of the column representing the variable that is plotted.
    title : str, optional
        The title of the plot.
    ylabel : str, optional
        The label for the y–axis.
    xlabel : str, optional
        The label for the x–axis.
    ylim : tuple, optional
        The y–limits for the plot.
    colors : list, optional
        A list of colors to use for each group.
    id_display : bool, optional
        Whether to display the ID of each data point on hover.
    qq_show : bool, optional
        Whether to display the QQ-plot.
    consider_normality : bool, optional
        If True, force the analysis to treat the data as normal even if the normality test fails.
    consider_homogeneity : bool, optional
        If True, force the analysis to treat the data as having homogeneous variances even if the test fails.

    Returns
    -------

    """
    global norm_pval_trans
    groups_names = sorted(data[between].unique())
    nb_groups = len(groups_names)
    measurements = sorted(data[within].unique())
    nb_subjects = len(data["ID"].unique())
    assert len(measurements) > 1, "Please provide a dataset with at least 2 measurements."
    # Defining colors if not provided
    if colors:
        assert len(colors) == nb_groups, f"Please provide {nb_groups} color(s) for the following group(s): {groups_names}"
    else:
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i / max(nb_groups - 1, 1)) for i in range(nb_groups)]  # Use max to avoid error if 1 group
    # Defining default values for xlabel, ylabel, and title
    ylabel = ylabel or variable
    xlabel = xlabel or within
    title = title or f"Evolution of {variable} across {within} for {between if nb_groups > 1 else groups_names[0]}"
    if data_points:
        # For each measurement, plot the data points associated with each group with their label when hovering
        x_pad = (measurements[1] - measurements[0]) / 10  # The maximum distance from the measurement x
        scatters = []
        all_labels = []
        for measure in measurements:
            # First defining the x coordinates for the different groups
            x_coords = np.linspace(start=measure - x_pad, stop=measure + x_pad, num=nb_groups) if nb_groups > 1 else [measure]
            for x, group, color in zip(x_coords, groups_names, colors):
                subset = data[(data[between] == group) & (data[within] == measure)]
                gp_measure = subset[variable].values
                if len(gp_measure) > 0:
                    gp_plot = ax.scatter([x] * len(gp_measure), gp_measure, color=color, alpha=0.5, marker="o", s=20)
                    scatters.append(gp_plot)
                    all_labels.append(subset["ID"].values)
                    # Displaying the ID when hovering the data points
        if id_display:
            cursor = mplcursors.cursor(scatters, hover=True)  # Single cursor for all scatters

            def on_add(sel):
                scatter_index = scatters.index(sel.artist)
                labels = all_labels[scatter_index]
                sel.annotation.set_text(labels[sel.index])
            cursor.connect("add", on_add)
    # Get the mean value of each group for each measurement and plot the corresponding curve
    for group, color in zip(groups_names, colors):
        mean_group = [np.nanmean(data[(data[between] == group) & (data[within] == measure)][variable].values) for measure in measurements]
        sem_group = [ss.sem(data[(data[between] == group) & (data[within] == measure)][variable].values) for measure in measurements]
        if data_points:
            ax.plot(measurements, mean_group, color=color, label=f"Mean {group}")
        else:
            ax.errorbar(measurements, mean_group, linestyle="-.", color=color, yerr=sem_group, capsize=6, markersize=15, lw=3, label=f"Mean {group}")

    if legend_display:
        ax.legend(fontsize=12)
    else:
        cursor = mplcursors.cursor(ax.lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    # Setting the title, axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(measurements)
    ax.tick_params(axis='both', labelsize=35)
    ax.spines[['right', 'top', ]].set_visible(False)
    # Setting the ylim
    min_data = np.nanmin(data[variable])
    max_data = np.nanmax(data[variable])
    if ylim:
        if data_points and not (ylim[0] <= min_data and ylim[1] >= max_data):
            warnings.warn("The ylim you have set don't cover the data range.")
            ax.set_ylabel("⚠⚠⚠ " + ylabel + " ⚠⚠⚠", fontweight="bold", color="red")
        ax.set_ylim(ylim)

    # === Testing for significant difference (RM ANOVA / Friedman test + Mann-Whitney U test) ===
    # --- Checking if the assumptions are met ---
    # Testing the normality of the whole dataset
    variable_col = variable
    # if only one group, its normality is assessed
    if nb_groups == 1:
        norm_pval = normality_check(data[variable_col], title=variable_col, plot=qq_show)
        test_norm = f"Shapiro-Wilk ➜ {norm_pval:.3f}"
        normality = norm_pval > 0.05
    # if several groups, the normality and homocedasticity of the residuals is assessed
    else:
        data["predicted"] = data.groupby([between, within])[variable_col].transform("mean")
        data["residuals"] = data[variable_col] - data["predicted"]
        norm_pval = normality_check(data["residuals"], title=f"{variable_col}_residuals", plot=qq_show)
        test_norm = f"Shapiro-Wilk (residuals) ➜ {norm_pval:.3f}"
        normality = norm_pval > 0.05
        homogen_pval = levene(*[data[data[between] == group]["residuals"] for group in groups_names]).pvalue
        homogeneity = True if (homogen_pval > 0.05 or consider_homogeneity) else False
        test_homogen = f"Levene (residuals) ➜ {homogen_pval:.3f}"

    # First solution would be to explore the transformation of the data, because it's always better to use
    # parametric tests whenever it's doable (don't stick to strict statistical tests to assess whether
    # the data is normal or not)
    if transformation:
        trans_functions = {
            "log": np.log,  # Compressing scale for high values (right skewness)
            "log1p": np.log1p,  # Shifted log(1+x) to handle 0 values
            "sqrt": np.sqrt,  # For count data or moderately skewed data
            "inv": inv_transform,  # Good for high skewness but exaggerate != for small values
            "arcsin": arcsin_transform,  # Percentage/proportion data, reduces variance
        }
        valid_transforms = list(trans_functions.keys()) + ["boxcox", "yeojohnson"]
        assert transformation in valid_transforms, f"Please use one of the following transformation: {valid_transforms}"
        variable_col = f"{transformation}_{variable}"
        # Applying the transformation
        if transformation in ["boxcox", "yeojohnson"]:
            if transformation == "boxcox": # data > 0, often achieve normality if the data follow a power-law type relationship
                transformed, fitted_lambda = boxcox(data[variable])
            else:  # yeojohnson, like boxcox but no need to be > 0
                transformed, fitted_lambda = yeojohnson(data[variable])
            data[variable_col] = transformed
            print(f"Fitted lambda for {variable}: {fitted_lambda}")
        else:
            data[variable_col] = trans_functions[transformation](data[variable])
        # Assessing the normality and homocedasticity after transformation
        if nb_groups == 1:
            norm_pval_trans = normality_check(data[variable_col], title=variable_col, plot=qq_show)
        else:
            data["predicted_trans"] = data.groupby([between, within])[variable_col].transform("mean")
            data["residuals_trans"] = data[variable_col] - data["predicted_trans"]
            norm_pval_trans = normality_check(data["residuals_trans"], title=f"{variable_col}_residuals", plot=qq_show)
            homogen_pval_trans = levene(*[data[data[between] == group]["residuals_trans"] for group in groups_names]).pvalue
            homogeneity = True if (homogen_pval_trans > 0.05 or consider_homogeneity) else False
            test_homogen += f"/{homogen_pval_trans:.3f} (after {transformation} transform)"
        test_norm = test_norm + f"/{norm_pval_trans:.3f} (after {transformation} transform)"
        normality = norm_pval_trans > 0.05

    if nb_groups > 1:
        resid_for_plot = data["residuals_trans"] if transformation and "residuals_trans" in data.columns else data["residuals"]
        pred_for_plot = data["predicted_trans"] if transformation and "predicted_trans" in data.columns else data["predicted"]
        homogeneity_check(predicted=pred_for_plot, residuals=resid_for_plot, title=variable_col)

    # Option to overwrite the normality assumption
    if consider_normality: normality = True

    # Testing sphericity, use Greenhouse–Geisser correction if not met
    sphericity_pval = pg.sphericity(data, dv="residuals" if not transformation else "residuals_trans", within=within, subject="ID").pval
    sphericity = sphericity_pval > 0.05
    GG_col = "p-unc" if sphericity else "p-GG-corr"
    GG_note = "" if sphericity else "[GG corrected]"

    # --- Performing the appropriate tests ---
    post_hoc_dict = {}
    if nb_groups > 1:
        if normality and homogeneity:
            # Performing the mixed ANOVA
            test_results = pg.mixed_anova(data=data, dv=variable_col, between=between, within=within, subject="ID", correction=True)
            p_val_between = test_results[test_results["Source"] == between]["p-unc"].iloc[0].astype(float)
            p_val_within = test_results[test_results["Source"] == within][GG_col].iloc[0].astype(float)
            p_val_inter = test_results[test_results["Source"] == "Interaction"][GG_col].iloc[0].astype(float)
            test_stats = f"Mixed ANOVA ➜ between: {p_val_between:.3f} - within: {p_val_within:.3f}{GG_note} - interaction: {p_val_inter:.3f}{GG_note}"
        else:
            variable_col = variable
            # TODO: implement the case of non-normality
            # If need to use non-parametric: use separate tests for the between and within effects
            test_results = {}
            test_stats = "[Non param] "
            if nb_groups == 2:
                # First, performing a Mann-Whitney test between the 2 groups (1 value per ID) to assess any difference
                group_1 = data[data[between] == groups_names[0]].groupby("ID")[variable].mean()
                group_2 = data[data[between] == groups_names[1]].groupby("ID")[variable].mean()
                test_results["mwu_stat"], test_results["mwu_p_val"] = mannwhitneyu(group_1, group_2, alternative="two-sided")
                p_val_between = test_results["mwu_p_val"]
                test_stats += f"Mann-Whitney U (between) ➜ {p_val_between:.3f} - Friedman (within) ➜ "
                # Using Friedman test to assess if there is a difference between the different measurements
                for group in groups_names:
                    if nb_subjects < 10 or len(measurements) < 6:
                        test_results[f"friedman_{group}"] = pg.friedman(data=data[data[between] == group], dv=variable, within=within, subject="ID", method="f")
                        p_val_within = test_results[f"friedman_{group}"]["p-unc"].iloc[0].astype(float)
                        test_stats += f"{group}(F): {p_val_within:.3f} "
                    else:
                        test_results[f"friedman_{group}"] = pg.friedman(data=data[data[between] == group], dv=variable, within=within, subject="ID", method="chisq")
                        p_val_within = test_results[f"friedman_{group}"]["p-unc"].iloc[0].astype(float)
                        test_stats += f"{group}(χ²): {p_val_within:.3f} "
                    test_stats += "/ " if group != groups_names[-1] else ""
            else:
                test_stats += "Implement this case"
                pass

        # Between which groups there is a difference, between which measurements in general and interaction of the 2 (T-tests / Mann-Whitney-U) TODO: Check for homogeneity as well
        post_hoc_dict["between"] = pg.pairwise_tests(data=data, dv=variable_col, between=between, within=within, subject="ID", padjust="bonf", parametric=normality)
        # For each group, test between which measurement there is a significant difference (Paired T-tests / Wilcoxon)
        for group in groups_names:
            post_hoc_dict[group] = pg.pairwise_tests(data=data[data[between] == group], dv=variable_col, within=within, subject="ID", padjust="bonf", parametric=normality) #OK, all assumptions checked
    else:
        if normality:
            # Performing the RM ANOVA
            test_results = pg.rm_anova(data=data, dv=variable_col, within=within, subject="ID")
            p_val_within = test_results["p-unc" if sphericity else "p-GG-corr"].iloc[0].astype(float)
            test_stats = f"RM ANOVA ➜ within: {p_val_within:.3f}{"[GG corrected]" if not sphericity else ""}"
        else:
            # Performing a Friedman Test
            if nb_subjects < 10 or len(measurements) < 6:
                test_results = pg.friedman(data=data, dv=variable_col, within=within, subject="ID", method="f")
                p_val_within = test_results["p-unc"].iloc[0].astype(float)
                test_stats = f"Friedman F Test ➜ within: {p_val_within:.3f}"
            else:
                test_results = pg.friedman(data=data, dv=variable_col, within=within, subject="ID", method="chisq")
                p_val_within = test_results["p-unc"].iloc[0].astype(float)
                test_stats = f"Friedman χ² Test ➜ within: {p_val_within:.3f}"
        # Posthocs tests (Paired T-tests / Wilcoxon)
        post_hoc_dict[groups_names[0]] = pg.pairwise_tests(data=data, dv=variable_col, within=within, subject="ID", padjust="bonf", parametric=normality)

    # Adding test results on the graph
    ax.text(x=0.01, y=0.97, s=test_stats, fontsize=8, transform=ax.transAxes, fontstyle="italic", color="gray", alpha=0.75)
    ax.text(x=0.01, y=0.95, s=f"{test_norm}", fontsize=8, transform=ax.transAxes, fontstyle="italic", color="green" if norm_pval > 0.05 and not transformation else ("teal" if transformation and norm_pval_trans > 0.05 else ("red" if normality else "orange")), alpha=0.5)
    if nb_groups > 1:
        ax.text(x=0.01, y=0.93, s=f"{test_homogen}", fontsize=8, transform=ax.transAxes, fontstyle="italic", color="green" if homogen_pval > 0.05 and not transformation else ("teal" if transformation and homogen_pval_trans > 0.05 else ("red" if homogeneity else "orange")), alpha=0.5)
    return test_results, post_hoc_dict


def normality_check(data, title="variable", plot=True):
    """
    Test whether the provided data follows a normal distribution or not. Perform a Shapiro-Wilk test, plot a histogram
    of the distribution and compare QQ-plot to 24 random generated datasets of the same size.

    Parameters
    ----------
    data : list/np.array of float
        The data for which the normality will be tested
    title : str
        A brief description of what is the data
    plot : bool
        Whether to display the plots or not

    Returns
    -------
    float
        The p_value resulting from the Shapiro-Wilk test
    """
    # Perform Shapiro-Wilk normality test
    data = np.array(data)
    stat, p_value = shapiro(data)
    if plot:
        sample_size = len(data)
        # Create a single figure with a GridSpec layout
        norm_fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        spec = norm_fig.add_gridspec(nrows=5, ncols=8)
        # --- Histogram (Top Left) ---
        ax_hist = norm_fig.add_subplot(spec[0:2, 0:3])
        ax_hist.hist(data, bins=15)
        ax_hist.set_ylabel("Frequency", fontsize=8)
        ax_hist.set_xlabel("Variable value", fontsize=8)
        ax_hist.set_title("Gaussian distribution?", fontsize=10)
        ax_hist.tick_params(axis='both', labelsize=8)
        # --- Single QQ-Plot (Bottom Left) ---
        ax_qq = norm_fig.add_subplot(spec[2:5, 0:3])
        sm.qqplot(data, line="s", ax=ax_qq)
        ax_qq.lines[0].set_markersize(5)
        ax_qq.lines[1].set_linewidth(2)
        ax_qq.text(0.05, 0.9, f"n = {sample_size}", transform=ax_qq.transAxes)
        ax_qq.set_title("Points aligned on the line?")
        ax_qq.set_ylabel(ax_qq.get_ylabel(), fontsize=8)
        ax_qq.set_xlabel(ax_qq.get_xlabel(), fontsize=8)
        ax_qq.tick_params(axis='both', labelsize=8)
        # --- 5×5 Grid of QQ-Plots (Right) ---
        random_datasets = [np.random.normal(loc=0, scale=1, size=sample_size) for _ in range(24)]
        random_datasets.insert(12, data)  # Insert real data in center
        axes_grid = []
        for row in range(5):
            for col in range(3, 8):
                ax_rnd = norm_fig.add_subplot(spec[row, col])
                sm.qqplot(random_datasets.pop(0), line="s", ax=ax_rnd)
                ax_rnd.lines[0].set_markersize(5)
                ax_rnd.lines[1].set_linewidth(2)
                ax_rnd.set_ylabel(ax_rnd.get_ylabel(), fontsize=8)
                ax_rnd.set_xlabel(ax_rnd.get_xlabel(), fontsize=8)
                ax_rnd.set_xticks([])
                ax_rnd.set_yticks([])
                ax_rnd.set_frame_on(False)
                # Add labels ('a' to 'x', 'Real' for actual data)
                index = row * 5 + (col - 2)
                label = chr(97 + index - 1) if index != 13 else "Real"
                ax_rnd.text(0.05, 0.9, label, transform=ax_rnd.transAxes, fontsize=12)
                axes_grid.append(ax_rnd)
        norm_fig.suptitle(f"Normality check for {title} \nShapiro-Wilk p-value = {p_value:.3f}", fontsize=12)
        norm_fig.canvas.manager.set_window_title(f"Normality Check - {title}")
    return p_value


def homogeneity_check(predicted, residuals, title="variable"):
    standard_resid = (residuals - residuals.mean()) / residuals.std()
    sqrt_standard_resid = np.sqrt(np.abs(standard_resid))
    # Create a separate figure for the scale-location plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(predicted, sqrt_standard_resid, alpha=0.6, s=5)
    # Compute the lowess smoothing line
    smoothed = lowess(sqrt_standard_resid, predicted, frac=0.3)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linestyle="--", label="Lowess trend", lw=2)
    ax.set_xlabel("Fitted values", fontsize=8)
    ax.set_ylabel(r"$\sqrt{|Standardized\ Residuals|}$", fontsize=8)
    ax.set_title(f"Scale-Location Plot for {title}", fontsize=12)
    ax.tick_params(axis='both', labelsize=8)
    fig.canvas.manager.set_window_title(f"Homogeneity Check - {title}")
    plt.show()


if __name__ == "__main__":
    wt_dmso = [11, 15, 13, 14, 16, 18, 19, 15]       # WT DMSO
    wt_bms =  [11, 14, 16, 15, 15, 18, 19, 14]   # WT BMS
    ko_dmso = [45, 49, 42, 41, 49, 47]               # KO DMSO
    ko_bms =  [11, 21, 18, 19, 20, 22]           # KO BMS
    cond = [["WT", "DMSO"], ["WT", "BMS"], ["KO", "BMS"], ["KO", "DMSO"]]
    labs = ["Genotype", "Treatment"]
    fig, ax = plt.subplots(figsize=(8, 8))

    # dmso_bms(ax, wt_dmso, wt_bms, ko_dmso, ko_bms, "Variable", "Titre", ylim=[],
    #          colors=[wt_color, wt_bms_color, all_ko_color, all_ko_bms_color])
    boxplot(ax, ko_dmso, ko_bms, "ylabel", paired=True, title="", ylim=[0, 20], colors=[wt_color, wt_light_color])
    # plt.tight_layout()
    # plt.show()
    gp1 = [[1, 2, np.nan, 3], [1, np.nan, 3, 5], [1, 2, 3, 5]]
    gp2 = [[1, 2, 4, 3], [np.nan, 3, 5, 6], [1, np.NaN, np.NaN, 5]]
    boxplot_3_conditions(gp1, gp2)
