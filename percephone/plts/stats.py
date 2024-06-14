"""
01/11/2024
Adrien Corniere

Stats related plot functions like boxplot, barplot...
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import math
from percephone.plts.style import *
from percephone.plts.utils import *

mpl.rcParams["axes.grid"] = False
mpl.rcParams['font.size'] = 35
font_signif = mpl.rcParams['font.size'] / 2
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['lines.linewidth'] = 5

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

mpl.use("Qt5Agg")

wt_color = "#326993"
light_wt_color = "#8db7d8"
ko_color = "firebrick"#CC0000"
light_ko_color = "#e49689" # "#ff8080"


@boxplot_style
def boxplot(ax, wt, ko, ylabel, ylim=[]):
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
    print("Boxplot plotting.")
    ko_nan = np.array(ko)[~np.isnan(ko)]
    wt_nan = np.array(wt)[~np.isnan(wt)]
    ax.set_ylabel(ylabel)
    ax.boxplot(wt_nan, positions=[0.15], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=wt_color, markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=wt_color),
               capprops=dict(color=wt_color),
               whiskerprops=dict(color=wt_color),
               medianprops=dict(color=wt_color))
    ax.boxplot(ko_nan, positions=[0.40], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=ko_color, markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=ko_color),
               capprops=dict(color=ko_color),
               whiskerprops=dict(color=ko_color),
               medianprops=dict(color=ko_color))
    y = wt_nan
    x = np.random.normal(0.15, 0.02, size=len(y))
    y1 = ko_nan
    x1 = np.random.normal(0.40, 0.02, size=len(y1))
    ax.plot(x, y, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=wt_color, markeredgewidth=4)
    ax.plot(x1, y1, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=ko_color, markeredgewidth=4)
    ax.set_title(None)
    ax.set_xlabel(None)

    if len(ylim) != 0:
        ax.set_ylim(ylim)
    else:
        max_y = max(np.nanmax(wt), np.nanmax(ko))
        lim_max = max(int(max_y * 0.15 + max_y), int(math.ceil(max_y / 2)) * 2)
        min_y = min(np.nanmin(wt), np.nanmin(ko))
        lim_inf = min(0, min_y + 0.15 * min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.set_xticks([])

    x_1, x_2 = [0.15, 0.40]
    max_data = max([np.nanmax(wt), np.nanmax(ko)])
    y, col = max_data + 0.10 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col, clip_on=False)

    try:
        pval = stat_boxplot(wt, ko, ylabel, paired=False)
        sig_symbol = symbol_pval(pval)
    except ValueError:
        sig_symbol = "N.A."


    ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_signif)


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
    ax.bar([1.7, 1.8], width=0.05, height=y, color=[light_wt_color, light_ko_color], edgecolor=[wt_color, ko_color],
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

@boxplot_style
def paired_boxplot(ax, det, undet, ylabel, title, ylim=[], colors=[ko_color, light_ko_color], allow_stats_skip=False, variant = False):
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
def dmso_bms(ax, wt_dmso, wt_bms, ko_dmso, ko_bms, ylabel, title, ylim=[], colors=[wt_color, light_wt_color, ko_color, light_ko_color]):
    print("Boxplot plotting.")
    wt_dmso_nan = np.array(wt_dmso)[~np.isnan(wt_dmso)]
    wt_bms_nan = np.array(wt_bms)[~np.isnan(wt_bms)]
    ko_dmso_nan = np.array(ko_dmso)[~np.isnan(ko_dmso)]
    ko_bms_nan = np.array(ko_bms)[~np.isnan(ko_bms)]
    x_1, x_2, x_3, x_4 = [0.05, 0.30, 0.60, 0.85]
    ax.set_ylabel(ylabel)
    ax.boxplot(wt_dmso_nan, positions=[x_1], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=colors[0], markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=colors[0]),
               capprops=dict(color=colors[0]),
               whiskerprops=dict(color=colors[0]),
               medianprops=dict(color=colors[0]))
    ax.boxplot(wt_bms_nan, positions=[x_2], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=colors[1], markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=colors[1]),
               capprops=dict(color=colors[1]),
               whiskerprops=dict(color=colors[1]),
               medianprops=dict(color=colors[1]))
    for i in range(len(wt_dmso)):
        ax.plot([x_1, x_2], [wt_dmso[i], wt_bms[i]], marker="o", color=colors[1], alpha=0.9, linewidth=1.5,
                markersize=10, markeredgewidth=2, markeredgecolor=colors[0], markerfacecolor=colors[1])

    ax.boxplot(ko_dmso_nan, positions=[x_3], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=colors[2], markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=colors[2]),
               capprops=dict(color=colors[2]),
               whiskerprops=dict(color=colors[2]),
               medianprops=dict(color=colors[2]))
    ax.boxplot(ko_bms_nan, positions=[x_4], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=colors[3], markeredgecolor='black'),
               boxprops=dict(facecolor='white', color=colors[3]),
               capprops=dict(color=colors[3]),
               whiskerprops=dict(color=colors[3]),
               medianprops=dict(color=colors[3]))
    for j in range(len(ko_dmso)):
        ax.plot([x_3, x_4], [ko_dmso[j], ko_bms[j]], marker="o", color=colors[3], alpha=0.9, linewidth=1.5,
                markersize=10, markeredgewidth=2, markeredgecolor=colors[2], markerfacecolor=colors[3])

    ax.set_xlabel(None)
    max_y = max(np.nanmax(wt_dmso), np.nanmax(wt_bms), np.nanmax(ko_dmso), np.nanmax(ko_bms))
    if len(ylim) != 0:
        ax.set_ylim(ylim)
    else:
        lim_max = max(int(max_y * 0.15 + max_y), int(math.ceil(max_y / 2)) * 2)
        min_y = min(np.nanmin(wt_dmso), np.nanmin(wt_bms))
        lim_inf = min(0, min_y + 0.15 * min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.set_xticks([])

    max_data = max([np.nanmax(wt_dmso), np.nanmax(wt_bms), np.nanmax(ko_dmso), np.nanmax(ko_bms)])
    y, col = max_data + 0.10 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col)
    ax.plot([x_3, x_4], [y, y], lw=3, c=col)

    pval = stat_boxplot(wt_dmso, wt_bms, ylabel, paired=True)
    sig_symbol = symbol_pval(pval)
    ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_signif)

    pval_2 = stat_boxplot(ko_dmso, ko_bms, ylabel, paired=True)
    sig_symbol_2 = symbol_pval(pval_2)
    ax.text((x_3 + x_4) * 0.5, y, sig_symbol_2, ha='center', va='bottom', c=col, fontsize=font_signif)


    ax.set_xticks([x_1, x_2, x_3, x_4], ["WT-DMSO", "WT-BMS", "KO-DMSO", "KO-BMS"])
    ax.tick_params(axis="x", which="both", bottom=False, labelrotation=45)
    ax.set_title(title)
    ax.tick_params(axis='y')


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
                         color2=ko_color,
                         legend_labels=None,
                         filename=None):
    """
    group1_data: list of 3 lists, one list per condition
    group2_data:  list of 3 lists, one list per condition
    Returns
    -------
    object
    """
    linewidth = 5

    fig, axs = plt.subplots(1, 3, figsize=(14, 8), sharey="all")
    for i, ax in enumerate(axs.flat):
        bx = ax.boxplot([group1_data[i], group2_data[i]],
                        positions=[0.15, 0.40],
                        showfliers=False,
                        widths=0.2,
                        boxprops=dict(linewidth=linewidth, color=color2),
                        whiskerprops=dict(color=color2, linewidth=linewidth),
                        capprops=dict(color=color2, linewidth=linewidth),
                        medianprops=dict(color=color2, linewidth=linewidth),
                        meanline=True,
                        showmeans=True)
        bx["boxes"][0].set(color=color1, linewidth=linewidth)
        bx["whiskers"][0].set(color=color1, linewidth=linewidth)
        bx["whiskers"][1].set(color=color1, linewidth=linewidth)
        bx["caps"][0].set(color=color1, linewidth=linewidth)
        bx["caps"][1].set(color=color1, linewidth=linewidth)
        bx["medians"][0].set(color=color1, linewidth=linewidth)
        bx["means"][0].set(linewidth=linewidth)
        bx["means"][1].set(linewidth=linewidth)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.set_xticks([0.15, 0.40], ['', ""])
        ax.set_xlabel(cond_labels[i])
        ax.grid(False)
        ax.set_facecolor("white")
        y = group1_data[i]
        x = np.random.normal(0.15, 0.02, size=len(y))
        y1 = group2_data[i]
        x1 = np.random.normal(0.40, 0.02, size=len(y1))
        ax.plot(x1, y1, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color2, markeredgewidth=4)
        ax.plot(x, y, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color1, markeredgewidth=4)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis="y", which="both", left=False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, left=True)
        pval = stat_boxplot(group1_data[i], group2_data[i], f"{cond_labels[i]} group comp")
        sig_symbol = symbol_pval(pval)

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


if __name__ == "__main__":
    wt_dmso = [11, 15, 13, 14, 16, 18, 19, 15]       # WT DMSO
    wt_bms = [11, 14, 16, 15, 15, 18, 19, 14]   # WT BMS
    ko_dmso = [48, 49, 49, 41, 49, 47]               # KO DMSO
    ko_bms = [20, 21, 18, 19, 20, 22]           # KO BMS
    cond = [["WT", "DMSO"], ["WT", "BMS"], ["KO", "BMS"], ["KO", "DMSO"]]
    labs = ["Genotype", "Treatment"]
    fig, ax = plt.subplots(figsize=(8, 8))

    paired_boxplot(ax, wt_dmso, wt_bms, "Variable", "Titre", ylim=[],
             colors=[light_ko_color, ko_color], variant=True)
    # boxplot(ax, wt_dmso, wt_bms, ylabel="Test")
    plt.tight_layout()
    plt.show()
