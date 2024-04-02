"""
01/11/2024
Adrien Corniere

Stats related plot functions like boxplot, barplot...
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss
import math
from matplotlib.ticker import AutoMinorLocator

plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams["xtick.major.width"] = 3
plt.rcParams["ytick.major.width"] = 3
mpl.use("Qt5Agg")
wt_color = "#326993"
light_wt_color = "#8db7d8"
ko_color = "#CC0000"
light_ko_color = "#ff8080"


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
    lw = 5
    ax.set_ylabel(ylabel)
    ax.boxplot([wt], positions=[0.15], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=wt_color, markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=wt_color),
               capprops=dict(linewidth=lw, color=wt_color),
               whiskerprops=dict(linewidth=lw, color=wt_color),
               medianprops=dict(linewidth=lw, color=wt_color), )
    ax.boxplot([ko], positions=[0.40], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=ko_color, markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=ko_color),
               capprops=dict(linewidth=lw, color=ko_color),
               whiskerprops=dict(linewidth=lw, color=ko_color),
               medianprops=dict(linewidth=lw, color=ko_color), )
    y = wt
    x = np.random.normal(0.15, 0.02, size=len(y))
    y1 = ko
    x1 = np.random.normal(0.40, 0.02, size=len(y1))
    ax.plot(x, y, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=wt_color, markeredgewidth=4)
    ax.plot(x1, y1, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=ko_color, markeredgewidth=4)
    ax.grid(False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=6, width=3)
    ax.tick_params(axis='both', which='minor', length=4, width=3)
    if len(ylim)!=0:
        ax.set_ylim(ylim)
    else:
        max_y = max(max(wt), max(ko))
        lim_max = max(int(max_y*0.15 + max_y), int(math.ceil(max_y / 2 )) * 2)
        min_y = min(min(wt), min(ko))
        lim_inf = min(0, min_y + 0.15*min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])

    x_1, x_2 = [0.15, 0.40]
    max_data = max([max(wt), max(ko)])
    y, col = max_data + 0.05 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col)

    def stat_boxplot(sb_wt, sb_ko, ylabel):
        print(ylabel)
        data_wt = sb_wt
        data_ko = sb_ko
        print(ss.shapiro(data_wt))
        print(ss.shapiro(data_ko))
        stat, pvalue_WT = ss.shapiro(data_wt)
        stat, pvalue_KO = ss.shapiro(data_ko)
        if pvalue_WT > 0.05 and pvalue_KO > 0.05:
            stat, pvalue = ss.ttest_ind(data_wt, data_ko)
            print(ss.ttest_ind(data_wt, data_ko))
        else:
            stat, pvalue = ss.mannwhitneyu(data_wt, data_ko)
            print(ss.mannwhitneyu(data_wt, data_ko))
        return pvalue
    pval = stat_boxplot(wt, ko, ylabel)
    if pval < 0.001:
        sig_symbol = '***'
    elif pval < 0.01:
        sig_symbol = '**'
    elif pval < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col)
    # plt.tick_params(axis="x", which="both", bottom=False)
    # plt.xticks([0.15, 0.40], ['', ""])
    # plt.tight_layout()
    # plt.show()


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
    plt.show()
    ax.grid(False)
    ax.set_ylabel(ylabel + " Var ")
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=6, width=3)
    ax.tick_params(axis='both', which='minor', length=4, width=3)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    x1, x2 = 1.7, 1.8
    max_data = max([var_wt, var_ko])
    y, col = max_data + 0.15 * abs(max_data), 'k'
    ax.plot([x1, x2], [y, y], lw=3, c=col)

    def stat_varplot(s_wt, s_ko, s_y_label):
        """
        add stat on the barplot
        Parameters
        ----------
        s_wt : numpy.ndarray, series, list
            data of the wt group
        s_ko : numpy.ndarray, series, list
            data of the ko group
        s_y_label : string
            columns names

        """
        print(s_y_label)
        data_wt = s_wt
        data_ko = s_ko
        print(ss.shapiro(data_wt))
        print(ss.shapiro(data_ko))
        stat, pvalue_wt = ss.shapiro(data_wt)
        stat, pvalue_ko = ss.shapiro(data_ko)
        if pvalue_wt > 0.05 and pvalue_ko > 0.05:
            stat, pvalue = ss.bartlett(data_wt, data_ko)
            print(ss.bartlett(data_wt, data_ko))
        else:
            stat, pvalue = ss.levene(data_wt, data_ko)
            print(ss.levene(data_wt, data_ko))
        return pvalue

    pval = stat_varplot(wt, ko, ylabel)
    if pval < 0.001:
        sig_symbol = '***'
    elif pval < 0.01:
        sig_symbol = '**'
    elif pval < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    ax.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col)


def paired_boxplot(ax, det, undet, ylabel, title, ylim=[]):
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

    """
    print("Boxplot plotting.")
    lw = 5
    ax.set_ylabel(ylabel)
    ax.boxplot([det], positions=[0.15], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=light_ko_color, markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=light_ko_color),
               capprops=dict(linewidth=lw, color=light_ko_color),
               whiskerprops=dict(linewidth=lw, color=light_ko_color),
               medianprops=dict(linewidth=lw, color=light_ko_color))
    ax.boxplot([undet], positions=[0.40], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=ko_color, markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=ko_color),
               capprops=dict(linewidth=lw, color=ko_color),
               whiskerprops=dict(linewidth=lw, color=ko_color),
               medianprops=dict(linewidth=lw, color=ko_color))
    for i in range(len(det)):
        ax.plot([0.15, 0.40], [det[i], undet[i]], marker="o", color=light_ko_color, alpha=0.9, linewidth=1.5,
        markersize=10, markeredgewidth=2, markeredgecolor=ko_color, markerfacecolor=light_ko_color)

    ax.grid(False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=6, width=3)
    ax.tick_params(axis='both', which='minor', length=4, width=3)
    max_y = max(max(det), max(undet))
    if len(ylim)!=0:
        ax.set_ylim(ylim)
    else:
        max_y = max(max(det), max(undet))
        lim_max = max(int(max_y*0.15 + max_y), int(math.ceil(max_y / 2 )) * 2)
        min_y = min(min(det), min(undet))
        lim_inf = min(0, min_y + 0.15*min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])

    x_1, x_2 = [0.15, 0.40]
    max_data = max([max(det), max(undet)])
    y, col = max_data + 0.05 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col)

    def stat_paired_boxplot(sb_det, sb_undet, ylabel):
        print(ylabel)
        data_det = sb_det
        data_undet = sb_undet
        print(ss.shapiro(data_det))
        print(ss.shapiro(data_undet))
        stat, pvalue_det = ss.shapiro(data_det)
        stat, pvalue_undet = ss.shapiro(data_undet)
        if pvalue_det > 0.05 and pvalue_undet > 0.05:
            stat, pvalue = ss.ttest_rel(data_det, data_undet)
            print(ss.ttest_rel(data_det, data_undet))
        else:
            stat, pvalue = ss.wilcoxon(data_det, data_undet)
            print(ss.wilcoxon(data_det, data_undet))
        return pvalue
    pval = stat_paired_boxplot(det, undet, ylabel)
    if pval < 0.001:
        sig_symbol = '***'
    elif pval < 0.01:
        sig_symbol = '**'
    elif pval < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col)
    ax.set_xticks([0.15, 0.40], ['', ""])
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.set_title(title)


def boxplot_anova(group1_data, group2_data, group3_data, lim_y, label_y, filename, color1, color2, color3, annot_text=[]
                  , title="", thickformater=True):

    fig = plt.figure(figsize=(6, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.Axes(fig, [0., 0.5, 1., 1.])
    ax1 = fig.add_subplot(1, 1, 1, title=title)
    band = [0, 0.8]
    plt.xlim(band)
    linewidth = 5
    a = ax1.boxplot([group1_data, group2_data, group3_data],
                    positions=[0.15, 0.40, 0.65],
                    showfliers=False,
                    widths=0.2,
                    boxprops=dict(linewidth=linewidth, color=color2),
                    whiskerprops=dict(color=color2, linewidth=linewidth),
                    capprops=dict(color=color2, linewidth=linewidth),
                    medianprops=dict(color=color2, linewidth=linewidth),
                    meanline=True,
                    showmeans=True)
    a["boxes"][0].set(color=color1, linewidth=linewidth)
    a["boxes"][2].set(color=color3, linewidth=linewidth)
    a["whiskers"][0].set(color=color1, linewidth=linewidth)
    a["whiskers"][1].set(color=color1, linewidth=linewidth)
    a["whiskers"][4].set(color=color3, linewidth=linewidth)
    a["whiskers"][5].set(color=color3, linewidth=linewidth)
    a["caps"][0].set(color=color1, linewidth=linewidth)
    a["caps"][1].set(color=color1, linewidth=linewidth)
    a["caps"][4].set(color=color3, linewidth=linewidth)
    a["caps"][5].set(color=color3, linewidth=linewidth)
    a["medians"][0].set(color=color1, linewidth=linewidth)
    a["medians"][1].set(color=color2, linewidth=linewidth)
    a["medians"][2].set(color=color3, linewidth=linewidth)
    a["means"][0].set(linewidth=linewidth)
    a["means"][1].set(linewidth=linewidth)
    a["means"][2].set(linewidth=linewidth)
    plt.xticks([0.15, 0.40], ['', ""])
    plt.ylim(lim_y)
    plt.ylabel(label_y)
    ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    y = group1_data
    x = np.random.normal(0.15, 0.02, size=len(y))
    y1 = group2_data
    x1 = np.random.normal(0.40, 0.02, size=len(y1))
    y2 = group3_data
    x2 = np.random.normal(0.65, 0.02, size=len(y2))
    ax1.plot(x, y, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color1, markeredgewidth=4)
    ax1.plot(x1, y1, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color2, markeredgewidth=4)
    ax1.plot(x2, y2, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=color3, markeredgewidth=4)
    ax1.tick_params(axis='both', labelsize=35)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(which='both', width=4)
    ax1.tick_params(which='major', length=10)
    ax1.tick_params(which='minor', length=8)
    if thickformater:
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    plt.tick_params(axis="x", which="both", bottom=False, top=False)
    ax1.yaxis.labelpad = 10
    plt.subplots_adjust(left=None, bottom=0.2, right=0.99, top=0.9, wspace=None, hspace=None)
    # stats and annotations
    print(" ")
    print("##############################")
    print(label_y)
    print(np.mean(group1_data))
    print(np.mean(group2_data))
    print(np.mean(group3_data))
    # if sc.shapiro(group1_data)

    x1, x2, x3 = 0.15, 0.40, 0.65
    max_d = np.concatenate([group1_data, group2_data, group3_data]).max()
    y, h, col = max_d + abs(0.10 * max_d), 0.025 * abs(max_d), 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=3, c=col)
    plt.plot([x2, x2, x3, x3], [y + + 3 * h, y + 4 * h, y + 4 * h, y + 3 * h], lw=3, c=col)
    plt.plot([x1, x1, x3, x3], [y + 7 * h, y + 8 * h, y + 8 * h, y + 7 * h], lw=3, c=col)

    plt.text((x1 + x2) * .5, y + h, annot_text[0], ha='center', va='bottom', color=col, weight='bold')
    plt.text((x2 + x3) * .5, y + 4 * h, annot_text[2], ha='center', va='bottom', color=col, weight='bold')
    plt.text((x1 + x3) * .5, y + 8 * h, annot_text[1], ha='center', va='bottom', color=col, weight='bold')
    fig.tight_layout()
    # fig.savefig(filename)