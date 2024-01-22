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

plt.rcParams['font.size'] = 40
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


def boxplot(wt, ko, ylabel):
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
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), sharey=True)
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
    max_y = max(max(wt), max(ko))
    lim_max = max(int(max_y)*0.15, int(math.ceil(max_y / 2 + 0.5)) * 2)
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
    ax.plot([x_1, x_2], [y, y], lw=1.5, c=col)

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
    elif pval > 0.05:
        sig_symbol = 'ns'
    ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col)
    plt.tick_params(axis="x", which="both", bottom=False)
    plt.xticks([0.15, 0.40], ['', ""])
    plt.tight_layout()
    plt.show()


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
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    x1, x2 = 1.7, 1.8
    max_data = max([var_wt, var_ko])
    y, col = max_data + 0.15 * abs(max_data), 'k'
    ax.plot([x1, x2], [y, y], lw=1.5, c=col)

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
    elif pval > 0.05:
        sig_symbol = 'ns'
    ax.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col)
