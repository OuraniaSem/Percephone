"""
01/11/2024
Adrien Corniere

Stats related plot functions like boxplot, barplot...
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math
from matplotlib.ticker import AutoMinorLocator

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
font_s = 30


def symbol_pval(pval):
    """
    Returns the significance symbol associated with the given pvalue.

    Parameters
    ----------
    pval : float
        The p-value used to determine the significance level.

    Returns
    -------
    sig_symbol : str
        The symbol representing the significance level. '***' denotes extremely significant,
        '**' denotes very significant, '*' denotes significant, and 'n.s' denotes not significant.
    """
    if pval < 0.001:
        sig_symbol = '***'
    elif pval < 0.01:
        sig_symbol = '**'
    elif pval < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'n.s'
    return sig_symbol


def stat_boxplot(group_1, group_2, ylabel, paired=False):
    """
    Returns the p-value for the comparison between 2 independant or paired sample's distribution.

    Parameters
    ----------
    group_1 : array_like
        The data for the first group.
    group_2 : array_like
        The data for the second group.
    ylabel : str
        The label for the y-axis of the boxplot.
    paired : bool, optional
        Boolean indicating if the 2 samples are paired or not

    Returns
    -------
    pvalue : float
        The p-value resulting from the statistical test.

    Notes
    -----
    * The threshold pvalue used for tests is 0.05.
    * The Shapiro-Wilk test is used to test the normality of the distribution of each sample.
    * For independant samples:
        * If the normality can be assumed, the Levene test is used to test the equality of the samples' variance. If the
    variances are equal a standard t-test is used, otherwise, a Welch's t-test is performed.
        * If the normality can't be assumed, a Mann-Whitney U test is performed.
    * For paired samples: If the normality can be assumed a standard t-test is used, otherwise, a Wilcoxon signed-rank
    test is performed.
    """
    print(f"--- {ylabel} ---")
    print(ss.shapiro(group_1))
    print(ss.shapiro(group_2))
    # Normality of the distribution testing
    pvalue_n1 = ss.shapiro(group_1).pvalue
    pvalue_n2 = ss.shapiro(group_2).pvalue
    if pvalue_n1 > 0.05 and pvalue_n2 > 0.05:   # Normality of the samples
        if paired:
            pvalue = ss.ttest_rel(group_1, group_2).pvalue
            print(ss.ttest_rel(group_1, group_2))
        else:
            # Equality of the variances testing
            pvalue_v = ss.levene(group_1, group_2).pvalue
            print(ss.levene(group_1, group_2))
            if pvalue_v > 0.05:
                pvalue = ss.ttest_ind(group_1, group_2).pvalue
                print(f"Equal variances :{ss.ttest_ind(group_1, group_2)}")
            else:
                pvalue = ss.ttest_ind(group_1, group_2, equal_var=False).pvalue
                print(f"Unequal variances: {ss.ttest_ind(group_1, group_2)}")
    else:   # Non-Normality of the samples
        if paired:
            pvalue = ss.wilcoxon(group_1, group_2).pvalue
            print(ss.wilcoxon(group_1, group_2))
        else:
            pvalue = ss.mannwhitneyu(group_1, group_2).pvalue
            print(ss.mannwhitneyu(group_1, group_2))
    return pvalue


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
    lw = 5
    ax.set_ylabel(ylabel,fontsize=font_s)
    ax.boxplot(wt_nan, positions=[0.15], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=wt_color, markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=wt_color),
               capprops=dict(linewidth=lw, color=wt_color),
               whiskerprops=dict(linewidth=lw, color=wt_color),
               medianprops=dict(linewidth=lw, color=wt_color), )
    ax.boxplot(ko_nan, positions=[0.40], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=ko_color, markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=ko_color),
               capprops=dict(linewidth=lw, color=ko_color),
               whiskerprops=dict(linewidth=lw, color=ko_color),
               medianprops=dict(linewidth=lw, color=ko_color), )
    y = wt_nan
    x = np.random.normal(0.15, 0.02, size=len(y))
    y1 = ko_nan
    x1 = np.random.normal(0.40, 0.02, size=len(y1))
    ax.plot(x, y, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=wt_color, markeredgewidth=4)
    ax.plot(x1, y1, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=ko_color, markeredgewidth=4)
    ax.grid(False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_facecolor("white")
    ax.spines["left"].set_color("black")
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=8, width=3, color="black", left=True)
    ax.tick_params(axis='both', which='minor', length=6, width=3, color="black", left=True)
    if len(ylim)!=0:
        ax.set_ylim(ylim)
    else:
        max_y = max(np.nanmax(wt), np.nanmax(ko))
        lim_max = max(int(max_y*0.15 + max_y), int(math.ceil(max_y / 2)) * 2)
        min_y = min(np.nanmin(wt), np.nanmin(ko))
        lim_inf = min(0, min_y + 0.15*min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])

    x_1, x_2 = [0.15, 0.40]
    max_data = max([np.nanmax(wt), np.nanmax(ko)])
    y, col = max_data + 0.10 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col)

    pval = stat_boxplot(wt, ko, ylabel, paired=False)
    sig_symbol = symbol_pval(pval)

    ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_s-8, weight='bold')
    # ax.tick_params(axis='y', labelsize=font_s)
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
    ax.set_facecolor("white")
    ax.set_ylabel(ylabel + " Var ")
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=8, width=3)
    ax.tick_params(axis='both', which='minor', length=6, width=3)
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
    sig_symbol = symbol_pval(pval)

    ax.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, weight='bold')


def paired_boxplot(ax, det, undet, ylabel, title, ylim=[], colors=[ko_color,light_ko_color], allow_stats_skip=False):
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
    det_nan = np.array(det)[~np.isnan(det)]
    undet_nan = np.array(undet)[~np.isnan(undet)]
    lw = 5
    ax.set_ylabel(ylabel, fontsize=font_s)
    ax.boxplot(det_nan, positions=[0.15], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=colors[0], markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=colors[0]),
               capprops=dict(linewidth=lw, color=colors[0]),
               whiskerprops=dict(linewidth=lw, color=colors[0]),
               medianprops=dict(linewidth=lw, color=colors[0]))
    ax.boxplot(undet_nan, positions=[0.40], patch_artist=True, showfliers=False, widths=0.2,
               meanprops=dict(marker='o', markerfacecolor=colors[1], markeredgecolor='black'),
               boxprops=dict(linewidth=lw, facecolor='white', color=colors[1]),
               capprops=dict(linewidth=lw, color=colors[1]),
               whiskerprops=dict(linewidth=lw, color=colors[1]),
               medianprops=dict(linewidth=lw, color=colors[1]))
    for i in range(len(det)):
        ax.plot([0.15, 0.40], [det[i], undet[i]], marker="o", color=colors[1], alpha=0.9, linewidth=1.5,
        markersize=10, markeredgewidth=2, markeredgecolor=colors[0], markerfacecolor=colors[1])

    ax.grid(False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_facecolor("white")
    ax.spines["left"].set_color("black")
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=8, width=3, left=True)
    ax.tick_params(axis='both', which='minor', length=6, width=3, left=True)
    max_y = max(np.nanmax(det), np.nanmax(undet))
    if len(ylim)!=0:
        ax.set_ylim(ylim)
    else:
        max_y = max(np.nanmax(det), np.nanmax(undet))
        lim_max = max(int(max_y*0.15 + max_y), int(math.ceil(max_y / 2 )) * 2)
        min_y = min(np.nanmin(det), np.nanmin(undet))
        lim_inf = min(0, min_y + 0.15*min_y)
        ax.set_ylim(ymin=lim_inf, ymax=lim_max)
    yticks = list(ax.get_yticks())
    ax.set_yticks(sorted(yticks))
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])

    x_1, x_2 = [0.15, 0.40]
    max_data = max([np.nanmax(det), np.nanmax(undet)])
    y, col = max_data + 0.10 * abs(max_data), 'k'
    ax.plot([x_1, x_2], [y, y], lw=3, c=col)

    if allow_stats_skip:
        try:
            pval = stat_boxplot(det, undet, ylabel, paired=True)
            sig_symbol = symbol_pval(pval)
            ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_s - 8,
                    weight='bold')
        except ValueError:
            pass
    else:
        pval = stat_boxplot(det, undet, ylabel, paired=True)
        sig_symbol = symbol_pval(pval)
        ax.text((x_1 + x_2) * 0.5, y, sig_symbol, ha='center', va='bottom', c=col, fontsize=font_s-8, weight='bold')

    ax.set_xticks([0.15, 0.40], ['', ""])
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.set_title(title)
    ax.tick_params(axis='y', labelsize=font_s)

def stats_anova(*groups_data):
    # Perform one-way ANOVA
    f_stat, p_value = ss.f_oneway(*groups_data)
    print(f"Statistical F: {f_stat}")
    print(f"p-value: {p_value}")

    # Determine significance symbols
    stars = []

    # Check p-value and perform Tukey's HSD test if significant
    if p_value > 0.05:
        stars = ["n.s"] * len(groups_data)
    else:
        # Prepare data for Tukey's HSD test
        all_data = np.concatenate(groups_data)
        group_labels = [f"group{i}_data" for i in range(len(groups_data)) for _ in groups_data[i]]
        data = {'value': all_data, 'group': group_labels}
        df = pd.DataFrame(data)

        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['group'], alpha=0.05)
        print(tukey)
        tukey_p_values = tukey.pvalues

        # Determine significance symbols for each pairwise comparison
        stars = [symbol_pval(p_val) for p_val in tukey_p_values]
        print(stars)

    return stars

def calculate_distance(pos1, pos2):
    return abs(pos1 - pos2)


def boxplot_anova(groups_data, lim_y, label_y, filename, colors, annot_text=[],
                  title="", thickformater=True, show_only_significant=False):

    fig = plt.figure(figsize=(3*len(groups_data), 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.Axes(fig, [0., 0.5, 1., 1.])
    ax1 = fig.add_subplot(1, 1, 1, title=title)
    band = [0, 0.33*len(groups_data)]
    plt.xlim(band)
    linewidth = 5
    positions = np.linspace(0.20, band[1]-0.20, len(groups_data))
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
        a["whiskers"][i*2].set(color=colors[i], linewidth=linewidth)
        a["whiskers"][i*2+1].set(color=colors[i], linewidth=linewidth)
        a["caps"][i*2].set(color=colors[i], linewidth=linewidth)
        a["caps"][i*2+1].set(color=colors[i], linewidth=linewidth)
        a["medians"][i].set(color=colors[i], linewidth=linewidth)
        a["means"][i].set(linewidth=linewidth)

    plt.xticks(positions, [''] * len(groups_data))
    plt.ylim(lim_y)
    plt.ylabel(label_y)
    ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    for i, group_data in enumerate(groups_data):
        x = np.random.normal(positions[i], 0.02, size=len(group_data))
        ax1.plot(x, group_data, ".", alpha=0.5, ms=28, markerfacecolor='none', markeredgecolor=colors[i], markeredgewidth=4)

    ax1.tick_params(axis='both', labelsize=35)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(which='both', width=3)
    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=6)
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
        plt.text((pos1 + pos2) * .5, y + y_offset + fixed_star_distance, star, ha='center', va='bottom', color=col,
                 weight='bold')

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
        bx = ax.boxplot([group1_data[i],group2_data[i]],
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
        ax.set_xlabel(cond_labels[i], fontsize=font_s)
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
        axs[i].text((x1 + x2) * .5, y + h, sig_symbol, ha='center', va='bottom', color=col, weight='bold', fontsize=20)

    axs[0].set_ylabel(label_y, fontsize=font_s)
    axs[0].tick_params(axis='y', labelsize=font_s)
    axs[0].yaxis.set_visible(True)
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    if y_percent:
        axs[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axs[0].tick_params(which='both', width=4)
    axs[0].tick_params(which='major', length=10)
    axs[0].tick_params(which='minor', length=8)

    if legend_labels is not None:
        hB, = ax.plot([1, 1], wt_color)
        hR, = ax.plot([1, 1], ko_color)
        ax.legend((hB, hR), legend_labels)
        hB.set_visible(False)
        hR.set_visible(False)

    if lim_y != "auto":
        plt.ylim(lim_y)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.2, hspace=None)
    plt.yticks(fontsize=30)
    plt.suptitle(title)

    if filename is not None:
        fig.savefig(filename)
    plt.show()
    # fig.tight_layout(pad=0.1)


if __name__ == '__main__':
    group1 = [[12, 3, 4, 4, 8, 8, 8], [8, 8, 9, 12], [14, 10, 5, 8, 10]]
    group2 = [[8, 8, 9, 12], [12, 3, 4, 4, 8, 8, 8], [14, 10, 5, 8, 10]]
    # boxplot_3_conditions(group1, group2, ["Spe", "Sen", "Acc"],
    #                      legend_labels=("WT", "KO-Hypo"),
    #                      title="Nice plot",
    #                      y_percent=True)
    fig, ax = plt.subplots()
    wt = [1, 2, 5, 6, 9, 4, 3]
    ko = [7, 8, 4, 9, 6, 5, 6, 8]
    boxplot(ax, wt, ko, "test", ylim=[0, 10])
    plt.show()

