"""
01/11/2024
Adrien Corniere

Basic plot functions like boxplot, barplot...
"""
import core as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as ss
import os

plt.rcParams['font.size'] = 20
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
    fig, ax = plt.subplots(1, 1, figsize=(4, 6), sharey=True)
    ax.set_ylabel(ylabel)
    ax.boxplot([wt], positions=[1], patch_artist=True, showfliers=False,
               meanprops=dict(marker='o', markerfacecolor=wt_color, markeredgecolor='black'),
               boxprops=dict(linewidth=2.0, facecolor='white', color=wt_color),
               capprops=dict(linewidth=2.0, color=wt_color),
               whiskerprops=dict(linewidth=2.0, color=wt_color),
               medianprops=dict(linewidth=2.0, color=wt_color), )
    ax.boxplot([ko], positions=[2], patch_artist=True, showfliers=False,
               meanprops=dict(marker='o', markerfacecolor=ko_color, markeredgecolor='black'),
               boxprops=dict(linewidth=2.0, facecolor='white', color=ko_color),
               capprops=dict(linewidth=2.0, color=ko_color),
               whiskerprops=dict(linewidth=2.0, color=ko_color),
               medianprops=dict(linewidth=2.0, color=ko_color), )

    ax.grid(False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])
    plt.tight_layout()
    plt.show()


def barplot(wt,ko, ylabel):
    """
    create barplot for two data groups.

    Parameters
    ----------
    wt : numpy.ndarray, series, list
        data of the wt group,
    ko : numpy.ndarray, series, list
        data of the ko group,
    y_label : string
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
    ax.set_ylabel(col_name + " Var ")
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


if __name__ == '__main__':
    directory = ""
    roi_info = pd.read_excel(directory + "/FmKO_ROIs&inhibitory.xlsx")
    folders = os.listdir(directory)
    folder = folders[4]
    path = directory + folder + '/'
    rec = pc.RecordingAmplDet(path, 0, folder, roi_info, correction=False)
    roi_info.replace("NA", np.nan, inplace=True)
    for col_name in roi_info.columns[12:16]:
        y_wt = roi_info[col_name][roi_info['Genotype'] == 'WT'].dropna()
        y_ko = roi_info[col_name][roi_info['Genotype'] == 'KO'].dropna()
        boxplot(wt=y_wt, ko=y_ko, ylabel=col_name)
        barplot(wt=y_wt, ko=y_ko, ylabel=col_name)

