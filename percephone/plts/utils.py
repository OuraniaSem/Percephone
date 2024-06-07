"""
Théo Gauvrit 18/01/2024
Utility function for plots
"""
import scipy.stats as ss
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
    if pvalue_n1 > 0.05 and pvalue_n2 > 0.05:  # Normality of the samples
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
    else:  # Non-Normality of the samples
        if paired:
            pvalue = ss.wilcoxon(group_1, group_2).pvalue
            print(ss.wilcoxon(group_1, group_2))
        else:
            pvalue = ss.mannwhitneyu(group_1, group_2).pvalue
            print(ss.mannwhitneyu(group_1, group_2))
    return pvalue


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


def build_aov_df(data, conditions, cond_labels):
    flat_data = []
    for group in data:
        flat_data.extend(group)
    df = {"Variable": flat_data}
    for cond_idx, label in enumerate(cond_labels):
        df[label] = []
        for gp_idx, group in enumerate(data):
            df[label].extend(len(group) * [conditions[gp_idx][cond_idx]])
    return pd.DataFrame(df)

def anova():
    pass

if __name__ == "__main__":
    group1 = [11, 15, 13, 14, 16, 18, 19]       # WT DMSO
    group2 = [20, 26, 23, 24, 29, 28, 29, 25]   # WT BMS
    group3 = [33, 39, 34, 31, 39, 36]           # KO BMS
    group4 = [48, 49, 49, 41, 49]               # KO DMSO
    cond = [["WT", "DMSO"], ["WT", "BMS"], ["KO", "BMS"], ["KO", "DMSO"]]
    labs = ["Genotype", "Treatment"]
    df = build_aov_df([group1, group2, group3, group4], cond, labs)
    print(df)

