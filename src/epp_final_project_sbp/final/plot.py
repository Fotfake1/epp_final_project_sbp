"""Functions plotting results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from epp_final_project_sbp.config import CORR_PLOT_VARIABLES, LABELS
from epp_final_project_sbp.simulation import betting_strategies as bs


def plot_correlation_matrix(data, labels=LABELS, corr_variables=CORR_PLOT_VARIABLES):
    data_corr = data[corr_variables]

    for label in corr_variables:
        data_corr = data_corr[data_corr[label] != -33]

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.triu(np.ones_like(data_corr.corr(), dtype=bool))
    fig, ax = plt.subplots()
    sns.heatmap(
        data_corr.corr(),
        cmap=cmap,
        center=0,
        mask=mask,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    return fig


def plot_boxplots(data, feature):
    fig, ax = plt.subplots()
    sns.boxplot(
        x=data[data[feature] != -33]["full_time_result"],
        y=data[data[feature] != -33][feature],
        data=data,
        ax=ax,
    )
    ax.set_xlabel("Full Time Results", fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    return fig


def plot_profits_lineplot(data, column):
    simulation_wins = bs.cumulative_sum(data=data, column=column)
    fig, ax = plt.subplots()
    sns.lineplot(data=simulation_wins, x=simulation_wins.index, y=column, ax=ax)
    ax.set_xlabel(fontsize=10)
    ax.set_ylabel(fontsize=10)
    return fig
