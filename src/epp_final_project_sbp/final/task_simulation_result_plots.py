"""Tasks running the results formatting (tables, figures)."""

import pickle

import pandas as pd
import pytask

from epp_final_project_sbp.analysis import data_preparation as dp
from epp_final_project_sbp.config import (
    BETTING_STRATEGIES,
    FEATURES_CREATED,
    LEAGUES,
    path_to_plots,
    path_to_simulation_results,
)
from epp_final_project_sbp.final import plot as pl


def _create_parametrization(leagues, betting_strategies):
    id_to_kwargs = {}
    for league in leagues:
        profit_line_plots = {}
        path_simulation_data = path_to_simulation_results(league)
        for betting_strategy in betting_strategies:
            plot_name = f"profit_line_plots{league}_{betting_strategy}"
            plot_path = path_to_plots(plot_name)
            profit_line_plots[betting_strategy] = plot_path

        id_to_kwargs[league] = {
            "depends_on": {"datasource": path_simulation_data},
            "produces": {
                "profit_line_plots": profit_line_plots,
            },
        }

    return id_to_kwargs


_ID_TO_KWARGS = _create_parametrization(
    leagues=LEAGUES,
    betting_strategies=BETTING_STRATEGIES,
)


for id_, kwargs in _ID_TO_KWARGS.items():

    @pytask.mark.task(id=id_, kwargs=kwargs)
    def tasks_final_simulation_profits_plot(
        depends_on,
        produces,
        features_created=FEATURES_CREATED,
    ):
        """Plot the correlation and the boxplots for the respective league."""
        dp.get_league(depends_on["datasource"])
        data = pd.read_csv(depends_on["datasource"])
        with open(depends_on["datasource"], "rb") as f:
            pickle.load(f)
        for betting_strategy in produces["profit_line_plots"]:
            profit_plot = pl.plot_profits_lineplot(
                data=data,
                betting_strategy=produces["profit_line_plots"][betting_strategy],
            )
            profit_plot.savefig(produces["profit_line_plots"][betting_strategy])
