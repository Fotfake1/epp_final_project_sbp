"""Tasks running the results formatting (tables, figures)."""

import pickle

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
        path_simulation_data = path_to_simulation_results(league)
        for betting_strategy in betting_strategies:
            name = f"{league}_{betting_strategy}"
            plot_name = f"profit_line_plots{league}_{betting_strategy}_model_forecast"
            plot_path = path_to_plots(plot_name)

            id_to_kwargs[name] = {
                "depends_on": path_simulation_data,
                "produces": plot_path,
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
        betting_strategies=BETTING_STRATEGIES,
    ):
        """Plot the correlation and the boxplots for the respective league."""
        with open(depends_on, "rb") as f:
            data = pickle.load(f)
        betting_strategy = dp.get_betting_strategy(
            betting_strategies=betting_strategies,
            path=produces,
        )
        profit_plot = pl.plot_profits_lineplot(
            data=data,
            column=betting_strategy,
        )
        profit_plot.savefig(produces)
