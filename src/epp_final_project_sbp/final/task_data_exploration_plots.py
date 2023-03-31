"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from epp_final_project_sbp.analysis import data_preparation as dp
from epp_final_project_sbp.config import (
    BLD,
    FEATURES_CREATED,
    LEAGUES,
    path_to_plots,
)
from epp_final_project_sbp.final import plot as pl


def _create_parametrization(datasource, leagues, features_created):
    id_to_kwargs = {}
    for league in leagues:
        boxplots_paths = {}
        corr_plot_name = f"corr_plot_{league}"
        corr_plot_path = path_to_plots(corr_plot_name)
        for feature in features_created:
            plot_name = f"boxplot_{league}_{feature}"
            plot_path = path_to_plots(plot_name)
            boxplots_paths[feature] = plot_path
        id_to_kwargs[league] = {
            "depends_on": {"datasource": datasource},
            "produces": {
                "corr_plot": corr_plot_path,
                "boxplots": boxplots_paths,
            },
        }

    return id_to_kwargs


_ID_TO_KWARGS = _create_parametrization(
    datasource=BLD / "python" / "data" / "data_features_added.csv",
    leagues=LEAGUES,
    features_created=FEATURES_CREATED,
)


for id_, kwargs in _ID_TO_KWARGS.items():

    @pytask.mark.task(id=id_, kwargs=kwargs)
    def task_plots_initial_data_exploration(
        depends_on,
        produces,
        features_created=FEATURES_CREATED,
    ):
        """Plot the correlation and the boxplots for the respective league.

        Input:
            data_features_added.csv: csv file with the data with the features added.
        Output:
            corr_plot: png file with the correlation plot.

        """
        league = dp.get_league(produces["corr_plot"])
        data = pd.read_csv(depends_on["datasource"])

        data = dp.data_preparation(data=data, league=league)
        corr_plot = pl.plot_correlation_matrix(data=data)
        for feature in features_created:
            boxplot = pl.plot_boxplots(data=data, feature=feature)
            boxplot.savefig(produces["boxplots"][feature])
        corr_plot.savefig(produces["corr_plot"], bbox_inches="tight")
