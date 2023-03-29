"""Functions for fitting the regression model."""
import pickle

import pandas as pd
import pytask

from epp_final_project_sbp.analysis import data_preparation as dp
from epp_final_project_sbp.config import (
    BLD,
    LEAGUES,
    SIMULATION_RELEVANT_COLUMNS,
    path_to_final_model,
    path_to_simulation_results,
)
from epp_final_project_sbp.simulation import betting_strategies as bs
from epp_final_project_sbp.simulation import simulation as sim


def _create_parametrization(datasource, leagues):
    id_to_kwargs = {}
    datasource = datasource
    for league in leagues:
        name = f"{league}"
        model = path_to_final_model(name)
        produces = path_to_simulation_results(name)
        id_to_kwargs[name] = {
            "depends_on": {"datasource": datasource, "model": model},
            "produces": produces,
        }

    return id_to_kwargs


_ID_TO_KWARGS = _create_parametrization(
    datasource=BLD / "python" / "data" / "data_features_added.csv",
    leagues=LEAGUES,
)

for id_, kwargs in _ID_TO_KWARGS.items():

    @pytask.mark.task(id=id_, kwargs=kwargs)
    def task_simulate_betting(depends_on, produces):
        data = pd.read_csv(depends_on["datasource"])
        league = dp.get_league(depends_on["model"])
        data = dp.data_preparation(data=data, league=league)

        model = pickle.load(open(depends_on["model"], "rb"))

        simulation_results = sim.simulate_forecasting(
            data=data,
            number_of_initial_training_dates=200,
            model=model,
        )
        simulation_results = simulation_results[SIMULATION_RELEVANT_COLUMNS]

        simulation_results = bs.compute_outcomes_betting_strategies(
            simulation_data=simulation_results,
            odds=SIMULATION_RELEVANT_COLUMNS,
        )

        with open(produces, "wb") as f:
            pickle.dump(simulation_results, f)
