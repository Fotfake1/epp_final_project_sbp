"""Functions for fitting the regression model."""
import pickle

import pandas as pd
import pytask
from sklearn.model_selection import (
    TimeSeriesSplit,
)

from epp_final_project_sbp.analysis import data_preparation as dp
from epp_final_project_sbp.analysis import model_build as mb
from epp_final_project_sbp.config import (
    BLD,
    LEAGUES,
    MODELS,
    N_SPLIT,
    TRAIN_SHARE,
    path_to_processed_models,
)


def _create_parametrization(datasource, leagues, models):
    id_to_kwargs = {}
    depends_on = datasource
    try:
        for league in leagues:
            for model in models:
                name = f"{league}_{model}"
                produces = path_to_processed_models(name)
                id_to_kwargs[name] = {"depends_on": depends_on, "produces": produces}
    except AssertionError:
        "The parametrization is gone wrong."

    return id_to_kwargs


_ID_TO_KWARGS = _create_parametrization(
    datasource=BLD / "python" / "data" / "data_features_added.csv",
    leagues=LEAGUES,
    models=MODELS,
)

for id_, kwargs in _ID_TO_KWARGS.items():

    @pytask.mark.task(id=id_, kwargs=kwargs)
    def task_compute_models(depends_on, produces):
        """task creates all models, which are considered in this project.

        The models are saved as pickle files. All models are computed with the same
        data and a time series cross validation splitter.
        Input:
            data_features_added.csv: csv file with the data with the features added.
        Output:
            models: pickle files with the models computed.

        """
        data = pd.read_csv(depends_on)
        league = dp.get_league(string=produces)
        model = dp.get_model_name(produces=produces)

        data = dp.data_preparation(data=data, league=league)
        tscv = TimeSeriesSplit(n_splits=N_SPLIT, max_train_size=None, test_size=None)

        train_data, test_data = dp.split_data(data=data, train_share=TRAIN_SHARE)
        model = mb.get_model_computed(model=model, data=train_data, tscv=tscv)

        with open(produces, "wb") as f:
            pickle.dump(model, f)
