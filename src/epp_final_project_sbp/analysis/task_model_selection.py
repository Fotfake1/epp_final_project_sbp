import pickle

import pandas as pd
import pytask

from epp_final_project_sbp.analysis import data_preparation as dp
from epp_final_project_sbp.analysis import model_selection as ms
from epp_final_project_sbp.config import (
    BLD,
    LEAGUES,
    MODELS,
    path_to_final_model,
    path_to_performance_store,
    path_to_processed_models,
)


def _create_parametrization(datasource, leagues, models):
    id_to_kwargs = {}
    for league in leagues:
        name = f"{league}"
        model_paths = {}
        for m in models:
            modelname = f"{league}_{m}"
            model_path = path_to_processed_models(modelname)
            model_paths[m] = model_path
        final_model_path = path_to_final_model(name)
        performance_store = path_to_performance_store(name)
        id_to_kwargs[name] = {
            "depends_on": {"datasource": datasource, "models": model_paths},
            "produces": {
                "model": final_model_path,
                "performance_store": performance_store,
            },
        }

    return id_to_kwargs


_ID_TO_KWARGS = _create_parametrization(
    datasource=BLD / "python" / "data" / "data_features_added.csv",
    leagues=LEAGUES,
    models=MODELS,
)

for id_, kwargs in _ID_TO_KWARGS.items():

    @pytask.mark.task(id=id_, kwargs=kwargs)
    def task_model_selection(depends_on, produces):
        """task creates all models, which are considered in this project."""
        data = pd.read_csv(depends_on["datasource"])
        league = dp.get_league(
            depends_on["models"][list(depends_on["models"].keys())[0]],
        )
        data = dp.data_preparation(data=data, league=league)
        rf_model_cv = pickle.load(open(depends_on["models"]["RF_model"], "rb"))
        knn_model = pickle.load(open(depends_on["models"]["KNN_model"], "rb"))
        logit_model = pickle.load(open(depends_on["models"]["LOGIT_model"], "rb"))

        model, performances = ms.model_selection(
            data=data,
            rf_model_cv=rf_model_cv,
            knn_model=knn_model,
            logit_model=logit_model,
        )

        with open(produces["model"], "wb") as f:
            pickle.dump(model, f)
        with open(produces["performance_store"], "wb") as f:
            pickle.dump(performances, f)
