import pandas as pd
import pytest
from epp_final_project_sbp.analysis.model_build import get_model_computed
from epp_final_project_sbp.config import (
    BLD,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
)


@pytest.fixture()
def model():
    model = "RF"
    return model


@pytest.fixture()
def data():
    data = pd.DataFrame(
        pd.read_csv(BLD / "python" / "data" / "data_features_added.csv"),
    )
    return data


@pytest.fixture()
def tscv():
    tscv = TimeSeriesSplit(n_splits=5)
    return tscv


def test_model_built_wrong_inputs(data=data, model=model, tscv=tscv, league="E0"):
    with pytest.raises(AssertionError):
        data_false = [1, 2, 3, 4]
        get_model_computed(model=model, data=data_false, tscv=tscv)

    with pytest.raises(AssertionError):
        tscv_false = "TimeSeriesSplit"
        get_model_computed(model=model, data=data, tscv=tscv_false)

    with pytest.raises(AssertionError):
        model_false = "SVM"
        get_model_computed(model=model_false, data=data, tscv=tscv)
