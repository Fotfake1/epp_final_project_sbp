import pandas as pd
import pytest
from epp_final_project_sbp.analysis.model_build import get_model_computed
from sklearn.model_selection import (
    TimeSeriesSplit,
)


@pytest.fixture()
def model():
    model = "RF"
    return model


@pytest.fixture()
def wrong_data():
    data = pd.DataFrame([1, 2, 3, 4], columns=["test_wrong_data"])
    return data


@pytest.fixture()
def tscv():
    tscv = TimeSeriesSplit(n_splits=5)
    return tscv


def test_model_built_wrong_inputs(data=wrong_data, model=model, tscv=tscv, league="E0"):
    """Test if the function raises an error if the input is not a dataframe or the
    number of last games is not an integer."""
    with pytest.raises(AssertionError):
        data_false = [1, 2, 3, 4]
        get_model_computed(model=model, data=data_false, tscv=tscv)

    with pytest.raises(AssertionError):
        tscv_false = "TimeSeriesSplit"
        get_model_computed(model=model, data=data, tscv=tscv_false)

    with pytest.raises(AssertionError):
        model_false = "SVM"
        get_model_computed(model=model_false, data=data, tscv=tscv)
