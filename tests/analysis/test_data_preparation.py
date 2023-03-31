import pandas as pd
import pytest
from epp_final_project_sbp.analysis.data_preparation import data_preparation
from epp_final_project_sbp.config import (
    BLD,
)


@pytest.fixture()
def data():
    return pd.read_csv(BLD / "python" / "data" / "data_cleaned.csv")


@pytest.fixture()
def league():
    return "D1"


def test_data_preperation_wrong_inputs(data, league):
    """Test if the function raises an error if the input is not a dataframe or the
    number of last games is not an integer."""
    with pytest.raises(AssertionError):
        data_false = [1, 2, 3, 4]
        data_preparation(data=data_false, league=league)

    with pytest.raises(AssertionError):
        league_false = 5
        data_preparation(data=data, league=league_false)
