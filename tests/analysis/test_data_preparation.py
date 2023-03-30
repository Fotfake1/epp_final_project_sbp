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
    data = [1, 2, 3, 4]
    with pytest.raises(AssertionError):
        data_preparation(data, league)
    league = 5
    with pytest.raises(AssertionError):
        data_preparation(data, league)


# def test_normal input

# def test_not_cleaned_data
