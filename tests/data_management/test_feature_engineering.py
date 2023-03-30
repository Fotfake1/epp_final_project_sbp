import pandas as pd
import pytest
from epp_final_project_sbp.config import (
    BLD,
)
from epp_final_project_sbp.data_management.feature_engineering import (
    compute_features_last_n_games,
)


@pytest.fixture()
def data():
    return pd.read_csv(
        BLD / "python" / "data" / "data_cleaned.csv",
    )


def test_feature_engineering_wrong_input(data, n=5):
    """Test if the function raises an error if the input is not a dataframe or the
    number of last games is not an integer."""
    data = [1, 2, 3, 4]
    n = 4
    with pytest.raises(AssertionError):
        compute_features_last_n_games(data, n)
    n = 5.4
    with pytest.raises(AssertionError):
        compute_features_last_n_games(data, n)


# def test_normal input

# def test_not_cleaned_data
