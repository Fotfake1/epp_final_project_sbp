import pandas as pd
import pytest
from epp_final_project_sbp.config import (
    ODD_FEATURES,
    TEST_DIR,
)
from epp_final_project_sbp.data_management.feature_engineering import (
    compute_features_last_n_games,
)


def test_feature_engineering_wrong_input(data=[1, 2, 3, 4], n=5):
    """Test if the function raises an error if the input is not a dataframe or the
    number of last games is not an integer."""

    with pytest.raises(AssertionError):
        compute_features_last_n_games(data, n)
    n = 5.4
    with pytest.raises(AssertionError):
        compute_features_last_n_games(data, n)


@pytest.fixture()
def input_data():
    return TEST_DIR / "data_management" / "Fixture_input_data_feature_engineering.csv"


@pytest.fixture()
def desired_output():
    return TEST_DIR / "data_management" / "Fixture_output_data_engineering.csv"


# def test_normal input


def test_feature_engineering_normal_input(
    input_data,
    desired_output,
    n=5,
    columns=ODD_FEATURES,
):
    """Test if the function returns the correct dataframe, which I created by excel
    manually."""
    data = pd.read_csv(input_data)
    desired_output = pd.read_csv(desired_output)
    actual_output = compute_features_last_n_games(df=data, n=5)
    assert actual_output.equals(desired_output)
