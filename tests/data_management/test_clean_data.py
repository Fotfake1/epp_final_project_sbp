import pandas as pd
import pytest
from epp_final_project_sbp.config import (
    CATEGORICAL_FEATURES,
    CONSIDERED_FEATURES,
    INTEGER_FEATURES,
    TEST_DIR,
)
from epp_final_project_sbp.data_management.clean_data import manage_data


@pytest.fixture()
def data():
    return pd.read_csv(TEST_DIR / "data_management" / "data_fixture.csv")


@pytest.fixture()
def feature_info():
    return TEST_DIR / "data_management" / "Features_fixture.xlsx"


def test_manage_data_wrong_inputs(data, feature_info):
    """Test if the function raises an error if the input is not a dataframe or the
    number of last games is not an integer."""
    data = pd.read_csv(TEST_DIR / "data_management" / "data_fixture.csv")
    feature_info = pd.read_excel(TEST_DIR / "data_management" / "Features_fixture.xlsx")
    with pytest.raises(AssertionError):
        manage_data(
            data=[1, 2, 3, 4],
            name_information=feature_info,
            considered_features=CONSIDERED_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            integer_features=INTEGER_FEATURES,
        )
    with pytest.raises(AssertionError):
        manage_data(
            data=data,
            name_information=[1, 2, 3, 4],
            considered_features=CONSIDERED_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            integer_features=INTEGER_FEATURES,
        )
