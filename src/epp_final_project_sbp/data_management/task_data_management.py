"""Tasks for initial data management."""

import pandas as pd
import pytask

import epp_final_project_sbp.data_management.clean_data as cd
from epp_final_project_sbp.config import (
    BLD,
    CATEGORICAL_FEATURES,
    CONSIDERED_FEATURES,
    INTEGER_FEATURES,
    SRC,
)


@pytask.mark.depends_on(
    {
        "data": SRC / "data" / "data_raw.csv",
        "sourcefile": SRC / "data" / "Features.xlsx",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_cleaned.csv")
def task_clean_data(depends_on, produces):
    """This task handles initial data cleaning, sorting the data and ensuring the.

    right data types.

    """
    data = pd.read_csv(depends_on["data"])
    name_information = pd.DataFrame(
        pd.read_excel(depends_on["sourcefile"], sheet_name="Considered_features"),
    )
    data = cd.manage_data(
        data=data,
        name_information=name_information,
        categorical_features=CATEGORICAL_FEATURES,
        considered_features=CONSIDERED_FEATURES,
        integer_features=INTEGER_FEATURES,
    )
    data.to_csv(produces, index=False)
