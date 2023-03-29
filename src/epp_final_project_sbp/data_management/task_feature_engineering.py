import pandas as pd
import pytask

import epp_final_project_sbp.data_management.feature_engineering as fe
from epp_final_project_sbp.config import BLD, ODD_FEATURES


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "data_cleaned.csv",
        "scripts": ["feature_engineering.py"],
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_features_added.csv")
def task_feature_engineering(depends_on, produces):
    """Feature engineering."""
    odd_features = ODD_FEATURES

    data = pd.read_csv(depends_on["data"])
    data = fe.add_percentages_to_odds(df=data, columns=odd_features)
    data = fe.compute_features_last_n_games(df=data, n=5)
    data.to_csv(produces, index=False)
