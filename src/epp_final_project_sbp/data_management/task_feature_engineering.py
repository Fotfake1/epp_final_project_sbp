import pandas as pd
import pytask

import epp_final_project_sbp.data_management.feature_engineering as fe
from epp_final_project_sbp.config import BLD


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "data_cleaned.csv",
        "scripts": ["feature_engineering.py"],
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_features_added.csv")
def task_feature_engineering(depends_on, produces):
    """Feature engineering."""
    odd_features = [
        "B365H",
        "B365D",
        "B365A",
        "BSH",
        "BSD",
        "BSA",
        "BWH",
        "BWD",
        "BWA",
        "GBH",
        "GBD",
        "GBA",
        "IWH",
        "IWD",
        "IWA",
        "LBH",
        "LBD",
        "LBA",
        "PSH",
        "PSD",
        "PSA",
        "SBH",
        "SBD",
        "SBA",
        "SJH",
        "SJD",
        "SJA",
        "VCH",
        "VCD",
        "VCA",
        "WHH",
        "WHD",
        "WHA",
    ]
    not_known_on_game_day = [
        "full_time_goals_hometeam",
        "full_time_goals_awayteam",
        "full_time_result",
        "half_time_goals_hometeam",
        "half_time_goals_awayteam",
        "half_time_result",
        "hometeam_shots",
        "awayteam_shots",
        "hometeam_shots_on_target",
        "awayteam_shots_on_target",
        "hometeam_corners",
        "awayteam_corners",
        "hometeam_fouls_done",
        "awayteam_fouls_done",
        "hometeam_yellow_cards",
        "awayteam_yellow_cards",
        "hometeam_red_cards",
        "awayteam_red_cards",
    ]

    data = pd.read_csv(depends_on["data"])
    data = fe.add_percentages_to_odds(df=data, columns=odd_features)
    data = fe.compute_features_last_n_games(df=data, n=5)
    data = data.drop(columns=not_known_on_game_day)
    data = data.drop(columns=odd_features)

    data.to_csv(produces, index=False)
