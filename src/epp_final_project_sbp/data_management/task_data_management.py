"""Tasks for cleaning the data."""


import pandas as pd
import pytask

import epp_final_project_sbp.data_management.clean_data as cd
from epp_final_project_sbp.config import BLD, SRC


@pytask.mark.depends_on(
    {
        "data": SRC / "data" / "data_raw.csv",
        "sourcefile": SRC / "data" / "Features.xlsx",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_cleaned.csv")
def task_clean_data(depends_on, produces):
    """Clean the data."""
    # these information will be in the data_info.yaml file in a later version
    considered_features = [
        "Date",
        "league",
        "kick_off_time",
        "HomeTeam",
        "AwayTeam",
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

    # all elements to make categorical

    categorical_features = [
        "league",
        "HomeTeam",
        "AwayTeam",
        "full_time_result",
        "half_time_result",
    ]
    integer_features = [
        "full_time_goals_hometeam",
        "full_time_goals_awayteam",
        "half_time_goals_hometeam",
        "half_time_goals_awayteam",
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

    # all columns with features that are not known on game day

    data = pd.read_csv(depends_on["data"])
    name_information = pd.DataFrame(
        pd.read_excel(depends_on["sourcefile"], sheet_name="Considered_features"),
    )
    data = cd.convert_column_names_to_sensible_names(
        name_information=name_information,
        df=data,
    )
    data = data[considered_features]
    data = cd.convert_to_categorical(df=data, columns=categorical_features)
    data["Date"] = pd.to_datetime(
        data["Date"],
        format="%d/%m/%Y",
        errors="coerce",
    ).fillna(pd.to_datetime(data["Date"], format="%d/%m/%y", errors="coerce"))
    data.sort_values(by=["Date"], inplace=True)
    data = data.reset_index()
    data = cd.convert_to_integer(df=data, columns=integer_features)
    data.to_csv(produces, index=False)
