import pandas as pd

from epp_final_project_sbp.config import NOT_KNOWN_ON_GAMEDAY, ODD_FEATURES


def data_preparation(
    data,
    league,
    not_known_on_game_day=NOT_KNOWN_ON_GAMEDAY,
    odds=ODD_FEATURES,
):
    """prepares the data, to be used in the model built and so on
    Input:
        data: dataframe, which is already cleaned
        not_known_on_game_day: list of columns, which are not known on game day
        odds: list of columns, which are the odds.
    Output:
        data: dataframe.
    """
    assert isinstance(
        data,
        pd.DataFrame,
    ), "data is not a dataframe, please check the datasource"
    assert isinstance(
        league,
        str,
    ), "league is not a string, please check the league variable"

    data = data.drop(columns="index")
    data = data.set_index("Date")
    data = data.loc[data["league"] == league]
    data = __compute_consensus_odds(df=data, columns_with_odds=odds)
    data = __compute_percentages_out_of_consensus_odds(df=data)
    data = data.drop(columns=not_known_on_game_day)
    data = data.drop(columns=["league", "kick_off_time"], axis=1)
    data = pd.get_dummies(data)
    data = data.fillna(-33)
    data = data_robustness_check(data=data)
    return pd.DataFrame(data)


def get_league(string):
    """extracts the league out of the path
    Input:
        produces: path
         Output:
            league: str.
    """
    if "E0" in str(string):
        league = "E0"
    elif "D1" in str(string):
        league = "D1"
    elif "SP1" in str(string):
        league = "SP1"
    elif "I1" in str(string):
        league = "I1"
    return league


def get_model(produces):
    """Extracts the model out of the path, used multiple times.

    throughout the project.
    Input:
        produces: path
    Output
        model: str.

    """
    if "LOGIT" in str(produces):
        model = "LOGIT"
    elif "RF" in str(produces):
        model = "RF"
    elif "KNN" in str(produces):
        model = "KNN"
    return model


def __compute_consensus_odds(df, columns_with_odds):
    """This function computes the consensus odds, based on the ending of the meaningful
    column names.

    For this, the mean of the relevant columns is computed and stored in a new column.
    Input:
        df: dataframe
        columns_with_odds: list of columns with the odds
    Output:
        df: dataframe with the consensus odds added.

    """
    columns_with_odds = [x for x in columns_with_odds if x in list(df.columns)]
    home_odd_columns = [col for col in columns_with_odds if col.endswith("H")]
    draw_odd_columns = [col for col in columns_with_odds if col.endswith("D")]
    away_odd_columns = [col for col in columns_with_odds if col.endswith("A")]

    df["consensus_odds_home"] = df[home_odd_columns].mean(axis=1)
    df["consensus_odds_draw"] = df[draw_odd_columns].mean(axis=1)
    df["consensus_odds_away"] = df[away_odd_columns].mean(axis=1)
    return df


def data_robustness_check(data):
    """Drop the columns, where all entries are NaN
    Input:
        data: dataframe
    Output:
    data: dataframe
    .
    """
    data = data.dropna(axis=1, how="all")
    return data


def __compute_percentages_out_of_consensus_odds(df):
    """This function computes the percentages out of the consensus odds
    Input:
        df: dataframe
        columns_with_consensus_odds: list of columns with the consensus odds
    Output:
        df: dataframe with the percentages out of the consensus odds added.
    """
    df["consensus_percentage_home"] = 1 / df["consensus_odds_home"]
    df["consensus_percentage_draw"] = 1 / df["consensus_odds_draw"]
    df["consensus_percentage_away"] = 1 / df["consensus_odds_away"]
    df["consensus_sum_of_percentages"] = (
        df["consensus_percentage_home"]
        + df["consensus_percentage_draw"]
        + df["consensus_percentage_away"]
    )
    df["consensus_percentage_home"] = (
        df["consensus_percentage_home"] / df["consensus_sum_of_percentages"]
    )
    df["consensus_percentage_draw"] = (
        df["consensus_percentage_draw"] / df["consensus_sum_of_percentages"]
    )
    df["consensus_percentage_away"] = (
        df["consensus_percentage_away"] / df["consensus_sum_of_percentages"]
    )
    return df


def split_data(data, train_share):
    """Splits the data into training and test split.

    Since the data is already sorted by date the first train_share per cent
    are taken as a training set, the rest is the test set.
    Input:
        data: dataframe
        train_share: float
        Output:
        train_data: dataframe

    """
    train_sample = int(len(data) * train_share)
    test_data = data.iloc[train_sample:]
    train_data = data.iloc[:train_sample]
    return train_data, test_data


def get_betting_strategy(path, list_of_betting_strategies):
    """Extracts the betting strategy out of the path."""
    path = str(path)
    for strategy in list_of_betting_strategies:
        if strategy in path:
            return strategy
