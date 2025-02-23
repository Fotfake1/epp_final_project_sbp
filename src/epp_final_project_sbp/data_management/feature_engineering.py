import numpy as np
import pandas as pd


def compute_features_last_n_games(df, n):
    """
    This function computes the features for the last n games
    Input:
        df: dataframe
        n: number of games
    Output:
        df: dataframe with the features added.
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    assert isinstance(n, int), "n needs to be an integer"

    try:
        df = __compute_sum_of_points_last_n_games(df=df, number_of_matches=n)
    except:
        raise Exception(
            "Could not compute sum of points last n games. Please check, if the right datasource is given.",
        )
    try:
        df = __compute_mean_shots_on_target(df=df, number_of_matches=n)
    except:
        raise Exception(
            "Could not compute mean shots on target.  Please check, if the right datasource is given.",
        )
    try:
        df = __compute_mean_shots_on_target_opponents(df=df, number_of_matches=n)
    except:
        raise Exception(
            "Could not compute mean shots on target opponents.  Please check, if the right datasource is given.",
        )
    try:
        df = __compute_mean_goals_shot_last_n_matches(df=df, number_of_matches=n)
    except:
        raise Exception(
            "Could not compute mean goals shot last n matches.  Please check, if the right datasource is given.",
        )
    try:
        df = __compute_mean_goals_against_team_last_n_matches(
            df=df,
            number_of_matches=n,
        )
    except:
        raise Exception(
            "Could not compute mean goals against team last n matches.  Please check, if the right datasource is given.",
        )
    try:
        df = __compute_mean_goal_difference_last_n_matches(df=df, number_of_matches=n)
    except:
        raise Exception(
            "Could not compute mean goal difference last n matches.  Please check, if the right datasource is given.",
        )
    try:
        df = __compute_mean_corners_got_last_n_games(df=df, number_of_matches=n)
    except:
        raise Exception(
            "Could not compute mean corners got last n games.  Please check, if the right datasource is given.",
        )

    df["full_time_result"] = np.where(
        df.full_time_result == "H",
        2,
        np.where((df.full_time_result == "A"), 1, 0),
    )

    df.fillna(np.nan, inplace=True)

    return df


def add_percentages_to_odds(df, columns):
    """
    This function adds the percentages to the odds
    Input:
        df: dataframe
        columns: list of columns with odds
    Output:
        df: dataframe with the percentages added.
    """
    for col in columns:
        df[col + "_percentage"] = 1 / df[col]
    return df


def __add_points_based_on_game_outcome(df):
    """
    This function adds the points of the teams based on the outcome of the game
    Input:
        df: dataframe
    Output:
        df: dataframe with the points added.
    """
    df["HomeTeam_points"] = np.where(
        df["full_time_result"] == "H",
        3,
        np.where(df["full_time_result"] == "D", 1, 0),
    )
    df["AwayTeam_points"] = np.where(
        df["full_time_result"] == "A",
        3,
        np.where(df["full_time_result"] == "D", 1, 0),
    )
    return df


def __get_home_and_away_team(df, row_number):
    """
    This function returns the home and away team if a given row
    Input:
        df: dataframe
    Output:
        home_team: home team
        away_team: away team.
    """
    home_team = df.iloc[row_number]["HomeTeam"]
    away_team = df.iloc[row_number]["AwayTeam"]
    return home_team, away_team


def __get_last_n_matches(df, number_of_matches, row_number):
    """
    This function returns the last n matches of the home and away team
    Input:
        data: dataframe
        number_of_matches: number of matches
        row_number: row number
    Output:
        home_matches: dataframe with the last n matches of the home team
        away_matches: dataframe with the last n matches of the away team.
    """
    home_team = df.iloc[row_number]["HomeTeam"]
    away_team = df.iloc[row_number]["AwayTeam"]
    home_matches = df[
        ((df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team))
        & (df.index < row_number)
    ]
    away_matches = df[
        ((df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team))
        & (df.index < row_number)
    ]

    if home_matches.shape[0] >= number_of_matches:
        home_matches = home_matches.tail(number_of_matches)
    if away_matches.shape[0] >= number_of_matches:
        away_matches = away_matches.tail(number_of_matches)

    return home_matches, away_matches


def __extract_list_of_points(matches, team):
    """
    This function computes the sum of points of a team in a set of matches
    Input:
        matches: dataframe with the matches
        team: team name
    Output:
        matches_points: list of points.
    """
    matches_points = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            matches_points.append(matches.iloc[i]["HomeTeam_points"])
        else:
            matches_points.append(matches.iloc[i]["AwayTeam_points"])
    return matches_points


def __compute_sum_of_points_last_n_games(df, number_of_matches):
    """
    This function adds the sum of points in the last n games without the current game
    Input:
        df: dataframe
        n: number of games
    Output:
        df: dataframe with the sum of points added.
    """
    df = __add_points_based_on_game_outcome(df)

    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df=df,
            number_of_matches=number_of_matches,
            row_number=row_number,
        )

        if home_matches.shape[0] > 0:
            home_matches_points = []
            home_matches_points = __extract_list_of_points(
                matches=home_matches,
                team=home_team,
            )
            df.loc[
                row_number,
                "HomeTeam_sum_points_last_" + str(number_of_matches) + "_matches",
            ] = sum(home_matches_points)

        if away_matches.shape[0] > 0:
            away_matches_points = []
            away_matches_points = __extract_list_of_points(
                matches=away_matches,
                team=away_team,
            )
            df.loc[
                row_number,
                "AwayTeam_sum_points_last_" + str(number_of_matches) + "_matches",
            ] = sum(away_matches_points)

    return df


def __extract_list_of_shots_on_target(matches, team):
    """
    This function extracts the number of shots on target for a given team
    Input:
        matches: dataframe with the matches
        team: team
    Output:
        list_of_shots_on_target: list with the number of shots on target.
    """
    list_of_shots_on_target = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            list_of_shots_on_target.append(
                matches.iloc[i].loc["hometeam_shots_on_target"],
            )
        else:
            list_of_shots_on_target.append(
                matches.iloc[i].loc["awayteam_shots_on_target"],
            )
    return list_of_shots_on_target


def __compute_mean_shots_on_target(df, number_of_matches):
    """
    This function computes the mean shots on target in the last n matches
    Input:
        data: dataframe
        number_of_matches: number of matches
    Output:
        data: dataframe with the mean shots on target added.
    """
    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df,
            number_of_matches,
            row_number,
        )

        if home_matches.shape[0] > 0:
            home_team_shots_on_target = []
            home_team_shots_on_target = __extract_list_of_shots_on_target(
                matches=home_matches,
                team=home_team,
            )
            df.loc[
                row_number,
                "HomeTeam_mean_shots_on_target_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(home_team_shots_on_target)

        if away_matches.shape[0] > 0:
            away_team_shots_on_target = []
            away_team_shots_on_target = __extract_list_of_shots_on_target(
                matches=away_matches,
                team=away_team,
            )
            df.loc[
                row_number,
                "AwayTeam_mean_shots_on_target_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(away_team_shots_on_target)

    return df


def __extract_list_of_shots_on_target_opponents(matches, team):
    """
    This function extracts the number of shots on target for the opponents of a given team
    Input:
        matches: dataframe with the matches
        team: team
    Output:
        list_of_shots_on_target: list with the number of shots on target of the opponents.
    """
    list_of_shots_on_target_opponents = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            list_of_shots_on_target_opponents.append(
                matches.iloc[i].loc["awayteam_shots_on_target"],
            )
        else:
            list_of_shots_on_target_opponents.append(
                matches.iloc[i].loc["hometeam_shots_on_target"],
            )
    return list_of_shots_on_target_opponents


def __compute_mean_shots_on_target_opponents(df, number_of_matches):
    """
    This function computes the mean shots on target of the opponents in the last n matches
    Input:
        data: dataframe
        number_of_matches: number of matches
    Output:
        data: dataframe with the mean shots on target of the opponents added.
    """
    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df,
            number_of_matches,
            row_number,
        )

        if home_matches.shape[0] > 0:
            home_team_shots_on_target_opponents = []
            home_team_shots_on_target_opponents = (
                __extract_list_of_shots_on_target_opponents(
                    matches=home_matches,
                    team=home_team,
                )
            )
            df.loc[
                row_number,
                "HomeTeam_mean_shots_on_target_opponents_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(home_team_shots_on_target_opponents)

        if away_matches.shape[0] > 0:
            away_team_shots_on_target_opponents = []
            away_team_shots_on_target_opponents = (
                __extract_list_of_shots_on_target_opponents(
                    matches=away_matches,
                    team=away_team,
                )
            )
            df.loc[
                row_number,
                "AwayTeam_mean_shots_on_target_opponents_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(away_team_shots_on_target_opponents)

    return df


def __extract_list_of_goals(matches, team):
    """
    This function extracts a list of goals for a given team
    Input:
        matches: dataframe with the matches
        team: team
    Output:
        list_of_goals: list of goals.
    """
    list_of_goals = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            list_of_goals.append(matches.iloc[i].loc["full_time_goals_hometeam"])
        else:
            list_of_goals.append(matches.iloc[i].loc["full_time_goals_awayteam"])
    return list_of_goals


def __compute_mean_goals_shot_last_n_matches(df, number_of_matches):
    """
    This function computes the mean goals shot in the last n matches
    Input:
        data: dataframe
        number_of_matches: number of matches
    Output:
        data: dataframe with the mean goals shot added.
    """
    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df,
            number_of_matches,
            row_number,
        )

        if home_matches.shape[0] > 0:
            home_team_goals = []
            home_team_goals = __extract_list_of_goals(
                matches=home_matches,
                team=home_team,
            )
            df.loc[
                row_number,
                "HomeTeam_mean_goals_shot_last_" + str(number_of_matches) + "_matches",
            ] = np.mean(home_team_goals)
        if away_matches.shape[0] > 0:
            away_team_goals = []
            away_team_goals = __extract_list_of_goals(
                matches=away_matches,
                team=away_team,
            )
            df.loc[
                row_number,
                "AwayTeam_mean_goals_shot_last_" + str(number_of_matches) + "_matches",
            ] = np.mean(away_team_goals)
    return df


def __extract_list_of_goals_against(matches, team):
    """
    This function extracts a list of goals against for a given team
    Input:
        matches: dataframe with the matches
        team: team
    Output:
        list_of_goals: list of goals.
    """
    list_of_goals = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            list_of_goals.append(matches.iloc[i].loc["full_time_goals_awayteam"])
        else:
            list_of_goals.append(matches.iloc[i].loc["full_time_goals_hometeam"])
    return list_of_goals


def __compute_mean_goals_against_team_last_n_matches(df, number_of_matches):
    """
    This function computes the mean goals against in the last n matches
    Input:
        data: dataframe
        number_of_matches: number of matches
    Output:
        data: dataframe with the mean goals against added.
    """
    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df,
            number_of_matches,
            row_number,
        )

        if home_matches.shape[0] > 0:
            home_team_goals = []
            home_team_goals = __extract_list_of_goals_against(
                matches=home_matches,
                team=home_team,
            )
            df.loc[
                row_number,
                "HomeTeam_mean_goals_against_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(home_team_goals)
        if away_matches.shape[0] > 0:
            away_team_goals = []
            away_team_goals = __extract_list_of_goals_against(
                matches=away_matches,
                team=away_team,
            )
            df.loc[
                row_number,
                "AwayTeam_mean_goals_against_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(away_team_goals)
    return df


def __extract_the_goal_difference_list(matches, team):
    """
    This function computes the mean goal difference of a team in a set of matches
    Input:
        matches: dataframe with the matches
        team: team name
    Output:
        matches_goal_difference: list of goal differences.
    """
    matches_goal_difference = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            matches_goal_difference.append(
                matches.iloc[i]["full_time_goals_hometeam"]
                - matches.iloc[i]["full_time_goals_awayteam"],
            )
        else:
            matches_goal_difference.append(
                matches.iloc[i]["full_time_goals_awayteam"]
                - matches.iloc[i]["full_time_goals_hometeam"],
            )
    return matches_goal_difference


def __compute_mean_goal_difference_last_n_matches(df, number_of_matches):
    """
    This function computes the mean goal difference in the last n matches
    Input:
        data: dataframe
        number_of_matches: number of matches
    Output:
        data: dataframe with the mean goal difference added.
    """
    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df,
            number_of_matches,
            row_number,
        )

        if home_matches.shape[0] > 0:
            home_team_goal_difference = []
            home_team_goal_difference = __extract_the_goal_difference_list(
                matches=home_matches,
                team=home_team,
            )
            df.loc[
                row_number,
                "HomeTeam_mean_goal_difference_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(home_team_goal_difference)
        if away_matches.shape[0] > 0:
            away_team_goal_difference = []
            away_team_goal_difference = __extract_the_goal_difference_list(
                matches=away_matches,
                team=away_team,
            )
            df.loc[
                row_number,
                "AwayTeam_mean_goal_difference_last_"
                + str(number_of_matches)
                + "_matches",
            ] = np.mean(away_team_goal_difference)
    return df


def __extract_the_list_of_corners(matches, team):
    """
    This function computes the sum of corners of a team in a set of matches
    Input:
        matches: dataframe with the matches
        team: team name
    Output:
        matches_corners: list of corners.
    """
    matches_corners = []
    for i in range(len(matches)):
        if matches.iloc[i].loc["HomeTeam"] == team:
            matches_corners.append(matches.iloc[i]["hometeam_corners"])
        else:
            matches_corners.append(matches.iloc[i]["awayteam_corners"])
    return matches_corners


def __compute_mean_corners_got_last_n_games(df, number_of_matches):
    """
    This function computes the mean corners got in the last n games
    Input:
        df: dataframe
        n: number of games
    Output:
        df: dataframe with the mean corners got added.
    """
    for row_number in range(df.shape[0]):
        home_team, away_team = __get_home_and_away_team(df=df, row_number=row_number)
        home_matches, away_matches = __get_last_n_matches(
            df,
            number_of_matches,
            row_number,
        )

        if home_matches.shape[0] > 0:
            home_team_corners = []
            home_team_corners = __extract_the_list_of_corners(
                matches=home_matches,
                team=home_team,
            )
            df.loc[
                row_number,
                "HomeTeam_mean_corners_got_last_" + str(number_of_matches) + "_matches",
            ] = np.mean(home_team_corners)

        if away_matches.shape[0] > 0:
            away_team_corners = []
            away_team_corners = __extract_the_list_of_corners(
                matches=away_matches,
                team=away_team,
            )
            df.loc[
                row_number,
                "AwayTeam_mean_corners_got_last_" + str(number_of_matches) + "_matches",
            ] = np.mean(away_team_corners)

    return df
