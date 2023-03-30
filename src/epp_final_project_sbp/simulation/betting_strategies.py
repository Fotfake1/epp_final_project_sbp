import pandas as pd


def compute_outcomes_betting_strategies(simulation_data, odds):
    """This function computes the outcomes of the different betting strategies.

    Input:
        simulation_data: dataframe with the data on which the betting strategies are simulated
        odds: dictionary with the odds for the different outcomes
    Output:
        result_simulation: dataframe with the profits of the different betting strategies

    """
    simulation_data["consensus_forecast_bookmakers"] = simulation_data.apply(
        consensus_forecast_bookmakers,
        axis=1,
    )
    result_simulation = __bet_on_outcome_plain(
        data=simulation_data,
        amount=1,
        forecast_column="model_forecast",
        odds=odds,
    )
    result_simulation = __bet_on_outcome_if_not_in_line_with_consensus(
        data=result_simulation,
        amount=1,
        forecast_column="model_forecast",
        odds=odds,
    )
    result_simulation = __bet_on_outcome_if_in_line_with_consensus(
        data=result_simulation,
        amount=1,
        forecast_column="model_forecast",
        odds=odds,
    )
    return result_simulation


def __bet_on_outcome_plain(data, amount, forecast_column, odds):
    """This function simulates the behaviour of always betting on the outcome the model
    predicts.

    The amount of money you bet is always 1 dollar. The best odds are always chosesn.
    Input:
        data: dataframe with the data on which the betting strategies are simulated
        amount: amount of money you bet
        forecast_column: column with the forecast of the model
        odds: dictionary with the odds for the different outcomes
    Output:
        data: dataframe with the profits of the different betting strategies

    """
    profit_string = "bet_on_outcome_plain_" + forecast_column
    data[profit_string] = 0
    for i in range(len(data)):
        if data.iloc[i, data.columns.get_loc(forecast_column)] == 2:
            best_odds = find_best_odds(
                data=pd.DataFrame(data.iloc[i]).transpose(),
                HDA_outcome="H",
                odds=odds,
            )
            if data.iloc[i, data.columns.get_loc("full_time_result")] == 2:
                data[profit_string][i] = amount * best_odds
            else:
                data[profit_string][i] = -amount

        elif data.iloc[i, data.columns.get_loc(forecast_column)] == 0:
            best_odds = find_best_odds(
                data=pd.DataFrame(data.iloc[i]).transpose(),
                HDA_outcome="D",
                odds=odds,
            )
            if data.iloc[i, data.columns.get_loc("full_time_result")] == 0:
                data[profit_string][i] = amount * best_odds
            else:
                data[profit_string][i] = -amount

        elif data.iloc[i, data.columns.get_loc(forecast_column)] == 1:
            best_odds = find_best_odds(
                data=pd.DataFrame(data.iloc[i]).transpose(),
                HDA_outcome="A",
                odds=odds,
            )
            if data.iloc[i, data.columns.get_loc("full_time_result")] == 1:
                data[profit_string][i] = amount * best_odds
            else:
                data[profit_string][i] = -amount
    return data


def __bet_on_outcome_if_in_line_with_consensus(data, amount, forecast_column, odds):
    """This function simulates the behaviour of betting on the outcome the model.

    predicts, but only if the model is in line with the consensus.
    Input:
        data: dataframe with the data on which the betting strategies are simulated
        amount: amount of money you bet
        forecast_column: column with the forecast of the model
        odds: dictionary with the odds for the different outcomes
    Output:
        data: dataframe with the profits of the different betting strategies

    """
    profit_string = "bet_on_outcome_if_in_line_with_consensus_profit_" + forecast_column
    data[profit_string] = 0
    for i in range(len(data)):
        if data["consensus_forecast_bookmakers"][i] == data[forecast_column][i]:
            if data.iloc[i, data.columns.get_loc(forecast_column)] == 2:
                best_odds = find_best_odds(
                    data=pd.DataFrame(data.iloc[i]).transpose(),
                    HDA_outcome="H",
                    odds=odds,
                )
                if data.iloc[i, data.columns.get_loc("full_time_result")] == 2:
                    data[profit_string][i] = amount * best_odds
                else:
                    data[profit_string][i] = -amount

            elif data.iloc[i, data.columns.get_loc(forecast_column)] == 0:
                best_odds = find_best_odds(
                    data=pd.DataFrame(data.iloc[i]).transpose(),
                    HDA_outcome="D",
                    odds=odds,
                )
                if data.iloc[i, data.columns.get_loc("full_time_result")] == 0:
                    data[profit_string][i] = amount * best_odds
                else:
                    data[profit_string][i] = -amount

            elif data.iloc[i, data.columns.get_loc(forecast_column)] == 1:
                best_odds = find_best_odds(
                    data=pd.DataFrame(data.iloc[i]).transpose(),
                    HDA_outcome="A",
                    odds=odds,
                )
                if data.iloc[i, data.columns.get_loc("full_time_result")] == 1:
                    data[profit_string][i] = amount * best_odds
                else:
                    data[profit_string][i] = -amount
        else:
            data[profit_string][i] = 0
    return data


def __bet_on_outcome_if_not_in_line_with_consensus(data, amount, forecast_column, odds):
    """This function simulates the behaviour of betting on the outcome the model.

    predicts, but only if the model is not in line with the consensus.
    Input:
        data: dataframe with the data on which the betting strategies are simulated
        amount: amount of money you bet
        forecast_column: column with the forecast of the model
        odds: dictionary with the odds for the different outcomes
    Output:
        data: dataframe with the profits of the different betting strategies

    """
    profit_string = (
        "bet_on_outcome_if_not_in_line_with_consensus_profit_" + forecast_column
    )
    data[profit_string] = 0

    for i in range(len(data)):
        if data["consensus_forecast_bookmakers"][i] != data[forecast_column][i]:
            if data.iloc[i, data.columns.get_loc(forecast_column)] == 2:
                best_odds = find_best_odds(
                    data=pd.DataFrame(data.iloc[i]).transpose(),
                    HDA_outcome="H",
                    odds=odds,
                )
                if data.iloc[i, data.columns.get_loc("full_time_result")] == 2:
                    data[profit_string][i] = amount * best_odds
                else:
                    data[profit_string][i] = -amount

            elif data.iloc[i, data.columns.get_loc(forecast_column)] == 0:
                best_odds = find_best_odds(
                    data=pd.DataFrame(data.iloc[i]).transpose(),
                    HDA_outcome="D",
                    odds=odds,
                )
                if data.iloc[i, data.columns.get_loc("full_time_result")] == 0:
                    data[profit_string][i] = amount * best_odds
                else:
                    data[profit_string][i] = -amount

            elif data.iloc[i, data.columns.get_loc(forecast_column)] == 1:
                best_odds = find_best_odds(
                    data=pd.DataFrame(data.iloc[i]).transpose(),
                    HDA_outcome="A",
                    odds=odds,
                )
                if data.iloc[i, data.columns.get_loc("full_time_result")] == 1:
                    data[profit_string][i] = amount * best_odds
                else:
                    data[profit_string][i] = -amount
        else:
            data[profit_string][i] = 0
    return data


def find_best_odds(data, HDA_outcome, odds):
    """This function finds the best odds for a given outcome. The outcome can be either
    "H", "D" or "A". Based on this, the best odds are searched for this outcome.

    Input:
        data: dataframe with the data on which the betting strategies are simulated
        HDA_outcome: the outcome for which you want to find the best odds
        odds: dictionary with the odds for the different outcomes
    Output:
        best_odds: the best odds for the given outcome.

    """
    odds_sim = [
        elem
        for elem in odds
        if not any(
            val in elem
            for val in [
                "consensus_odds_home",
                "consensus_odds_draw",
                "consensus_odds_away",
            ]
        )
    ]

    columns_with_odds = [col for col in odds_sim if col.endswith(HDA_outcome)]
    best_odds = data[columns_with_odds].max(axis="columns", numeric_only=True).values[0]
    return best_odds


def consensus_forecast_bookmakers(row):
    """This function computes the consensus forecast of the bookmakers for a given.

    row in the dataframe.
    Input:
        row: a row in the dataframe
    Output:
        0: draw
        1: away win
        2: home win

    """
    if (
        row["consensus_odds_home"] < row["consensus_odds_draw"]
        and row["consensus_odds_home"] < row["consensus_odds_away"]
    ):
        return 2
    elif (
        row["consensus_odds_draw"] < row["consensus_odds_home"]
        and row["consensus_odds_draw"] < row["consensus_odds_away"]
    ):
        return 0
    else:
        return 1


def cumulative_sum(data, column):
    """computes the rolling sum of a given column.

    Input: data: pd.DataFrame
              column: column name
    Output: dataframe with a new column containing the cumulative sum of the given column

    """
    data["cumulative_sum_" + column] = data[column].cumsum()
    return data
