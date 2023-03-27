import pandas as pd


def compute_outcomes_betting_strategies(simulation_data, ODDS_sim):
    ODDS_sim.remove("model_forecast")
    simulation_data["consensus_forecast_bookmakers"] = simulation_data.apply(
        consensus_forecast_bookmakers,
        axis=1,
    )
    result_simulation = bet_on_outcome_plain(
        data=simulation_data,
        amount=1,
        forecast_column="model_forecast",
        odds=ODDS_sim,
    )
    result_simulation = bet_on_outcome_if_not_in_line_with_consensus(
        data=result_simulation,
        amount=1,
        forecast_column="model_forecast",
        odds=ODDS_sim,
    )
    result_simulation = bet_on_outcome_if_in_line_with_consensus(
        data=result_simulation,
        amount=1,
        forecast_column="model_forecast",
        odds=ODDS_sim,
    )
    return result_simulation


def bet_on_outcome_plain(data, amount, forecast_column, odds):
    """This function simulates the behaviour of always betting on the outcome the model
    predicts.

    The amount of money you bet is always 1 dollar. The best odds are always chosesn.

    """
    profit_string = "profit_" + forecast_column
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


def bet_on_outcome_if_in_line_with_consensus(data, amount, forecast_column, odds):
    """This function simulates the behaviour of betting on the outcome the model.

    predicts, but only if the model is in line with the consensus.

    """
    profit_string = "profit_bet_with_consensus_" + forecast_column
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


def bet_on_outcome_if_not_in_line_with_consensus(data, amount, forecast_column, odds):
    """This function simulates the behaviour of betting on the outcome the model.

    predicts, but only if the model is not in line with the consensus.

    """
    profit_string = "profit_bet_againgst_consensus_" + forecast_column
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
    """This function finds the best odds for a given outcome.

    The outcome can be either "H", "D" or "A".

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
    # get the best odds
    best_odds = data[columns_with_odds].max(axis="columns", numeric_only=True).values[0]
    return best_odds


def consensus_forecast_bookmakers(row):
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
