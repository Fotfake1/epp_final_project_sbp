"""This module contains the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()


LEAGUES = ["E0", "SP1", "D1", "I1"]
MODELS = ["RF_model", "LOGIT_model", "KNN_model"]

NOT_KNOWN_ON_GAMEDAY = [
    "full_time_goals_hometeam",
    "full_time_goals_awayteam",
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
    "HomeTeam_points",
    "AwayTeam_points",
]

ODDS = [
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
    "consensus_odds_home",
    "consensus_odds_draw",
    "consensus_odds_away",
    "consensus_sum_of_percentages",
]


SIMULATION_RELEVANT_COLUMNS = [
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
    "consensus_odds_home",
    "consensus_odds_draw",
    "consensus_odds_away",
    "model_forecast",
    "full_time_result",
]

# general boundary parameter KNN model
TRAIN_SHARE = 0.8
# general boundary parameter Logit model
MIN_FEAT_LOG_REG = 4

# genereal boundary parameter RF model
MAX_DEPTH_OF_TREE = 100
N_BOOTSTRAP_ITERATIONS = 1000
N_SPLIT = 5
MAX_NEIGHBORS_KNN = 30


__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]


def path_to_input_data(name):
    return SRC / "data" / f"{name}.csv"


def path_to_processed_data(name):
    return BLD / "data" / f"processed_{name}.pkl"


def path_to_input_model(name):
    return BLD / "data" / f"processed_{name}.pkl"


def path_to_final_model(name):
    return BLD / "python" / "models" / f"final_model_{name}.pkl"


def path_to_performance_store(name):
    return BLD / "python" / "models" / f"performance_store_{name}.pkl"


def path_to_simulation_results(name):
    return BLD / "python" / "data" / f"simulation_results_{name}.pkl"
