"""This module contains the general configuration of the project."""
from pathlib import Path

from epp_final_project_sbp.utilities import read_yaml

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()
TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()
PROJECT_INFO = SRC / "data" / "project_specifics.yaml"

project_info_store = read_yaml(path=PROJECT_INFO)


INFORMATION_SCRAPING = project_info_store["INFORMATION_SCRAPING"]

LEAGUES = project_info_store["LEAGUES"]
MODELS = project_info_store["MODELS"]

BETTING_STRATEGIES = project_info_store["BETTING_STRATEGIES"]

FEATURES_CREATED = project_info_store["FEATURES_CREATED"]

NOT_KNOWN_ON_GAMEDAY = project_info_store["NOT_KNOWN_ON_GAMEDAY"]

ODD_FEATURES = project_info_store["ODD_FEATURES"]


SIMULATION_RELEVANT_COLUMNS = [
    *ODD_FEATURES,
    "consensus_odds_home",
    "consensus_odds_draw",
    "consensus_odds_away",
    "full_time_result",
    "model_forecast",
]


CONSIDERED_FEATURES = project_info_store["CONSIDERED_FEATURES"]


CATEGORICAL_FEATURES = project_info_store["CATEGORICAL_FEATURES"]
INTEGER_FEATURES = project_info_store["INTEGER_FEATURES"]


LABELS = project_info_store["LABELS"]

CORR_PLOT_VARIABLES = project_info_store["CORR_PLOT_VARIABLES"]

TRAIN_SHARE = project_info_store["TRAIN_SHARE"]
MIN_FEAT_LOG_REG = project_info_store["MIN_FEAT_LOG_REG"]
INITIAL_TRAINING_DAYS = project_info_store["INITIAL_TRAINING_DAYS"]

MAX_DEPTH_OF_TREE = project_info_store["MAX_DEPTH_OF_TREE_RF"]
N_BOOTSTRAP_ITERATIONS = project_info_store["N_BOOTSTRAP_ITERATIONS_RF"]
N_SPLIT = project_info_store["N_SPLIT_CV"]
MAX_NEIGHBORS_KNN = project_info_store["MAX_NEIGHBORS_KNN"]


__all__ = ["BLD", "SRC", "TEST_DIR"]


def path_to_input_data(name):
    return SRC / "data" / f"{name}.csv"


def path_to_processed_models(name):
    return BLD / "python" / "models" / f"processed_{name}.pkl"


def path_to_input_model(name):
    return BLD / "data" / f"processed_{name}.pkl"


def path_to_final_model(name):
    return BLD / "python" / "models" / f"final_model_{name}.pkl"


def path_to_performance_store(name):
    return BLD / "python" / "models" / f"performance_store_{name}.pkl"


def path_to_simulation_results(name):
    return BLD / "python" / "data" / f"simulation_results_{name}.pkl"


def path_to_plots(name):
    return BLD / "python" / "figures" / f"{name}.png"


def path_to_tables(name):
    pass
