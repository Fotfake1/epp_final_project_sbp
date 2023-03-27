import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from epp_final_project_sbp.analysis import data_preparation as dp
from epp_final_project_sbp.config import (
    TRAIN_SHARE,
)


def model_selection(data, rf_model_cv, knn_model, logit_model):
    """This function calls the intermediate steps to select the best model out of the.

    three and computes some performance metrics.
    Input:
        data: dataframe
        rf_model_cv: random forest model
        knn_model: knn model
        logit_model: logistic regression model
        Output:
        model: best model
        performances: dataframe with performance metrics

    """
    X_train, y_train, X_test, y_test, test_data, train_data = data_splitter(
        data,
        train_share=TRAIN_SHARE,
    )
    y_pred_rf, y_pred_prob_rf = rf_predict(rf_model_cv, X_test)
    y_pred_knn, y_pred_prob_knn = knn_predict(knn_model, X_test)
    y_pred_logit, y_pred_prob_logit = logit_predict(logit_model, X_test)
    test_data = combine_test_and_predict(
        test_data,
        y_pred_logit,
        y_pred_rf,
        y_pred_knn,
        y_pred_prob_logit,
        y_pred_prob_rf,
    )
    performances = compute_performance_metrics(test_data)
    model = decide_function_model(
        result_df=performances,
        model_logit=logit_model,
        model_rf=rf_model_cv.best_estimator_,
        model_knn=knn_model,
    )

    return model, performances


def data_splitter(data, train_share):
    """Splits the data into training and test split.

    Since the data is already sorted by date the first train_share per cent
    are taken as a training set, the rest is the test set.
    Input:
        data: dataframe
        train_share: float
        Output:
        train_data: dataframe

    """
    train_data, test_data = dp.split_data(data=data, train_share=TRAIN_SHARE)
    X_train = train_data.drop(columns="full_time_result")
    y_train = train_data["full_time_result"]
    X_test = test_data.drop(columns="full_time_result")
    y_test = test_data["full_time_result"]
    return X_train, y_train, X_test, y_test, test_data, train_data


def rf_predict(rf_model_cv, X_test):
    """This function predicts the outcome of a match using a random forest model
    Input:
        rf_model: random forest model
        X_test: dataframe with the features
    Output:
        y_pred: list of predictions.
    """
    rf = rf_model_cv.best_estimator_
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    y_pred = rf.predict(X_test_scaled)
    y_pred_prob = rf.predict_proba(X_test_scaled)
    return y_pred, y_pred_prob


def logit_predict(logit_model, X_test):
    """This function predicts the outcome of a match using a logistic regression model
    Input:
        logit_model: logistic regression model
        X_test: dataframe with the features
    Output:
        y_pred: list of predictions.
    """
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    y_pred = logit_model.predict(X_test_scaled)
    y_pred_prob = logit_model.predict_proba(X_test_scaled)
    return y_pred, y_pred_prob


def knn_predict(knn_model_cv, X_test):
    """This function predicts the outcome of a match using a knn model
    Input:
        knn_model: knn model
        X_test: dataframe with the features
    Output:
        y_pred: list of predictions.
    """
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    knn = knn_model_cv.best_estimator_
    y_pred = knn.predict(X_test_scaled)
    y_pred_prob = knn.predict_proba(X_test_scaled)
    return y_pred, y_pred_prob


def combine_test_and_predict(
    test_data,
    y_pred_logit,
    y_pred_rf,
    y_pred_knn,
    y_pred_prob_logit,
    y_pred_prob_rf,
):
    """This function combines the test data with the predictions of the models."""
    test_data["logit_pred"] = y_pred_logit
    test_data["rf_pred"] = y_pred_rf
    test_data["knn_pred"] = y_pred_knn
    return test_data


def __consensus_forecast_bookmakers(row):
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


def compute_performance_metrics(test_data):
    """'This function computes the performance metrics of the models."""
    columns_result_df = [
        "accuracy_overall",
        "accuracy_home",
        "accuracy_draw",
        "accuracy_away",
        "accuracy_in_odds",
        "accuracy_not_in_odds",
    ]
    index_result_df = ["logit", "rf", "knn", "always_home", "consensus_odds"]
    result_df = pd.DataFrame(index=index_result_df, columns=columns_result_df)

    test_data["always_home"] = 2
    test_data["consensus_odds_forecast"] = test_data.apply(
        __consensus_forecast_bookmakers,
        axis=1,
    )

    # overall accuracy
    result_df.loc["logit", "accuracy_overall"] = accuracy_score(
        y_true=test_data["full_time_result"],
        y_pred=test_data["logit_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["rf", "accuracy_overall"] = accuracy_score(
        y_true=test_data["full_time_result"],
        y_pred=test_data["rf_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["knn", "accuracy_overall"] = accuracy_score(
        y_true=test_data["full_time_result"],
        y_pred=test_data["knn_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["always_home", "accuracy_overall"] = accuracy_score(
        y_true=test_data["full_time_result"],
        y_pred=test_data["always_home"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["consensus_odds", "accuracy_overall"] = accuracy_score(
        y_true=test_data["full_time_result"],
        y_pred=test_data["consensus_odds_forecast"],
        normalize=True,
        sample_weight=None,
    )
    # accuracy home
    result_df.loc["logit", "accuracy_home"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 2]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 2]["logit_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["rf", "accuracy_home"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 2]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 2]["rf_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["knn", "accuracy_home"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 2]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 2]["knn_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["always_home", "accuracy_home"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 2]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 2]["always_home"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["consensus_odds", "accuracy_home"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 2]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 2]["consensus_odds_forecast"],
        normalize=True,
        sample_weight=None,
    )
    # accuracy draw
    result_df.loc["logit", "accuracy_draw"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 0]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 0]["logit_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["rf", "accuracy_draw"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 0]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 0]["rf_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["knn", "accuracy_draw"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 0]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 0]["knn_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["always_home", "accuracy_draw"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 0]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 0]["always_home"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["consensus_odds", "accuracy_draw"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 0]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 0]["consensus_odds_forecast"],
        normalize=True,
        sample_weight=None,
    )
    # accuracy away
    result_df.loc["logit", "accuracy_away"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 1]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 1]["logit_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["rf", "accuracy_away"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 1]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 1]["rf_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["knn", "accuracy_away"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 1]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 1]["knn_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["always_home", "accuracy_away"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 1]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 1]["always_home"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["consensus_odds", "accuracy_away"] = accuracy_score(
        y_true=test_data[test_data["full_time_result"] == 1]["full_time_result"],
        y_pred=test_data[test_data["full_time_result"] == 1]["consensus_odds_forecast"],
        normalize=True,
        sample_weight=None,
    )
    # accuracy, if in line with odds
    result_df.loc["logit", "accuracy_in_odds"] = accuracy_score(
        y_true=test_data[
            test_data["logit_pred"] == test_data["consensus_odds_forecast"]
        ]["full_time_result"],
        y_pred=test_data[
            test_data["logit_pred"] == test_data["consensus_odds_forecast"]
        ]["logit_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["rf", "accuracy_in_odds"] = accuracy_score(
        y_true=test_data[test_data["rf_pred"] == test_data["consensus_odds_forecast"]][
            "full_time_result"
        ],
        y_pred=test_data[test_data["rf_pred"] == test_data["consensus_odds_forecast"]][
            "rf_pred"
        ],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["knn", "accuracy_in_odds"] = accuracy_score(
        y_true=test_data[test_data["knn_pred"] == test_data["consensus_odds_forecast"]][
            "full_time_result"
        ],
        y_pred=test_data[test_data["knn_pred"] == test_data["consensus_odds_forecast"]][
            "knn_pred"
        ],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["always_home", "accuracy_in_odds"] = accuracy_score(
        y_true=test_data[
            test_data["always_home"] == test_data["consensus_odds_forecast"]
        ]["full_time_result"],
        y_pred=test_data[
            test_data["always_home"] == test_data["consensus_odds_forecast"]
        ]["always_home"],
        normalize=True,
        sample_weight=None,
    )
    # accuracy, if not in line with odds
    result_df.loc["logit", "accuracy_not_in_odds"] = accuracy_score(
        y_true=test_data[
            test_data["logit_pred"] != test_data["consensus_odds_forecast"]
        ]["full_time_result"],
        y_pred=test_data[
            test_data["logit_pred"] != test_data["consensus_odds_forecast"]
        ]["logit_pred"],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["rf", "accuracy_not_in_odds"] = accuracy_score(
        y_true=test_data[test_data["rf_pred"] != test_data["consensus_odds_forecast"]][
            "full_time_result"
        ],
        y_pred=test_data[test_data["rf_pred"] != test_data["consensus_odds_forecast"]][
            "rf_pred"
        ],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["knn", "accuracy_not_in_odds"] = accuracy_score(
        y_true=test_data[test_data["knn_pred"] != test_data["consensus_odds_forecast"]][
            "full_time_result"
        ],
        y_pred=test_data[test_data["knn_pred"] != test_data["consensus_odds_forecast"]][
            "knn_pred"
        ],
        normalize=True,
        sample_weight=None,
    )
    result_df.loc["always_home", "accuracy_not_in_odds"] = accuracy_score(
        y_true=test_data[
            test_data["always_home"] != test_data["consensus_odds_forecast"]
        ]["full_time_result"],
        y_pred=test_data[
            test_data["always_home"] != test_data["consensus_odds_forecast"]
        ]["always_home"],
        normalize=True,
        sample_weight=None,
    )

    return result_df


def decide_function_model(result_df, model_logit, model_rf, model_knn):
    """Function to decide which model to use for the simulation."""
    acc_rf = result_df.loc["rf", "accuracy_overall"]
    acc_logit = result_df.loc["logit", "accuracy_overall"]
    acc_knn = result_df.loc["knn", "accuracy_overall"]

    if acc_rf > acc_logit and acc_rf > acc_knn:
        model = model_rf

    elif acc_logit > acc_rf and acc_logit > acc_knn:
        model = model_logit

    elif acc_knn > acc_rf and acc_knn > acc_logit:
        model = model_knn

    else:
        model = model_rf
    return model
