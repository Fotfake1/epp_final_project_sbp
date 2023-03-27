import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def simulate_forecasting(data, number_of_initial_training_dates, model):
    import warnings

    warnings.filterwarnings("ignore")
    dates = pd.to_datetime(np.unique(np.array(data.index)))
    ###starting with the 50th date
    dates = dates[number_of_initial_training_dates:]
    simulated_data = pd.DataFrame()
    scaler = MinMaxScaler()
    for date in dates:
        subset_train = get_data_before_date(data, date)
        subset_test = get_data_on_date(data, date)

        X_train = subset_train.drop(columns="full_time_result")
        y_train = subset_train["full_time_result"]
        X_train = scaler.fit_transform(X_train.values)
        X_test = subset_test.drop(columns="full_time_result")
        subset_test["full_time_result"]
        X_test = scaler.fit_transform(X_test.values)

        subset_test["model_forecast"] = __model_forecast(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            model=model,
        )

        if simulated_data.empty:
            simulated_data = subset_test
        else:
            simulated_data = pd.concat([simulated_data, subset_test])
    return simulated_data


def get_data_before_date(df, target_date):
    # Convert the index to datetime format
    df.index = pd.to_datetime(df.index)

    # Subset the dataframe where dates are before the target date
    subset_df = df[df.index < target_date]

    return subset_df


def get_data_on_date(df, target_date):
    # Convert the index to datetime format
    df.index = pd.to_datetime(df.index)

    # Subset the dataframe where dates are before the target date
    subset_df = df[df.index == target_date]

    return subset_df


def __model_forecast(X_train, y_train, X_test, model):
    if isinstance(model.estimator, RandomForestClassifier):
        y_pred_model = __rf_model_forecast(X_train, y_train, X_test, model)
    elif isinstance(model.estimator, KNeighborsClassifier):
        y_pred_model = __knn_model_forecast(X_train, y_train, X_test, model)
    elif isinstance(model.estimator, LogisticRegression):
        y_pred_model = __logit_model_forecast(X_train, y_train, X_test, model)
    else:
        raise TypeError(
            "Invalid model type. Supported model types are RandomForestClassifier, KNeighborsClassifier and LogisticRegression",
        )
    return y_pred_model


def __logit_model_forecast(X_train, y_train, X_test, logit_model):
    x_train_transf = logit_model.transform(X_train)
    log_model_sim = logit_model.fit(x_train_transf, y_train)
    x_test_transf = log_model_sim.transform(X_test)
    y_pred_logit = log_model_sim.predict(x_test_transf)

    return y_pred_logit


def __rf_model_forecast(X_train, y_train, X_test, rf_model):
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    return y_pred_rf


def __knn_model_forecast(X_train, y_train, X_test, knn_model):
    knn_model_sim = knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model_sim.predict(X_test)
    return y_pred_knn
