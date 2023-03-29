import billiard as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def simulate_forecasting_parallel(data, number_of_initial_training_dates, model):
    """This function simulates the forecasting process of a given model, given the
    dataset.

    up until a certain date iteratively. The function returns a dataframe, containing the
    simulated forecasts of the given model.
    Input:
    data: pd.DataFrame
        The dataset, containing the features and the target variable.
        number_of_initial_training_dates: int
        The number of initial training dates, which are used to train the model.
        model: sklearn model
        The model, which is used to simulate the forecasting process.
        Output:
        simulated_data: pd.DataFrame
        The simulated forecasts of the given model.

    """
    import warnings

    dates = pd.to_datetime(np.unique(np.array(data.index)))
    dates = dates[number_of_initial_training_dates:]
    simulated_data = pd.DataFrame()
    scaler = MinMaxScaler()

    pool = mp.Pool(mp.cpu_count() - 1)
    warnings.filterwarnings("ignore")
    with mp.Pool(mp.cpu_count() - 1) as pool:
        results = pool.map(
            __simulate_date_parallel,
            [(date, data, model, scaler) for date in dates],
        )
    pool.close()
    for result in results:
        if simulated_data.empty:
            simulated_data = result
        else:
            simulated_data = pd.concat([simulated_data, result])

    return simulated_data


def __simulate_date_parallel(args):
    """This is a helper function, which is used to simulate the forecasting process of a
    given model, given the dataset.

    for one specific date.
    Input:
        date: pd.Timestamp - the date to simulate the forecasting process for

    Output:
        subset_test: pd.DataFrame - the simulated forecasts of the given model for the given date

    """
    date, data, model, scaler = args
    subset_train = get_data_before_date(data, date)
    subset_test = get_data_on_date(data, date)
    X_train, y_train, X_test = __create_training_test_data(
        subset_train=subset_train,
        subset_test=subset_test,
        scaler=scaler,
    )
    subset_test["model_forecast"] = __model_forecast(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model=model,
    )
    return subset_test


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
    if isinstance(model, RandomForestClassifier) or isinstance(
        model.estimator,
        RandomForestClassifier,
    ):
        y_pred_model = __rf_model_forecast(X_train, y_train, X_test, model)
    elif isinstance(model.estimator, KNeighborsClassifier) or isinstance(
        model,
        KNeighborsClassifier,
    ):
        y_pred_model = __knn_model_forecast(X_train, y_train, X_test, model)
    elif isinstance(model.estimator, LogisticRegression) or isinstance(
        model,
        LogisticRegression,
    ):
        y_pred_model = __logit_model_forecast(X_train, y_train, X_test, model)
    else:
        raise TypeError(
            "Invalid model type. Supported model types are RandomForestClassifier, KNeighborsClassifier and LogisticRegression",
        )
    return y_pred_model


def __create_training_test_data(subset_train, subset_test, scaler):
    X_train = subset_train.drop(columns="full_time_result")
    y_train = subset_train["full_time_result"]
    X_train = scaler.fit_transform(X_train.values)

    X_test = subset_test.drop(columns="full_time_result")
    X_test = scaler.fit_transform(X_test.values)

    return X_train, y_train, X_test


def __logit_model_forecast(X_train, y_train, X_test, logit_model):
    logit_model = logit_model.fit(X_train, y_train)
    y_pred_logit = logit_model.predict(X_test)

    return y_pred_logit


def __rf_model_forecast(X_train, y_train, X_test, rf_model):
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    return y_pred_rf


def __knn_model_forecast(X_train, y_train, X_test, knn_model):
    knn_model_sim = knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model_sim.predict(X_test)
    return y_pred_knn
