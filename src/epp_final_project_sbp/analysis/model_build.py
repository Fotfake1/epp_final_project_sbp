import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from epp_final_project_sbp.config import (
    MAX_DEPTH_OF_TREE,
    MAX_NEIGHBORS_KNN,
    MIN_FEAT_LOG_REG,
    N_BOOTSTRAP_ITERATIONS,
)


def get_model(model, data, tscv):
    """Calls the right subfunctions, depending on the model.

    Right now, KNN, a optimized RF and an optimized LR are implemented.
    Input:
        model: string
        data: dataframe
    Output:
        model: model

    """
    if model == "KNN":
        k_range = list(range(1, MAX_NEIGHBORS_KNN))
        param_grid = {"n_neighbors": k_range}
        model_trained = knn_model(data=data, grid=param_grid, split=tscv)

    elif model == "RF":
        model_trained = cv_get_rf_model(
            max_depth_of_trees=MAX_DEPTH_OF_TREE,
            n_bootstrap_iterations=N_BOOTSTRAP_ITERATIONS,
            tscv=tscv,
            data=data,
        )
    elif model == "LOGIT":
        clf = LogisticRegression(
            max_iter=1000,
            C=0.01,
            multi_class="multinomial",
            fit_intercept=True,
        )
        scaler = MinMaxScaler()
        model_trained = best_feature_selection_RFECV_logit(
            scaler=scaler,
            clf=clf,
            min_feat=MIN_FEAT_LOG_REG,
            data_dum=data,
            cv_split=tscv,
        )
    else:
        raise ValueError("Model not implemented")
    return model_trained


def best_feature_selection_RFECV_logit(scaler, clf, min_feat, data_dum, cv_split):
    """computes the best feature selection for Logistic Regression
    this function does "
    Input:
        scaler: MinMaxScaler
        clf: LogisticRegression
        i: number of features
        data_dum: dataframe
        Output:
        X_train: X_train
        Y_train: Y_train.
    """
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_train = data_dum.drop(columns=["full_time_result"])
    X_train = scaler.fit_transform(X_train)
    Y_train = data_dum["full_time_result"]

    model_rfecv = RFECV(
        estimator=clf,
        min_features_to_select=min_feat,
        step=1,
        cv=cv_split,
        n_jobs=-2,
        scoring="f1_macro",
    )

    model_rfecv.fit(X_train, Y_train)
    return model_rfecv


def cv_get_rf_model(max_depth_of_trees, n_bootstrap_iterations, tscv, data):
    n_estimators = [
        int(x) for x in np.linspace(start=50, stop=n_bootstrap_iterations, num=10)
    ]
    max_features = ["sqrt"]
    depth_of_trees = [int(x) for x in np.linspace(5, max_depth_of_trees, num=5)]
    depth_of_trees.append(None)
    grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": depth_of_trees,
        "bootstrap": [True],
    }
    X_train = data.drop(columns=["full_time_result"])
    Y_train = data["full_time_result"]

    rf_model_grid_search = random_forests_model(
        split=tscv,
        X_train=X_train,
        Y_train=Y_train,
        random_grid=grid,
    )
    # fine tune logistic regression
    return rf_model_grid_search


def random_forests_model(split, X_train, Y_train, random_grid):
    """This function does a grid search for the random forest model.

    Input:
        split: cross validation split
        X_train: training data, unscaled
        Y_train: Outcome variable
        random_grid: random grid,
    Output:
        rf_model: trained rf model

    """
    rf = RandomForestClassifier()
    rf_model = GridSearchCV(
        estimator=rf,
        param_grid=random_grid,
        cv=split,
        scoring="f1_macro",
        n_jobs=-2,
    )
    rf_model.fit(X=X_train, y=Y_train)
    return rf_model


def knn_model(data, grid, split):
    X_train = data.drop(columns=["full_time_result"])
    Y_train = data["full_time_result"]
    knn = KNeighborsClassifier()
    knn_model = GridSearchCV(
        knn,
        param_grid=grid,
        cv=split,
        scoring="f1_macro",
        return_train_score=False,
        n_jobs=-2,
    )

    knn_model.fit(X_train, Y_train)

    return knn_model
