"""Functions for predicting outcomes based on the estimated model."""

import numpy as np
import pandas as pd


def predict_prob_by_age(data, model, group):
    """Predict smoking probability for varying age values.

    For each group value in column data[group] we create new data that runs through a
    grid of age values from data.age.min() to data.age.max() and fixes all column
    values to the ones returned by data.mode(), except for the group column.

    Args:
        data (pandas.DataFrame): The data set.
        model (statsmodels.base.model.Results): The fitted model.
        group (str): Categorical column in data set. We create predictions for each
            unique value in column data[group]. Cannot be 'age' or 'smoke'.

    Returns:
        pandas.DataFrame: Predictions. Has columns 'age' and one column for each
            category in column group.

    """
    age_min = data["age"].min()
    age_max = data["age"].max()
    age_grid = np.arange(age_min, age_max + 1)

    mode = data.mode()

    new_data = pd.DataFrame(age_grid, columns=["age"])

    cols_to_set = list(set(data.columns) - {group, "age", "smoke"})
    new_data = new_data.assign(**dict(mode.loc[0, cols_to_set]))

    predicted = {"age": age_grid}
    for group_value in data[group].unique():
        _new_data = new_data.copy()
        _new_data[group] = group_value
        predicted[group_value] = model.predict(_new_data)

    predicted = pd.DataFrame(predicted)
    return predicted

    np.random.seed(101)

    X = df_dum.drop(columns=["winner"], axis=1)
    y = df_dum.winner.values

    # splitting into train and test set to check which model is the best one to work on
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # scaling features
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # creating models variable to iterate through each model and print result
    models = [
        LogisticRegression(max_iter=1000, multi_class="multinomial"),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
    ]

    names = ["Logistic Regression", "Random Forest", "Gradient Boost", "KNN"]

    # loop through each model and print train score and elapsed time
    for model, _name in zip(models, names):
        time.time()
        scores = cross_val_score(model, X_train, y_train, scoring="accuracy", cv=5)

    # Creating loop to test which set of features is the best one for Logistic Regression

    acc_results = []
    n_features = []

    # best classifier on training data
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")

    for i in range(5, 40):
        rfe = RFE(estimator=clf, n_features_to_select=i, step=1)
        rfe.fit(X, y)
        X_temp = rfe.transform(X)

        np.random.seed(101)

        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        time.time()
        scores = cross_val_score(clf, X_train, y_train, scoring="accuracy", cv=5)
        acc_results.append(scores.mean())
        n_features.append(i)

    plt.plot(n_features, acc_results)
    plt.ylabel("Accuracy")
    plt.xlabel("N features")
    plt.show()

    # getting the best 13 features from RFE
    rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
    rfe.fit(X, y)
    X_transformed = rfe.transform(X)

    np.random.seed(101)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # getting column names
    featured_columns = pd.DataFrame(rfe.support_, index=X.columns, columns=["is_in"])

    featured_columns = featured_columns[featured_columns.is_in is True].index.tolist()

    # column importances for each class
    pd.DataFrame(
        np.exp(rfe.estimator_.coef_[0]),
        index=featured_columns,
        columns=["coef"],
    ).sort_values("coef", ascending=False)

    pd.DataFrame(
        np.exp(rfe.estimator_.coef_[1]),
        index=featured_columns,
        columns=["coef"],
    ).sort_values("coef", ascending=False)

    pd.DataFrame(
        np.exp(rfe.estimator_.coef_[2]),
        index=featured_columns,
        columns=["coef"],
    ).sort_values("coef", ascending=False)

    # tuning logistic regression
    parameters = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "fit_intercept": (True, False),
        "solver": ("newton-cg", "sag", "saga", "lbfgs"),
        "class_weight": (None, "balanced"),
    }

    gs = GridSearchCV(clf, parameters, scoring="accuracy", cv=3)
    time.time()

    # printing best fits and time elapsed
    gs.fit(X_train, y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # testing models on unseen data
    tpred_lr = gs.best_estimator_.predict(X_test)
    tpred_rf = rf.predict(X_test)
    tpred_gb = gb.predict(X_test)
    tpred_knn = knn.predict(X_test)

    # function to get winning odd value in simulation dataset
    def get_winning_odd(df):
        if df.winner == 2:
            result = df.h_odd
        elif df.winner == 1:
            result = df.a_odd
        else:
            result = df.d_odd
        return result

    # creating dataframe with test data to simulate betting winnings with models

    test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=featured_columns)
    test_df["tpred_lr"] = tpred_lr
    test_df["tpred_rf"] = tpred_rf
    test_df["tpred_gb"] = tpred_gb
    test_df["tpred_knn"] = tpred_knn

    test_df["winner"] = y_test
    test_df["winning_odd"] = test_df.apply(lambda x: get_winning_odd(x), axis=1)

    test_df["lr_profit"] = (
        (test_df.winner == test_df.tpred_lr) * test_df.winning_odd * 100
    )
    test_df["rf_profit"] = (
        (test_df.winner == test_df.tpred_rf) * test_df.winning_odd * 100
    )
    test_df["gb_profit"] = (
        (test_df.winner == test_df.tpred_gb) * test_df.winning_odd * 100
    )
    test_df["knn_profit"] = (
        (test_df.winner == test_df.tpred_knn) * test_df.winning_odd * 100
    )

    investment = len(test_df) * 100

    lr_return = test_df.lr_profit.sum() - investment
    test_df.rf_profit.sum() - investment
    test_df.gb_profit.sum() - investment
    test_df.knn_profit.sum() - investment

    (lr_return / investment * 100).round(2)

    # retraining final model on full data
    gs.best_estimator_.fit(X_transformed, y)

    # Saving model and features
    model_data = pd.Series({"model": gs, "features": featured_columns})

    # saving model
    pickle.dump(model_data, open(os.path.join(MODEL_DIR, "lr_model.pkl"), "wb"))
    return None
