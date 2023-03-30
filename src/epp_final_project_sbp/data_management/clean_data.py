"""Function(s) for cleaning the data set(s)."""

import numpy as np
import pandas as pd


def manage_data(
    data,
    name_information,
    considered_features,
    categorical_features,
    integer_features,
):
    """This function handles initial data cleaning, sorting the data and ensuring the
    right data types.

    Input:
        data, pandas dataframe: raw data
        name_information, pandas dataframe: naming information about the data
        considered_features, list: list of features to be considered
        categorical_featuers, list: list of categorical features
        integer_features, list: list of integer features
    Output:
        data, pandas dataframe: cleaned and sorted data

    """
    assert isinstance(
        data,
        pd.DataFrame,
    ), "The data variable must be a pandas dataframe."
    assert isinstance(
        name_information,
        pd.DataFrame,
    ), "The name_information must be a pandas dataframe."
    assert isinstance(considered_features, list), "Considered_features must be a list."
    assert isinstance(
        categorical_features,
        list,
    ), "The categorical_features must be a list."
    assert isinstance(integer_features, list), "The integer_features must be a list."

    data = __convert_column_names_to_sensible_names(
        name_information=name_information,
        df=data,
    )
    try:
        data = data[considered_features]
    except KeyError:
        pass
    try:
        data = __convert_to_categorical(df=data, columns=categorical_features)
    except KeyError:
        pass
    try:
        data["Date"] = pd.to_datetime(
            data["Date"],
            format="%d/%m/%Y",
            errors="coerce",
        ).fillna(pd.to_datetime(data["Date"], format="%d/%m/%y", errors="coerce"))
    except KeyError:
        pass

    data.sort_values(by=["Date"], inplace=True)
    data = data.reset_index()
    try:
        data = __convert_to_integer(df=data, columns=integer_features)
    except KeyError:
        pass
    return data


def __harmonize_columns(df1, df2):
    """This function takes the union of two dataframes and returns to dataframes with
    these columns.

    Columns, which are only present in one of the two are created as a column and filled with NaNs in the other dataframe.
    Input:
        df1: first dataframe
        df2: second dataframe
    Output:
        df1: first dataframe with the same columns as the second one
        df2: second dataframe with the same columns as the first one

    """
    df1_columns = df1.columns
    df2_columns = df2.columns
    df1_columns_not_in_df2 = df1_columns.difference(df2_columns)
    df2_columns_not_in_df1 = df2_columns.difference(df1_columns)
    for column in df1_columns_not_in_df2:
        df2[column] = np.nan
    for column in df2_columns_not_in_df1:
        df1[column] = np.nan
    return df1, df2


def __rbind_dataframes(df1, df2):
    """
    This function binds two dataframes by rows
    Input:
        df1: first dataframe
        df2: second dataframe
    Output:
        df: dataframe with the rows of df1 and df2.
    """
    df = pd.concat([df1, df2], axis=0)
    return df


def delete_rows_with_just_nans(df):
    """
    This function deletes the rows of a dataframe that only contain NaNs
    Input:
        df: dataframe
    Output:
        df: dataframe without the rows that only contain NaNs.
    """
    df = df.dropna(how="all")
    return df


def rbind_list_of_dataframes(data_sources, data):
    """This function binds a list of dataframes by rows into one dataframe The for loop
    iterates over the list of dataframes and binds them to the dataframe data.

    The if statements check if the dataframes are the same one and if they have already been added.
    Input:
        data_source: dict of dataframes
    Output:
        data: dataframe with the rows of the dataframes in the list.

    """
    added_leagues = []
    for dataset_one in data_sources:
        for dataset_two in data_sources:
            if dataset_one != dataset_two:
                (
                    data_sources[dataset_one],
                    data_sources[dataset_two],
                ) = __harmonize_columns(
                    df1=data_sources[dataset_one],
                    df2=data_sources[dataset_two],
                )
        if dataset_one in added_leagues:
            continue
        else:
            if data.empty:
                data = pd.DataFrame()
                data = data_sources[dataset_one]
                added_leagues.append(dataset_one)
            else:
                data = __rbind_dataframes(data, data_sources[dataset_one])
                added_leagues.append(dataset_one)
        if dataset_two in added_leagues:
            continue
        else:
            data = __rbind_dataframes(data, data_sources[dataset_two])
            added_leagues.append(dataset_two)
    return data


def __convert_to_categorical(df, columns):
    """
    This function converts the columns of a dataframe to categorical
    Input:
        df: dataframe
        columns: list of columns to convert to categorical
    Output:
        df: dataframe with the columns converted to categorical.
    """
    for col in columns:
        df[col] = df[col].astype("category")
    return df


def __convert_to_integer(df, columns):
    """
    This function converts the columns of a dataframe to integer
    Input:
        df: dataframe
        columns: list of columns to convert to integer
    Output:
        df: dataframe with the columns converted to integer.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def __convert_column_names_to_sensible_names(name_information, df):
    """
    This function replaces the column names of a dataframe with sensible names
    The current columnnames are present in the column "current_names", the sensible names are present in the column "sensible_names"
    Not all columns need to be replaced, only the ones that are present in the dataframe
    Input:
        name_information: dictionary with the column names and the sensible names
        df: dataframe
    Output:
        df: dataframe with sensible column names.
    """
    for i in range(len(name_information["current_names"])):
        if name_information["current_names"][i] in df.columns:
            df = df.rename(
                columns={
                    name_information["current_names"][i]: name_information[
                        "sensible_names"
                    ][i],
                },
            )
    return df
