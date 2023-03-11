"""Function(s) for cleaning the data set(s)."""

import pandas as pd


def harmonize_columns(df1, df2):
    """This function harmonizes the columns of two dataframes.

    The columns of the first dataframe are kept and the second one is modified to have the same columns.
    Input:
        df1: first dataframe
        df2: second dataframe
    Output:
        df1: first dataframe with the same columns as the second one
        df2: second dataframe with the same columns as the first one

    """
    common_columns = list(df1.columns.intersection(df2.columns))
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    return df1, df2


def rbind_dataframes(df1, df2):
    """
    This function binds two dataframes by rows
    Input:
        df1: first dataframe
        df2: second dataframe
    Output:
        df: dataframe with the rows of df1 and df2.
    """
    df1, df2 = harmonize_columns(df1, df2)
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

    The if statement checks if the dataframe is already in the dataframe data.
    Input:
        data_source: list of dataframes
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
                ) = harmonize_columns(
                    df1=data_sources[dataset_one],
                    df2=data_sources[dataset_two],
                )
            if dataset_one in added_leagues:
                continue
            else:
                if data.empty:
                    data = data_sources[dataset_one]
                    added_leagues.append(dataset_one)
                if dataset_two in added_leagues:
                    continue
                else:
                    data = rbind_dataframes(data, data_sources[dataset_two])
                    added_leagues.append(dataset_two)

    return data


def convert_to_categorical(df, columns):
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


def convert_to_numeric(df, columns):
    """
    This function converts the columns of a dataframe to numeric
    Input:
        df: dataframe
        columns: list of columns to convert to numeric
    Output:
        df: dataframe with the columns converted to numeric.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def convert_to_integer(df, columns):
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


def convert_column_names_to_sensible_names(name_information, df):
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
