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
