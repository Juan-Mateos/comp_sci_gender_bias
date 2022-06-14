"""
Utils for processing pandas data
"""

import pandas as pd


def cols_replace_space_and_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces with underscores and make lowercase
    for column names in dataframe"""
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df


def remove_nonalphanum_lowercase(column: pd.Series) -> pd.Series:
    """Return input column with non alphanumeric characters
    removed and the text set to lowercase
    """
    return (
        column.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )


def remove_fw_slash(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove '/' from the end of all rows in specified
    df and col"""
    df[col] = df[col].map(lambda x: str(x)[:-1] if str(x)[-1] == "/" else x)
    return df


def remove_http_https(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove http:// and https:// from the start of all rows in
    specified df and col"""
    df[col] = df[col].map(lambda x: str(x).replace("https://", ""))
    df[col] = df[col].map(lambda x: str(x).replace("http://", ""))
    return df


def remove_new_para(df: pd.DataFrame, col: str) -> pd.DataFrame:
    "Remove \n from specified df and col"
    df[col] = df[col].map(lambda x: str(x).replace("\n", ""))
    return df


def add_www(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add www. to start of all rows in specified
    df and col"""
    df[col] = df[col].map(lambda x: f"www.{str(x)}" if str(x)[:4] != "www." else x)
    return df


def clean_website_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Clean website text to make it easier to join by:
        - removing / from end of website
        - remove http/https from start of website
        - remove new paragraph
        - add www. to start of website

    Args:
        df: Dataframe with column containing website text
        col: Column containing website text

    Returns:
        Dataframe with cleaned website text
    """
    return (
        df.pipe(remove_fw_slash, col)
        .pipe(remove_http_https, col)
        .pipe(remove_new_para, col)
        .pipe(add_www, col)
    )


def percent_to_float(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert values from percent to float in specified df and col"""
    df[col] = df[col].map(lambda x: float(str(x).strip("%")) / 100)
    return df
