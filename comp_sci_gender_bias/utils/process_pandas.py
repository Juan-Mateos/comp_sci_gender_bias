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
        .str.replace("[^a-z0-9 ]", "")
        .str.strip()
    )
