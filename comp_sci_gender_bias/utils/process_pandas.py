"""
Processing utils
"""

import pandas as pd


def cols_replace_space_and_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces with underscores and make lowercase
    for column names in dataframe"""
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df
