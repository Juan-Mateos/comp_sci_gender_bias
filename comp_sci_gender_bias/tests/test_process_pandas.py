import pandas as pd
from comp_sci_gender_bias.utils.process_pandas import (
    cols_replace_space_and_lowercase,
    remove_nonalphanum_lowercase,
)


def test_cols_replace_space_and_lowercase():
    df = pd.DataFrame(data={"Col 1": [1, 2], "Col 2": [3, 4]})
    assert list(cols_replace_space_and_lowercase(df).columns) == ["col_1", "col_2"]


def test_remove_nonalphanum_lowercase():
    column = pd.Series(data={"school_name": "Onslow St Audrey's School"})
    assert remove_nonalphanum_lowercase(column).equals(
        pd.Series(data={"school_name": "onslow st audreys school"})
    )
