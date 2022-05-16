from comp_sci_gender_bias.pipeline.additional_school_info.combine_dfe_school_data import (
    save_combined_dfe_data,
)
from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd

COMBINED_DFE_PATH = PROJECT_DIR / "inputs/data/dfe_school_info/dfe_combined_dataset.csv"


def dfe_combined_school_data() -> pd.DataFrame:
    """Load dataframe of combined department of education school
    datasets. This includes unique school reference number, location,
    ofsted rating, gender split, average attainment by gender.
    The combined dataset will be created if it doesn't exist already.
    """
    if not COMBINED_DFE_PATH.exists():
        save_combined_dfe_data()
    return pd.read_csv(COMBINED_DFE_PATH, index_col=0)
