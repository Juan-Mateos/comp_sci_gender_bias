from comp_sci_gender_bias.pipeline.additional_school_info.combine_dfe_school_data import (
    save_combined_dfe_data,
)
from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd

DFE_DIR = PROJECT_DIR / "inputs/data/dfe_school_info/"


def dfe_combined_school_data():
    """Load dataframe of combined department of
    education school datasets. If it doesn't exist,
    create the dataset.
    """
    combined_path = DFE_DIR / "dfe_combined_dataset.csv"
    if not combined_path.exists():
        save_combined_dfe_data()
    return pd.read_csv(combined_path, index_col=0)
