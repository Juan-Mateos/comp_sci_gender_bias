from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd

URN_SCHOOL_NAME_LOOKUP_PATH = (
    PROJECT_DIR / "inputs/data/urn_school_lookups/urn_school_lookup_full.csv"
)


def urn_to_school_name_lookup():
    """Load dataframe containing school names and
    related school unique reference number"""
    return pd.read_csv(URN_SCHOOL_NAME_LOOKUP_PATH, index_col=0)
