from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd

SCHOOL_LVL_BIAS_WITH_DFE_DATA_PATH = (
    PROJECT_DIR / "outputs/school_level/scraped_schools_urn_dfe.csv"
)


def school_lvl_bias_with_dfe_data() -> pd.DataFrame:
    """Returns a dataframe of the scraped school descriptions
    for Computer Science, Drama and Geography with a column
    for mean gender differences for each subject and the
    secondary DfE data"""
    return pd.read_csv(SCHOOL_LVL_BIAS_WITH_DFE_DATA_PATH, index_col=0)
