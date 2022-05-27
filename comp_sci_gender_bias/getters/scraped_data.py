from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd

SCRAPED_DATA_PATH = (
    PROJECT_DIR / "inputs/data/scraped_data/semi_manual/course_descriptions.csv"
)


def scraped_data() -> pd.DataFrame:
    """Returns dataframe of the scraped school descriptions
    for Computer Science, Drama and Geography"""
    return pd.read_csv(SCRAPED_DATA_PATH)
