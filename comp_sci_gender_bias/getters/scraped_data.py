from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd
import re

SCRAPED_DATA_PATH = (
    PROJECT_DIR / "inputs/data/scraped_data/semi_manual/course_descriptions.csv"
)


def scraped_data() -> pd.DataFrame:
    """Returns dataframe of the scraped school descriptions
    for Computer Science, Drama and Geography"""
    return pd.read_csv(SCRAPED_DATA_PATH)


def scraped_data_no_extra_whitespace() -> pd.DataFrame:
    """Gets course descriptions scraped by Nesta for all subjects and removes
    extraeneous whitepsace.

    Returns:
        Dataframe of clean course descriptions for all subjects.
    """
    scraped = scraped_data()

    scraped = scraped.drop("Website", axis=1).rename(
        columns={"CompSci": "cs", "Drama": "drama", "Geography": "geo"}
    )
    for subj in ["cs", "geo", "drama"]:
        scraped[subj] = scraped[subj].apply(lambda x: re.sub("\s+", " ", x)).str.strip()

    return scraped
