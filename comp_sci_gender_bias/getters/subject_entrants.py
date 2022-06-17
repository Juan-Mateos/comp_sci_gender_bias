from comp_sci_gender_bias import PROJECT_DIR, logger
import pandas as pd


def subject_entrants() -> pd.DataFrame:
    """Returns dataframe of entrants into GCSE subjects in 2021
    Source: https://explore-education-statistics.service.gov.uk/find-statistics/key-stage-4-performance-revised/2020-21
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/subject_entrants/2021_subject_s1245789_data.csv"
    )


def girls_entry_percentage() -> pd.DataFrame:
    """Returns dataframe of total entrants, girls entrants and
    girls entry percentage into GCSE subjects in 2021.
    To load this file, it must be created first by running:
    comp_sci_gender_bias/pipeline/subject_entry/girls_subject_entry.py
    """
    try:
        return pd.read_csv(
            PROJECT_DIR / "outputs/girls_entry_percentage/girls_entry_percentage.csv"
        )
    except FileNotFoundError:
        logger.error(
            "FileNotFoundError: To create this file run: comp_sci_gender_bias/pipeline/subject_entry/girls_subject_entry.py"
        )
