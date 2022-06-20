from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.getters.subject_entrants import subject_entrants
from comp_sci_gender_bias.utils.io import make_path_if_not_exist
import pandas as pd

GIRLS_ENTRY_PERCENTAGE_SAVE_PATH = PROJECT_DIR / "outputs/girls_entry_percentage"

ENTRANTS_KEEP_COLS = [
    "school_type",
    "country_name",
    "time_period",
    "characteristic_gender",
    "subject",
    "subject_entry",
]

TOTAL_DROP_COLS = [
    "characteristic_gender",
    "country_name",
    "time_period",
    "school_type",
]

GIRLS_ENTRY_PERCENTAGE_KEEP_COLS = [
    "time_period",
    "country_name",
    "school_type",
    "subject",
    "total_entry",
    "girls_entry",
    "girls_entry_percent",
]


def create_total_entry(entrants: pd.DataFrame) -> pd.DataFrame:
    """From entrants data, create dataframe containing subject
    and total_entry"""
    return (
        entrants.query("characteristic_gender == 'Total'")
        .rename(columns={"subject_entry": "total_entry"})
        .drop(columns=TOTAL_DROP_COLS)
    )


def create_girls_entry(entrants: pd.DataFrame) -> pd.DataFrame:
    """From entrants data, create dataframe containing:
    - school_type
    - country_name
    - time_period
    - subject
    - girls_entry
    """
    return (
        entrants.query("characteristic_gender == 'Girls'")
        .rename(columns={"subject_entry": "girls_entry"})
        .drop(columns="characteristic_gender")
    )


def create_girls_entry_percentage(
    total: pd.DataFrame, girls: pd.DataFrame
) -> pd.DataFrame:
    """Join dataframes containing total entry and girls entry
    together and calculate girls entry percentage"""
    return (
        total.merge(right=girls, on="subject", how="left")
        .drop_duplicates()
        .assign(
            girls_entry_percent=lambda x: round(
                x["girls_entry"] / x["total_entry"] * 100, 2
            )
        )
        .sort_values("girls_entry_percent", ascending=False)
        .reset_index(drop=True)[GIRLS_ENTRY_PERCENTAGE_KEEP_COLS]
    )


if __name__ == "__main__":
    entrants = subject_entrants().query("school_type == 'All state-funded'")[
        ENTRANTS_KEEP_COLS
    ]
    total = create_total_entry(entrants)
    girls = create_girls_entry(entrants)
    girls_entry_percentage = create_girls_entry_percentage(total, girls)

    make_path_if_not_exist(GIRLS_ENTRY_PERCENTAGE_SAVE_PATH)
    girls_entry_percentage.to_csv(
        GIRLS_ENTRY_PERCENTAGE_SAVE_PATH / "girls_entry_percentage.csv", index=False
    )
