from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.getters.school_data import school_table
from comp_sci_gender_bias.getters.dfe_combined_school_data import (
    dfe_combined_school_data,
)
from comp_sci_gender_bias.utils.process_pandas import remove_nonalphanum_lowercase
import pandas as pd

URN_SCHOOL_LOOKUPS_PATH = PROJECT_DIR / "inputs/data/urn_school_lookups"
SAVE_FULL_LOOKUP_PATH = URN_SCHOOL_LOOKUPS_PATH / "urn_school_lookup_full.csv"
MANUAL_LOOKUP_PATH = URN_SCHOOL_LOOKUPS_PATH / "urn_school_lookup_manual.csv"


def make_auto_lookup() -> pd.DataFrame:
    """Produce a unique reference number to school name lookup by matching
    between the combined Department for Education dataset and the school master table.
    There is some simple processing performed on the school names to enable more matches."""
    schools = school_table()
    dfe = dfe_combined_school_data()
    dfe["school_name_clean"] = remove_nonalphanum_lowercase(dfe["school_name"])
    schools["school_name_clean"] = remove_nonalphanum_lowercase(schools["school_name"])
    return (
        schools.merge(
            right=dfe,
            left_on="school_name_clean",
            right_on="school_name_clean",
            how="left",
        )
        .dropna(subset="school_unique_reference_number")[
            [
                "school_unique_reference_number",
                "school_name_x",
            ]
        ]
        .rename(columns={"school_name_x": "school_name"})
    )


def combine_auto_manual_lookups() -> pd.DataFrame:
    """Combine the automated and manual school reference number
    to school name lookups"""
    manual_lookup = pd.read_csv(MANUAL_LOOKUP_PATH)
    auto_lookup = make_auto_lookup()
    return pd.concat([manual_lookup, auto_lookup]).astype(
        {"school_unique_reference_number": int}
    )


def save_full_urn_school_lookup(path=SAVE_FULL_LOOKUP_PATH):
    """Save full school unique reference number to school
    name lookup to specified path"""
    combine_auto_manual_lookups().to_csv(path)


if __name__ == "__main__":
    save_full_urn_school_lookup()
