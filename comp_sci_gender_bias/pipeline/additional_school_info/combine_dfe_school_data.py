from comp_sci_gender_bias.getters.dfe_school_data import (
    school_info,
    census,
    ks4_results,
)
from comp_sci_gender_bias import PROJECT_DIR

INFO_KEEP_COLS = [
    "school_unique_reference_number",
    "school_name",
    "local_authority_name",
    "school_address_(1)",
    "school_address_(2)",
    "school_address_(3)",
    "school_town",
    "school_postcode",
    "ofsted_rating",
    "school_sex_type",
]

CENSUS_KEEP_COLS = [
    "school_unique_reference_number",
    "type_of_school",
    "percentage_of_girls_on_roll",
    "percentage_of_boys_on_roll",
    "percentage_pupils_fsm_past_6_years",
]

KS4_RESULTS_KEEP_COLS = [
    "school_unique_reference_number",
    "average_girls_attainment_8_gcse_score",
    "average_boys_attainment_8_gcse_score",
    "percentage_boys_strong_9to5_passes_eng_math_gcses",
    "percentage_girls_strong_9to5_passes_eng_math_gcses",
]

SAVE_PATH = PROJECT_DIR / "inputs/data/dfe_school_info/dfe_combined_dataset.csv"


def combine_dfe_datasets():
    """Combine department for education school info, census and
    key stage 4 datasets together"""
    return (
        school_info()[INFO_KEEP_COLS]
        .merge(
            right=census()[CENSUS_KEEP_COLS],
            on="school_unique_reference_number",
            how="left",
        )
        .merge(
            right=ks4_results()[KS4_RESULTS_KEEP_COLS],
            on="school_unique_reference_number",
            how="left",
        )
        .query("type_of_school == 'State-funded secondary'")
        .replace({"NE": -1, "SUPP": -1})
        .fillna(
            {
                "percentage_of_girls_on_roll": -1,
                "percentage_of_boys_on_roll": -1,
                "percentage_pupils_fsm_past_6_years": -1,
                "average_girls_attainment_8_gcse_score": -1,
                "average_boys_attainment_8_gcse_score": -1,
                "percentage_boys_strong_9to5_passes_eng_math_gcses": -1,
                "percentage_girls_strong_9to5_passes_eng_math_gcses": -1,
            }
        )
        .reset_index(drop=True)
    )


def make_and_save_combined_dfe_data(path=SAVE_PATH):
    """Save combined department for education dataset to
    specified path"""
    combine_dfe_datasets().to_csv(path)


if __name__ == "__main__":
    make_and_save_combined_dfe_data()
