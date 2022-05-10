from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd
import pathlib
from comp_sci_gender_bias.utils.process_pandas import cols_replace_space_and_lowercase


DFE_DIR = PROJECT_DIR / "inputs/data/dfe_school_info/"
DFE_2021_DIR = DFE_DIR / "2020-2021/"
DFE_2019_DIR = DFE_DIR / "2018-2019/"


def col_map(path: pathlib.Path, map_from: str, map_to: str) -> dict:
    """Create a lookup mapping dictionary

    Args:
        path: Path to create mapping from
        map_from: Column name to map from
        map_to: Column name to map to

    Returns:
        Lookup mapping dictionary
    """
    return pd.read_csv(path, index_col=0).set_index(map_from).to_dict()[map_to]


def school_info() -> pd.DataFrame:
    """Returns a dataframe containing school information as of 2021
    including:
        - school_unique_reference_number
        - school_name
        - local_authority_name
        - school_address
        - ofsted_rating
    """
    school_info_col_map = col_map(
        DFE_2021_DIR / "school_information_meta.csv", "Field Name", "Description"
    )
    return (
        pd.read_csv(DFE_2021_DIR / "england_school_information.csv")
        .rename(columns=school_info_col_map)
        .pipe(cols_replace_space_and_lowercase)
        .rename(
            columns={
                "indicates_whether_it's_a_mixed_or_single_sex_school": "school_sex_type"
            }
        )
    )


def census() -> pd.DataFrame:
    """Returns a dataframe containing school census data as of 2021
    including:
        - school_unique_reference_number
        - percentage_of_girls_on_roll
        - percentage_of_boys_on_roll
        - percentage_pupils_fsm_past_6_years
    """
    census_col_map = col_map(
        DFE_2021_DIR / "census_meta.csv", "Field Reference", "Field Name"
    )
    return (
        pd.read_csv(DFE_2021_DIR / "england_census.csv")
        .rename(columns=census_col_map)
        .pipe(cols_replace_space_and_lowercase)
        .rename(
            columns={
                "percentage_of_pupils_eligible_for_fsm_at_any_time_during_the_past_6_years": "percentage_pupils_fsm_past_6_years",
            }
        )
        .query("school_unique_reference_number != 'NAT'")
        .astype({"school_unique_reference_number": "int64"})
    )


def ks4_results() -> pd.DataFrame:
    """Returns a dataframe containing Key stage 4 (GCSE) results from
    2019 including:
        - school_unique_reference_number
        - average_girls_attainment_8_gcse_score
        - average_boys_attainment_8_gcse_score
        - percent_boys_strong_9to5_passes_eng_math_gcses
        - percent_girls_strong_9to5_passes_eng_math_gcses
    """
    ks4_col_map = col_map(
        DFE_2019_DIR / "ks4_final_meta.csv", "Metafile heading", "Metafile description"
    )
    return (
        pd.read_csv(DFE_2019_DIR / "2018-2019_england_ks4final.csv")
        .rename(columns=ks4_col_map)
        .pipe(cols_replace_space_and_lowercase)
        .rename(
            columns={
                "average_attainment_8_score_per_girl_-_gcse_only": "average_girls_attainment_8_gcse_score",
                "average_attainment_8_score_per_boy_-_gcse_only": "average_boys_attainment_8_gcse_score",
                "%_of_boys_achieving_strong_9-5_passes_in_both_english_and_mathematics_gcses_": "percent_boys_strong_9to5_passes_eng_math_gcses",
                "%_of_girls_achieving_strong_9-5_passes_in_both_english_and_mathematics_gcses_": "percent_girls_strong_9to5_passes_eng_math_gcses",
            }
        )
        .dropna(subset="school_unique_reference_number")
        .astype({"school_unique_reference_number": "int64"})
    )
