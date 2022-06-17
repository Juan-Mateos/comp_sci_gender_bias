from comp_sci_gender_bias import PROJECT_DIR, logger
import pandas as pd


def mean_gender_differences(data_source) -> pd.DataFrame:
    """Returns dataframe of the mean gender differences for each
    subject and POS for the specified data_source ('scraped' or 'bit').
    To load this file, it must be created first by running:
    comp_sci_gender_bias/pipeline/glove_differences/make_mean_differences.py
    """
    try:
        return pd.read_csv(
            PROJECT_DIR
            / f"outputs/mean_differences/mean_differences_pos_{data_source}.csv"
        )
    except FileNotFoundError:
        logger.error(
            "FileNotFoundError: To create the file, run comp_sci_gender_bias/pipeline/glove_differences/make_mean_differences.py to generate the file"
        )
