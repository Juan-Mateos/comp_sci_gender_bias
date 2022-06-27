from comp_sci_gender_bias import PROJECT_DIR, logger
import pandas as pd
from typing import Union


def mean_gender_differences(
    data_source: str, word_removal: Union[None, str]
) -> pd.DataFrame:
    """Load dataframe of the mean gender differences for each
    subject and POS.
    To load this file, it must be created first by running:
    comp_sci_gender_bias/pipeline/glove_differences/make_mean_differences.py

    Args:
        data_source: 'scraped' or 'bit'
        word_removal: 'crucial', 'optional' or None

    Returns:
        Dataframe of the mean gender differences for each
            subject and POS
    """
    word_removal = "no" if word_removal is None else word_removal
    try:
        return pd.read_csv(
            PROJECT_DIR
            / f"outputs/mean_differences/mean_differences_pos_{data_source}_remove_{word_removal}_words.csv"
        )
    except FileNotFoundError:
        logger.error(
            "FileNotFoundError: To create the file, run comp_sci_gender_bias/pipeline/glove_differences/make_mean_differences.py to generate the file"
        )
