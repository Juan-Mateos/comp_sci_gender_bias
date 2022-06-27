from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd


def categorised_words(subject: str) -> pd.DataFrame:
    """Load dataframe containing subject specific words

    Args:
        subject: "geo" or "cs"

    Returns:
        Dataframe of words in a corpus and columns for whether
        they have been classified as a 'crucial subject specific word'
        or an 'optional subject specific word'
    """
    return pd.read_csv(
        PROJECT_DIR
        / f"inputs/data/subject_specific_terminology/{subject}_words_categorised.csv",
        index_col=0,
    )


def subject_specific_words(subject: str, specific_word_type: str) -> list:
    """Create list of subject specific words

    Args:
        subject: "geo" or "cs"
        specific_word_type: "crucial" or "optional"

    Returns:
        List of subject specific words
    """
    cat_words = categorised_words(subject)
    return cat_words.query(f"`{specific_word_type} subject specific word` == 1")[
        "word"
    ].to_list()
