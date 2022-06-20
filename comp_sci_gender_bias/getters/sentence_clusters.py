import pandas as pd

from comp_sci_gender_bias import PROJECT_DIR


def get_sentence_clusters(subject: str) -> pd.DataFrame:
    """Gets sentence clusters for one subject.

    Args:
        subject: Subject name ('cs', 'geo' or 'drama').

    Returns:
        Dataframe of subject course description sentences and their cluster membership.
    """
    data_dir = (
        PROJECT_DIR / f"outputs/sentence_clusters/{subject}_sentence_clusters.csv"
    )
    return pd.read_csv(data_dir)
