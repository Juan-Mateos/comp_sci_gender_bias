import json
import re
import pandas as pd
from toolz import pipe
from comp_sci_gender_bias import PROJECT_DIR


def text_descriptions(subject: str) -> dict:
    """Reads a dict with the course descriptions.
    Args:
        subject: compsci or geo

    Returns:
        dict where...
            key: school id (see school metadata table)
            value: text description (unprocessed text)
    """

    with open(f"{PROJECT_DIR}/inputs/data/{subject}_descr.json", "r") as infile:
        return json.load(infile)


def school_table() -> pd.DataFrame:
    """Reads a table with school metadata including lookup between ids and names"""

    return pipe(
        pd.read_csv(f"{PROJECT_DIR}/inputs/data/school_master_table.csv"),
        lambda df: df.rename(
            columns={col: re.sub(" ", "_", col.strip().lower()) for col in df.columns}
        ),
        lambda df: df.rename(columns={"number": "id"}),
    )
