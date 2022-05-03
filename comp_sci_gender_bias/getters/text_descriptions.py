import json
from comp_sci_gender_bias import PROJECT_DIR


def text_descriptions(subject: str) -> dict:
    """Returns a dict with the course descriptions.
    Args:
        subject: compsci or geo

    Returns:
        dict where...
            key: school id (see school metadata table)
            value: text description (unprocessed text)
    """

    with open(f"{PROJECT_DIR}/inputs/data/{subject}_descr.json", "r") as infile:
        return json.load(infile)
