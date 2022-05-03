# Parse BIT raw data

import logging
import json
import os
import docx

from comp_sci_gender_bias import PROJECT_DIR

DATA_DIR = f"{PROJECT_DIR}/inputs/data/"
BIT_DIRS = ["bit_compsci_descriptions", "bit_geo_descriptions"]


def getText(filename: str) -> str:
    """Reads and parses a doc file from BIT"""

    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return " ".join(fullText).strip()


for dir_, name in zip(BIT_DIRS, ["compsci", "geo"]):
    logging.info(f"parsing {name}")
    descr_path = os.path.join(DATA_DIR, dir_)
    print(descr_path)
    docx_files = os.listdir(descr_path)

    # In this dict the keys are schools ids and the values the text descriptions
    # of a subject
    text_dict = {
        file_name.split("_")[0]: getText(f"{descr_path}/{file_name}")
        for file_name in docx_files
    }

    with open(
        f"{PROJECT_DIR}/inputs/data/{name}_descr.json",
        "w",
    ) as outfile:
        json.dump(text_dict, outfile, indent=1)
