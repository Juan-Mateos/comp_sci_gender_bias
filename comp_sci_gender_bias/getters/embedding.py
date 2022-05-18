from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.utils.io import load_pickle
import torch
from typing import Union
import numpy as np
import pathlib

EMBEDDING_DIR = PROJECT_DIR / "outputs/embeddings"


def load_embedding(
    subject: str, embedding_dir: pathlib.Path = EMBEDDING_DIR
) -> Union[torch.Tensor, np.ndarray]:
    """Load sentence embedding for school course descriptions

    Args:
        subject: Subject
        embedding_dir: Path that embeddings are saved to

    Returns:
        Sentence embedding for the specified subject
    """
    return load_pickle(embedding_dir / f"{subject}_embedding.pkl")
