import hdbscan
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import spacy_udpipe
from toolz import pipe
from typing import List, Iterable
from umap import UMAP

from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.getters.scraped_data import scraped_data_no_extra_whitespace
from comp_sci_gender_bias.utils.io import make_path_if_not_exist


SUBJECTS = ["cs", "geo", "drama"]
MIN_CLUSTER_SIZES = {
    "cs": 8,
    "geo": 8,
    "drama": 6,
}


sns.set_style("whitegrid", {"axes.grid": False})


def generate_sents(
    descriptions: Iterable[str], index: Iterable[int], min_len: int = 4
) -> pd.DataFrame:
    """Generates a dataframe of sentences from a dataframe of descriptions.

    Args:
        descriptions: List-like of course descritpions.
        index: Unique IDs for course descriptions.
        min_len: Any sentences shorter than this will be dropped.

    Returns:
        Dataframe where one column has sentences from the course descriptions
        and the other contains the unique ID of the description.
    """
    description_ids = []
    sents = []
    for idx, desc in zip(index, descriptions):
        doc = nlp(desc)
        for sent in doc.sents:
            if len(sent) >= min_len:
                sents.append(sent.text)
                description_ids.append(idx)
    return pd.DataFrame({"description_id": description_ids, "sentence": sents})


def scraped_sents(subject: str, min_len: int = 4) -> pd.DataFrame:
    """Generates sentences from descriptions of a particular subject.

    Args:
        subject: 'cs', 'geo' or 'drama'
        min_len: Any sentences shorter than this will be dropped.
    """
    scraped = scraped_data_no_extra_whitespace()
    return generate_sents(scraped[subject], scraped.index, min_len)


def embed(texts: List[str]) -> np.array:
    """Creates an embedding for each string in a corpus.

    Args:
        texts: Corpus of texts to embed.

    Returns:
        An embedding for each element in `texts`.
    """
    PRE_TRAINED_MODEL = "all-MiniLM-L6-v2"
    model = SentenceTransformer(PRE_TRAINED_MODEL)
    return model.encode(texts)


def reduce_dimensions(
    a: np.array, n_components_svd: int = 30, n_components_umap: int = 2
) -> np.array:
    """Reduces the dimensionality of a 2d array using UMAP.

    Args:
        a: `m` x `n` array of vectors to reduce.
        n_components_svd: Number of dimensions to reduce in intial reduction
            using SVD. If less than 1, then this step is skipped. Use if
            `n` is large (greater than 100).
        n_components_umap: Number of final dimensions to reduce using UMAP.

    Returns:
        Array with reduced dimensions.
    """
    umap = UMAP(n_components=n_components_umap, metric="cosine")
    if n_components_svd > 0:
        svd = TruncatedSVD(n_components=n_components_svd)
        return pipe(a, svd.fit_transform, umap.fit_transform)
    else:
        return pipe(a, umap.fit_transform)


def cluster_sents(embeddings: np.array, min_cluster_size: int) -> np.array:
    """Clusters sentences using HDBSCAN and returns labels for all points
    (including 'outliers').

    Args:
        embeddings: Embeddings of sentences to be clustered.
        min_cluster_size: Equivalent to the parameter of the same name for hdbscan.HSBSCAN

    Returns:
        Array of labels for sentences.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
    )
    clusterer.fit(sent_umap)
    return np.argmax(hdbscan.membership_vector(clusterer, embeddings), axis=1)


if __name__ == "__main__":

    spacy_udpipe.download("en")
    nlp = spacy_udpipe.load("en")

    for subject in SUBJECTS:
        sents = scraped_sents(subject)
        sent_embeddings = embed(list(sents["sentence"])).astype("double")
        sent_umap = reduce_dimensions(sent_embeddings)

        cluster_labels = cluster_sents(sent_umap, MIN_CLUSTER_SIZES[subject])
        sents["cluster"] = cluster_labels

        out_dir = PROJECT_DIR / "outputs/sentence_clusters/"
        make_path_if_not_exist(out_dir)
        sents.to_csv(out_dir / f"{subject}_sentence_clusters.csv", index=False)
