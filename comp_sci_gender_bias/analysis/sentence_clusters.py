import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.utils.io import make_path_if_not_exist
from comp_sci_gender_bias.getters.sentence_clusters import get_sentence_clusters


SUBJECTS = ["cs", "geo", "drama"]
CMAP = {"cs": "#B1D1FC", "geo": "#90E4C1", "drama": "#FFB07C"}

OUT_DIR = PROJECT_DIR / "outputs/figures/sentence_clusters/"


def category_token_distribution(sentences: pd.DataFrame) -> pd.DataFrame:
    """Calculates the percentage of descriptions belonging to a cluster type.

    Args:
        sentences: Dataframe of course description sentences, their cluster
            types and token count.

    Returns:
        Dataframe containing the percent of tokens in each cluster type by
            document.
    """
    total_tokens = sentences.groupby("description_id").sum().loc[:, "n_tokens"]
    category_token_dist = (
        sentences.pivot_table(
            index="description_id",
            columns="cluster_type",
            values="n_tokens",
            aggfunc="sum",
        )
        .divide(total_tokens, axis=0)
        .fillna(0)
        .multiply(100)
    )
    return category_token_dist


def category_token_distribution_boxplot(distribution, subj):
    _, ax = plt.subplots()
    sns.boxplot(data=distribution, color=CMAP[subj], ax=ax)
    ax.set_xlabel("Sentence Type")
    ax.set_ylabel("% of Tokens")

    out_dir = OUT_DIR / f"{subj}_sent_category_boxplot/"
    make_path_if_not_exist(out_dir)

    plt.savefig(out_dir / "fig.png", dpi=300)
    plt.close()
    distribution.to_csv(out_dir / "data.csv")


if __name__ == "__main__":

    make_path_if_not_exist(OUT_DIR)

    for subj in SUBJECTS:
        sents = get_sentence_clusters(subj)
        category_token_dist = category_token_distribution(sents)
        category_token_distribution_boxplot(category_token_dist, subj)
