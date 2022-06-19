import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textstat

import numpy.typing as npt
from typing import List

from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.getters.scraped_data import scraped_data_no_extra_whitespace
from comp_sci_gender_bias.utils.io import make_path_if_not_exist


SCRAPED_DIR = (
    PROJECT_DIR / "inputs/data/scraped_data/semi_manual/course_descriptions.csv"
)
SUBJECTS = ["cs", "geo", "drama"]
SUBJECT_NAMES = {
    "cs": "CS",
    "geo": "Geography",
    "drama": "Drama",
}
SCORE_NAMES = {"fr": "Flesch Reading Ease", "dc": "Dale-Chall Readability"}
CMAP = {"cs": "#B1D1FC", "geo": "#90E4C1", "drama": "#FFB07C"}

sns.set_style("whitegrid", {"axes.grid": False})


def calculate_subject_readability(
    descriptions: pd.DataFrame,
    subjects: List[str],
) -> pd.DataFrame:
    """Calculates the Flesch Reading Ease and Dale Chall Readability scores for all
    course descriptions.

    Returns:
        Dataframe of course descriptions updated with readability scores.
    """
    for subj in subjects:
        descriptions["fr_" + subj] = [
            textstat.flesch_reading_ease(d) for d in descriptions[subj]
        ]
        descriptions["dc_" + subj] = [
            textstat.dale_chall_readability_score(d) for d in descriptions[subj]
        ]

    return descriptions


def _metric_to_prefix(metric: str) -> str:
    if metric == "flesch":
        return "fr_"
    elif metric == "dale-chall":
        return "dc_"


def _metric_to_name(metric: str) -> str:
    if metric == "flesch":
        return "Flesch Reading Ease"
    elif metric == "dale-chall":
        return "Dale-Chall Readability"


def readability_boxplot(
    descriptions: pd.DataFrame,
    metric: str = "flesch",
):
    """Creates and saves boxplots (and underlying data) of readability scores."""
    prefix = _metric_to_prefix(metric)
    name = _metric_to_name(metric)

    data = descriptions[[prefix + s for s in SUBJECTS]].melt(
        value_name=name, var_name="Subject"
    )
    data["Subject"] = data["Subject"].str.replace(prefix, "")
    data["Color"] = data["Subject"].map(CMAP)
    data["Subject"] = data["Subject"].map(SUBJECT_NAMES)

    _, ax = plt.subplots()
    sns.boxplot(
        data=data,
        x="Subject",
        y=name,
        palette=dict(zip(SUBJECT_NAMES.values(), CMAP.values())),
        ax=ax,
    )

    plot_dir = PROJECT_DIR / f"outputs/figures/readability/{metric}_all_subj_boxplot"
    make_path_if_not_exist(plot_dir)

    plt.savefig(plot_dir / "fig.png", dpi=300)
    data.to_csv(plot_dir / "data.csv", index=False)


def readability_stats_table(scores: pd.DataFrame):
    """Generates tables with descriptive stats for readability scores per
    subject.

    Args:
        scores: Dataframe of readability scores.
    """

    def readability_subject_name(name):
        name = name.split("_")
        return f"{SCORE_NAMES[name[0]]} {SUBJECT_NAMES[name[1]]}"

    stats = scores.describe().drop("count")
    stats.columns = [readability_subject_name(c) for c in stats.columns]

    col_names = {
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Min",
        "max": "Max",
        "25%": "Lower Quartile",
        "75%": "Upper Quartile",
        "median": "Median",
    }

    stats = stats.T.rename(columns=col_names)
    out_dir = PROJECT_DIR / "outputs/tables/readability/descriptive_stats"
    make_path_if_not_exist(out_dir)
    stats.to_csv(out_dir / "descriptive_stats.csv")
    stats.to_markdown(out_dir / "descriptive_stats.md")


def find_nearest_idx(x: npt.ArrayLike, value: float) -> int:
    """Locates the index of the element in `x` that is closest to `value`.

    Args:
        x: Array of values.
        value: The value to minimise the search.

    Returns:
        idx: Index of the array element with the closest value.
    """
    x = np.asarray(x)
    idx = (np.abs(x - value)).argmin()
    return idx


def descriptions_at_subj_readability_quantiles(
    scores: pd.DataFrame, metric: str = "flesch", quantiles=[0.25, 0.5, 0.75]
):
    """Finds examples of course descriptions within a subject that have
    readability scores which are the closest to a set of quantiles in the
    distribution of scores for that subject.

    Args:
        scores: Dataframe of descriptions and their readability scores for
            all subjects.
        metric: `flesch` or `dale-chall`
        quantiles: Iterable of quantiles in the range 0 to 1.
    """

    prefix = _metric_to_prefix(metric)

    records = []
    for subj in SUBJECTS:
        col = f"{prefix}{subj}"
        for quantile in quantiles:
            quantile_score = np.quantile(scores[col], quantile)
            quantile_desc_idx = find_nearest_idx(scores[col], quantile_score)
            desc = scores[subj][quantile_desc_idx]

            records.append(
                {
                    "Score": _metric_to_name(metric),
                    "Subject": SUBJECT_NAMES[subj],
                    "Quantile": quantile,
                    "Example": desc,
                }
            )

    examples = pd.DataFrame.from_records(records)
    out_dir = PROJECT_DIR / "outputs/tables/readability/quantile_examples"
    make_path_if_not_exist(out_dir)
    examples.to_csv(out_dir / "quantile_examples.csv", index=False)
    examples.to_markdown(out_dir / "quantile_examples.md", index=False)


if __name__ == "__main__":

    scraped = scraped_data_no_extra_whitespace()
    readability = calculate_subject_readability(scraped, SUBJECTS)

    readability_stats_table(readability)

    for metric in ["flesch", "dale-chall"]:
        readability_boxplot(readability, metric=metric)
        descriptions_at_subj_readability_quantiles(readability, metric=metric)
