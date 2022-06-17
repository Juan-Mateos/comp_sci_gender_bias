from comp_sci_gender_bias.getters.school_lvl_bias_with_dfe_data import (
    school_lvl_bias_with_dfe_data,
)
from comp_sci_gender_bias.getters.subject_entrants import girls_entry_percentage
from comp_sci_gender_bias.getters.mean_gender_differences import mean_gender_differences
from comp_sci_gender_bias.utils.io import make_path_if_not_exist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from comp_sci_gender_bias import PROJECT_DIR
import pathlib

COLOURS_DICT = {"cs": "#b1d1fc", "drama": "#ffb07c", "geo": "#90e4c1"}
SUBJECT_PALETTE = [COLOURS_DICT["cs"], COLOURS_DICT["drama"], COLOURS_DICT["geo"]]
CS_GEO_PALETTE = [COLOURS_DICT["cs"], COLOURS_DICT["geo"]]
SAVE_FIGS_DIR = PROJECT_DIR / "outputs/figures/"

GEO_MGD = "Geography_Mean_Gender_Difference"
DRAMA_MGD = "Drama_Mean_Gender_Difference"
CS_MGD = "CompSci_Mean_Gender_Difference"

GROUPED_SUBJECTS = [
    "Any modern language",
    "Other Modern Languages",
    "English, mathematics & science",
    "Mathematics & science",
    "Combined Science",
    "Any subject",
    "Any design & technology",
    "Any science",
    "English & mathematics",
]


def save_single_histplot(
    data: pd.DataFrame,
    x: str,
    color: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_dir: pathlib.Path,
    save_fn: str,
):
    """Plots and save a hist plot and saves related data

    Args:
        data: Data to be used in the plots
        x: Name of the column of data to plot on the x axis
        color: Colour for the chart
        xlabel: Label to put on the x axis
        ylabel: Label to put on the y axis
        title: Title to give the chart
        save_dir: Path to save the chart to
        save_fn: Filename to be used for the chart and data
    """
    ax = sns.histplot(
        data=school_lvl_dfe,
        x=x,
        kde=True,
        binwidth=0.001,
        binrange=(0.006, 0.018),
        color=color,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 16), title=title)
    plt.savefig(save_dir / f"{save_fn}_img", dpi=300)
    plt.close()
    data.to_csv(save_dir / f"{save_fn}_data.csv")


def save_scatterplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    palette: list,
    xlabel: str,
    ylabel: str,
    save_dir: pathlib.Path,
    save_fn: str,
    move_legend: str = None,
):
    """Plots and saves a scatterplot and related data

    Args:
        data: Data to be used in the plot
        x: Name of the column of data to plot on the x axis
        y: Name of the column of data to plot on the y axis
        hue: Grouping variable that will produce points with
            different colours
        palette: Colours to be used
        xlabel: Label to put on the x axis
        ylabel: Label to put on the y axis
        save_dir: Path to save the chart to
        save_fn: Filename to be used for the chart and data
        move_legend: Position for the legend, for example 'upper left'.
            Defaults to None. If None, will auto locate the legend.
    """
    ax = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
    )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
    )
    if move_legend is not None:
        sns.move_legend(ax, move_legend)
    plt.savefig(save_dir / f"{save_fn}_img", dpi=300)
    plt.close()
    data.to_csv(save_dir / f"{save_fn}_data.csv")


def save_mgd_barplot(
    data: pd.DataFrame,
    palette: list,
    title: str,
    save_dir: pathlib.Path,
    save_fn: str,
    xlabel: str = "Subject",
    ylabel: str = "Mean gender difference",
    x: str = "subject",
    y: str = "mean_gender_diff",
):
    """Plots and saves a mean gender difference barplot

    Args:
        data: Data to be used in the plot
        palette: Colours to be used
        title: Title to give the chart
        save_dir: Path to save the chart to
        save_fn: Filename to be used for the chart and data
        xlabel: Label to put on the x axis.
            Defaults to "Subject".
        ylabel: Label to put on the y axis.
            Defaults to "Mean gender difference".
        x: Name of the column of data to plot on the x axis.
            Defaults to "subject".
        y: Name of the column of data to plot on the y axis.
            Defaults to "mean_gender_diff".
    """
    ax = sns.barplot(
        x=x,
        y=y,
        data=data,
        palette=palette,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, ylim=(0, 0.018))
    plt.savefig(save_dir / f"{save_fn}_img", dpi=300)
    plt.close()
    data.to_csv(save_dir / f"{save_fn}_data.csv")


if __name__ == "__main__":

    school_lvl_dfe = school_lvl_bias_with_dfe_data()

    cs_mean_gender_diff = (
        school_lvl_dfe.drop(columns=[GEO_MGD, DRAMA_MGD])
        .assign(Subject="CS")
        .rename(columns={CS_MGD: "Mean_Gender_Difference"})
    )

    drama_mean_gender_diff = (
        school_lvl_dfe.drop(columns=[GEO_MGD, CS_MGD])
        .assign(Subject="Drama")
        .rename(columns={DRAMA_MGD: "Mean_Gender_Difference"})
    )

    geo_mean_gender_diff = (
        school_lvl_dfe.drop(columns=[DRAMA_MGD, CS_MGD])
        .assign(Subject="Geography")
        .rename(columns={GEO_MGD: "Mean_Gender_Difference"})
    )

    stacked_mean_gender_diff = pd.concat(
        [cs_mean_gender_diff, drama_mean_gender_diff, geo_mean_gender_diff]
    ).reset_index(drop=True)

    make_path_if_not_exist(SAVE_FIGS_DIR)

    ax = sns.histplot(
        data=stacked_mean_gender_diff,
        x="Mean_Gender_Difference",
        hue="Subject",
        palette=SUBJECT_PALETTE,
        binwidth=0.001,
        binrange=(0.006, 0.018),
        element="step",
        multiple="layer",
        kde=True,
    )

    ax.set(
        xlabel="Mean gender difference",
        ylabel="Number of schools",
        title="Distribution of mean gender difference for each subject",
        ylim=(0, 16),
    )
    sns.move_legend(ax, "upper left")
    plt.savefig(SAVE_FIGS_DIR / "dist_overlay_img.png", dpi=300)
    plt.close()
    stacked_mean_gender_diff[["Mean_Gender_Difference", "Subject"]].to_csv(
        SAVE_FIGS_DIR / "dist_overlay_data.csv"
    )

    save_single_histplot(
        data=school_lvl_dfe[["CompSci_Mean_Gender_Difference"]],
        x="CompSci_Mean_Gender_Difference",
        color=COLOURS_DICT["cs"],
        xlabel="CS mean gender difference",
        ylabel="Number of schools",
        title="Distribution of CS mean gender difference",
        save_dir=SAVE_FIGS_DIR,
        save_fn="dist_cs_mean_gender_diff",
    )

    save_single_histplot(
        data=school_lvl_dfe[["Drama_Mean_Gender_Difference"]],
        x="Drama_Mean_Gender_Difference",
        color=COLOURS_DICT["drama"],
        xlabel="Drama mean gender difference",
        ylabel="Number of schools",
        title="Distribution of Drama mean gender difference",
        save_dir=SAVE_FIGS_DIR,
        save_fn="dist_drama_mean_gender_diff",
    )

    save_single_histplot(
        data=school_lvl_dfe[["Geography_Mean_Gender_Difference"]],
        x="Geography_Mean_Gender_Difference",
        color=COLOURS_DICT["geo"],
        xlabel="Geography mean gender difference",
        ylabel="Number of schools",
        title="Distribution of Geography mean gender difference",
        save_dir=SAVE_FIGS_DIR,
        save_fn="dist_geo_mean_gender_diff",
    )

    save_scatterplot(
        data=stacked_mean_gender_diff[
            ["Mean_Gender_Difference", "percentage_of_boys_on_roll", "Subject"]
        ],
        x="Mean_Gender_Difference",
        y="percentage_of_boys_on_roll",
        hue="Subject",
        palette=SUBJECT_PALETTE,
        xlabel="Mean gender difference",
        ylabel="Percentage of boys on roll",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gender_diff_vs_boys_on_roll",
    )

    save_scatterplot(
        data=stacked_mean_gender_diff[
            ["Mean_Gender_Difference", "percentage_pupils_fsm_past_6_years", "Subject"]
        ],
        x="Mean_Gender_Difference",
        y="percentage_pupils_fsm_past_6_years",
        hue="Subject",
        palette=SUBJECT_PALETTE,
        xlabel="Mean gender difference",
        ylabel="Percentage of pupils with free school meals",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gender_diff_vs_fsm",
        move_legend="upper left",
    )

    save_scatterplot(
        data=stacked_mean_gender_diff[
            [
                "Mean_Gender_Difference",
                "average_girls_attainment_8_gcse_score",
                "Subject",
            ]
        ],
        x="Mean_Gender_Difference",
        y="average_girls_attainment_8_gcse_score",
        hue="Subject",
        palette=SUBJECT_PALETTE,
        xlabel="Mean gender difference",
        ylabel="Mean girls attainment 8 gcse score",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gender_diff_vs_girls_attainment8",
    )

    save_scatterplot(
        data=stacked_mean_gender_diff[
            [
                "Mean_Gender_Difference",
                "average_boys_attainment_8_gcse_score",
                "Subject",
            ]
        ],
        x="Mean_Gender_Difference",
        y="average_boys_attainment_8_gcse_score",
        hue="Subject",
        palette=SUBJECT_PALETTE,
        xlabel="Mean gender difference",
        ylabel="Mean boys attainment 8 gcse score",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gender_diff_vs_boys_attainment8",
    )

    girls_entry = (
        girls_entry_percentage()
        .query("total_entry > 10000")
        .query(f"subject not in {GROUPED_SUBJECTS}")
    )

    girls_entry_bar_colors = [
        COLOURS_DICT["cs"]
        if (sub == "Computer Science")
        else COLOURS_DICT["geo"]
        if (sub == "Geography")
        else COLOURS_DICT["drama"]
        if (sub == "Drama")
        else "#d8dcd6"
        for sub in girls_entry.subject.values
    ]

    ax = sns.barplot(
        x="girls_entry_percent",
        y="subject",
        data=girls_entry,
        palette=girls_entry_bar_colors,
    )

    ax.set(
        xlabel="Girls entry percentage",
        ylabel="Subject",
    )
    plt.tight_layout()
    plt.savefig(SAVE_FIGS_DIR / "girls_entry_percentage_img.png", dpi=300)
    plt.close()
    girls_entry.to_csv(SAVE_FIGS_DIR / "girls_entry_percentage_data.csv")

    mgd_bit = mean_gender_differences("bit")

    save_mgd_barplot(
        data=mgd_bit.query("POS == 'Noun'"),
        palette=CS_GEO_PALETTE,
        title="Nouns",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gend_diff_noun_bit",
    )

    save_mgd_barplot(
        data=mgd_bit.query("POS == 'Verb'"),
        palette=CS_GEO_PALETTE,
        title="Verbs",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gend_diff_verb_bit",
    )

    save_mgd_barplot(
        data=mgd_bit.query("POS == 'Adj/Adv'"),
        palette=CS_GEO_PALETTE,
        title="Adj/Adv",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gend_diff_adjadv_bit",
    )

    mgd_scraped = mean_gender_differences("scraped")

    save_mgd_barplot(
        data=mgd_scraped.query("POS == 'Noun'"),
        palette=SUBJECT_PALETTE,
        title="Nouns",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gend_diff_noun_scraped",
    )

    save_mgd_barplot(
        data=mgd_scraped.query("POS == 'Verb'"),
        palette=SUBJECT_PALETTE,
        title="Verbs",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gend_diff_verb_scraped",
    )

    save_mgd_barplot(
        data=mgd_scraped.query("POS == 'Adj/Adv'"),
        palette=SUBJECT_PALETTE,
        title="Adj/Adv",
        save_dir=SAVE_FIGS_DIR,
        save_fn="mean_gend_diff_adjadv_scraped",
    )
