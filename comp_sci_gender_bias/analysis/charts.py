# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from comp_sci_gender_bias.getters.dfe_combined_school_data import (
    dfe_combined_school_data,
)
from comp_sci_gender_bias.getters.scraped_data import scraped_data
from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd
from statistics import mean
from comp_sci_gender_bias.pipeline.glove_differences.process_text_utils import (
    TokenTagger,
    TextCleaner,
    GloveDistances,
)
import seaborn as sns
import matplotlib.pyplot as plt

# %%
SAVE_FIGS_DIR = PROJECT_DIR / "outputs/figures/"

# %%
COLOURS_DICT = {"cs": "#b1d1fc", "drama": "#ffb07c", "geo": "#90e4c1"}


# %%
def remove_fw_slash(df, col):
    """Remove '/' from the end of all rows in specified
    df and col"""
    df[col] = df[col].map(lambda x: str(x)[:-1] if str(x)[-1] == "/" else x)
    return df


def remove_http_https(df, col):
    """Remove http:// and https:// from the start of all rows in
    specified df and col"""
    df[col] = df[col].map(lambda x: str(x).replace("https://", ""))
    df[col] = df[col].map(lambda x: str(x).replace("http://", ""))
    return df


def remove_new_para(df, col):
    "Remove \n from specified df and col"
    df[col] = df[col].map(lambda x: str(x).replace("\n", ""))
    return df


def add_www(df, col):
    """Add www. to start of all rows in specified
    df and col"""
    df[col] = df[col].map(lambda x: f"www.{str(x)}" if str(x)[:4] != "www." else x)
    return df


def clean_website_col(df, col):
    """Clean website text to make it easier to join"""
    return (
        df.pipe(remove_fw_slash, col)
        .pipe(remove_http_https, col)
        .pipe(remove_new_para, col)
        .pipe(add_www, col)
    )


def percent_to_float(df, col):
    df[col] = df[col].map(lambda x: float(str(x).strip("%")) / 100)
    return df


def mean_gender_cosine_sim(text: str) -> float:
    clean_text = text_cleaner.clean(text)
    tags = token_tagger.tag(clean_text)
    words_list = [tag[0].lower() for tag in tags]
    word_male_minus_fem_distances = glove_dists.gender_similarity_difference_word_list(
        words_list
    )
    return mean(word_male_minus_fem_distances.values())


def urn_website_lookup():
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/get_information_schools_urn_website.csv"
    )


# %%
"""Functions for testing things out

def n_most_gender_cosine_sim(text:str) -> float:
    clean_text = text_cleaner.clean(text)
    tags = token_tagger.tag(clean_text)
    words_list = [tag[0].lower() for tag in tags]
    return glove_dists.gender_similarity_difference_word_list(words_list)

def mean_gender_cosine_sim(text:str) -> float:
    clean_text = text_cleaner.clean(text)
    tags = token_tagger.tag(clean_text)
    words_list = [tag[0].lower() for tag in tags]
    word_male_minus_fem_distances = glove_dists.gender_similarity_difference_word_list(words_list)
    return word_male_minus_fem_distances.values()
"""

# %%

# %%
n_most_gender_cosine_sim(list(scraped_urn["CompSci"].values)[0])

# %%
list(scraped_urn["CompSci"].values)[0]

# %%
dfe_data = dfe_combined_school_data()
urn_web = urn_website_lookup().pipe(clean_website_col, "SchoolWebsite")
scraped = scraped_data().pipe(clean_website_col, "Website")
scraped_urn = scraped.merge(
    right=urn_web, how="left", left_on="Website", right_on="SchoolWebsite"
).drop(columns="SchoolWebsite")

# %%
glove_dists = GloveDistances(glove_d=300)
glove_dists.load_glove2word2vec()
text_cleaner = TextCleaner()
token_tagger = TokenTagger()

# %%
scraped_urn["CompSci_Mean_Gender_Similarity"] = scraped_urn["CompSci"].apply(
    mean_gender_cosine_sim
)
scraped_urn["Geography_Mean_Gender_Similarity"] = scraped_urn["Geography"].apply(
    mean_gender_cosine_sim
)
scraped_urn["Drama_Mean_Gender_Similarity"] = scraped_urn["Drama"].apply(
    mean_gender_cosine_sim
)

# %%
scraped_dfe = scraped_urn.merge(
    right=dfe_data, left_on="URN", right_on="school_unique_reference_number", how="left"
)

# %%
PERCENTAGE_COLS = [
    "percentage_of_girls_on_roll",
    "percentage_of_boys_on_roll",
    "percentage_pupils_fsm_past_6_years",
    "percentage_boys_strong_9to5_passes_eng_math_gcses",
    "percentage_girls_strong_9to5_passes_eng_math_gcses",
]
for col in PERCENTAGE_COLS:
    scraped_dfe = scraped_dfe.pipe(percent_to_float, col)

# %%
scraped_dfe.drop(
    columns=[
        "URN",
        "school_unique_reference_number",
        "percentage_boys_strong_9to5_passes_eng_math_gcses",
        "percentage_girls_strong_9to5_passes_eng_math_gcses",
    ]
).corr()

# %%
compsci = scraped_dfe.drop(
    columns=["Geography_Mean_Gender_Similarity", "Drama_Mean_Gender_Similarity"]
)
compsci["Subject"] = "CS"
compsci = compsci.rename(
    columns={"CompSci_Mean_Gender_Similarity": "Mean_Gender_Similarity"}
)

drama = scraped_dfe.drop(
    columns=["Geography_Mean_Gender_Similarity", "CompSci_Mean_Gender_Similarity"]
)
drama["Subject"] = "Drama"
drama = drama.rename(columns={"Drama_Mean_Gender_Similarity": "Mean_Gender_Similarity"})

geo = scraped_dfe.drop(
    columns=["Drama_Mean_Gender_Similarity", "CompSci_Mean_Gender_Similarity"]
)
geo["Subject"] = "Geography"
geo = geo.rename(columns={"Geography_Mean_Gender_Similarity": "Mean_Gender_Similarity"})

stacked = pd.concat([compsci, drama, geo]).reset_index(drop=True)

# %%
SUBJECT_PALETTE = [COLOURS_DICT["cs"], COLOURS_DICT["drama"], COLOURS_DICT["geo"]]

# %%
ax = sns.histplot(
    data=stacked,
    x="Mean_Gender_Similarity",
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
plt.savefig(SAVE_FIGS_DIR / "dist_overlay.png", dpi=300)


# %%
def save_single_histplot(data, x, color, xlabel, ylabel, title, save_dir, save_fn):

    ax = sns.histplot(
        data=scraped_dfe,
        x=x,
        kde=True,
        binwidth=0.001,
        binrange=(0.006, 0.018),
        color=color,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 16), title=title)

    plt.savefig(save_dir / save_fn, dpi=300)


# %%
save_single_histplot(
    data=scraped_dfe,
    x="CompSci_Mean_Gender_Similarity",
    color=COLOURS_DICT["cs"],
    xlabel="CS mean gender difference",
    ylabel="Number of schools",
    title="Distribution of CS mean gender difference",
    save_dir=SAVE_FIGS_DIR,
    save_fn="dist_cs_mean_gender_diff",
)

# %%
save_single_histplot(
    data=scraped_dfe,
    x="Drama_Mean_Gender_Similarity",
    color=COLOURS_DICT["drama"],
    xlabel="Drama mean gender difference",
    ylabel="Number of schools",
    title="Distribution of Drama mean gender difference",
    save_dir=SAVE_FIGS_DIR,
    save_fn="dist_drama_mean_gender_diff",
)

# %%
save_single_histplot(
    data=scraped_dfe,
    x="Geography_Mean_Gender_Similarity",
    color=COLOURS_DICT["geo"],
    xlabel="Geography mean gender difference",
    ylabel="Number of schools",
    title="Distribution of Geography mean gender difference",
    save_dir=SAVE_FIGS_DIR,
    save_fn="dist_geo_mean_gender_diff",
)

# %%
ax = sns.scatterplot(
    data=stacked,
    x="Mean_Gender_Similarity",
    y="percentage_of_boys_on_roll",
    hue="Subject",
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Mean gender difference",
    ylabel="Percentage of boys on roll",
)


plt.savefig(SAVE_FIGS_DIR / "gender_sim_vs_boys_on_roll", dpi=300)

# %%
ax = sns.scatterplot(
    data=stacked,
    x="Mean_Gender_Similarity",
    y="percentage_pupils_fsm_past_6_years",
    hue="Subject",
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Mean gender difference",
    ylabel="Percentage of pupils with free school meals",
)

sns.move_legend(ax, "upper left")
plt.savefig(SAVE_FIGS_DIR / "gender_sim_vs_fsm", dpi=300)

# %%
ax = sns.scatterplot(
    data=stacked,
    x="Mean_Gender_Similarity",
    y="average_girls_attainment_8_gcse_score",
    hue="Subject",
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Mean gender difference",
    ylabel="Mean girls attainment 8 gcse score",
)


plt.savefig(SAVE_FIGS_DIR / "gender_sim_vs_girls_attainment8", dpi=300)

# %%
ax = sns.scatterplot(
    data=stacked,
    x="Mean_Gender_Similarity",
    y="average_boys_attainment_8_gcse_score",
    hue="Subject",
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Mean gender difference",
    ylabel="Mean boys attainment 8 gcse score",
)


plt.savefig(SAVE_FIGS_DIR / "gender_sim_vs_boys_attainment8", dpi=300)

# %%
sns.scatterplot(
    data=stacked,
    x="Mean_Gender_Similarity",
    y="average_boys_attainment_8_gcse_score",
    hue="ofsted_rating",
)

# %%
sns.scatterplot(
    data=scraped_dfe,
    x="Geography_Mean_Gender_Similarity",
    y="CompSci_Mean_Gender_Similarity",
)

# %%
sns.scatterplot(
    data=scraped_dfe,
    x="Drama_Mean_Gender_Similarity",
    y="CompSci_Mean_Gender_Similarity",
)

# %%
sns.scatterplot(
    data=scraped_dfe,
    x="Drama_Mean_Gender_Similarity",
    y="Geography_Mean_Gender_Similarity",
)

# %%
"""
https://explore-education-statistics.service.gov.uk/find-statistics/key-stage-4-performance-revised/2020-21
"""

# %%
data = pd.read_csv(PROJECT_DIR / "inputs/data/2021_subject_s1245789_data.csv").query(
    "school_type == 'All state-funded'"
)[
    [
        "school_type",
        "country_name",
        "time_period",
        "characteristic_gender",
        "subject",
        "subject_entry",
    ]
]

# %%
total = (
    data.query("characteristic_gender == 'Total'")
    .rename(columns={"subject_entry": "total_entry"})
    .drop(
        columns=["characteristic_gender", "country_name", "time_period", "school_type"]
    )
)

# %%
girls = (
    data.query("characteristic_gender == 'Girls'")
    .rename(columns={"subject_entry": "girls_entry"})
    .drop(columns="characteristic_gender")
)

# %%
combined = (
    total.merge(right=girls, on="subject", how="left")
    .drop_duplicates()
    .assign(
        girls_entry_percent=lambda x: round(
            x["girls_entry"] / x["total_entry"] * 100, 2
        )
    )
    .sort_values("girls_entry_percent", ascending=False)
    .reset_index(drop=True)[
        [
            "time_period",
            "country_name",
            "school_type",
            "subject",
            "total_entry",
            "girls_entry",
            "girls_entry_percent",
        ]
    ]
)

combined = combined.query("total_entry > 10000")

grouped_subjects = [
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

combined = combined.query("subject not in @grouped_subjects")

# %%
colors = [
    COLOURS_DICT["cs"]
    if (sub == "Computer Science")
    else COLOURS_DICT["geo"]
    if (sub == "Geography")
    else COLOURS_DICT["drama"]
    if (sub == "Drama")
    else "#d8dcd6"
    for sub in combined.subject.values
]

# %%
ax = sns.barplot(x="girls_entry_percent", y="subject", data=combined, palette=colors)

ax.set(
    xlabel="Girls entry percentage",
    ylabel="Subject",
)
plt.tight_layout()
plt.savefig(SAVE_FIGS_DIR / "girls_entry_percentage.png", dpi=300)

# %%
dfe_data

# %%

# %%
from comp_sci_gender_bias.pipeline.glove_differences.process_text_utils import (
    TextCleaner,
    TokenTagger,
    GloveDistances,
    combined_pos_freq_and_count,
    word_pos_corpus_df,
)
from comp_sci_gender_bias.pipeline.glove_differences.make_differences import (
    make_top_freq_word_male_fem_diff,
)
from comp_sci_gender_bias.getters.school_data import text_descriptions
from comp_sci_gender_bias.getters.scraped_data import scraped_data
from statistics import mean

# %%
GLOVE_DIMENSIONS = 300

# %%
text_cleaner = TextCleaner()
token_tagger = TokenTagger()
glove_dists = GloveDistances(glove_d=GLOVE_DIMENSIONS)
glove_dists.load_glove2word2vec()

# %%
compsci_descr = text_descriptions(subject="compsci").values()
geo_descr = text_descriptions(subject="geo").values()

# %%
len(compsci_descr)

# %%
len(geo_descr)

# %%
cs_bit_word_pos_corpus_df = word_pos_corpus_df(
    subject_descs=compsci_descr,
    text_cleaner=text_cleaner,
    token_tagger=token_tagger,
    subject_label="CS",
    word_or_lemma="word",
)
geo_bit_word_pos_corpus_df = word_pos_corpus_df(
    subject_descs=geo_descr,
    text_cleaner=text_cleaner,
    token_tagger=token_tagger,
    subject_label="Geo",
    word_or_lemma="word",
)


# %%
def calc_mean_gender_diff(
    sub_word_pos_corpus_df, glove_dists, data_source_lbl, subject_lbl
):
    glove_sims = glove_dists.gender_similarity_difference_word_list(
        sub_word_pos_corpus_df["Word"].tolist()
    )
    sub_word_pos_corpus_df["Male - Female"] = sub_word_pos_corpus_df["Word"].map(
        glove_sims
    )
    sub_word_pos_corpus_df = sub_word_pos_corpus_df.dropna()
    pos_queries = ["POS == 'NOUN'", "POS in ['ADJ', 'ADV']", "POS == 'VERB'"]
    pos_lbls = ["Noun", "Adj/Adv", "Verb"]
    return pd.DataFrame.from_dict(
        {
            "POS": pos_lbls,
            "mean_gender_diff": [
                sub_word_pos_corpus_df.query(query)["Male - Female"].mean()
                for query in pos_queries
            ],
            "subject": [subject_lbl] * len(pos_lbls),
            "data_source": [data_source_lbl] * len(pos_lbls),
        }
    )


# %%
mdf_bit_cs = calc_mean_gender_diff(cs_bit_word_pos_corpus_df, glove_dists, "bit", "CS")

# %%
mdf_scraped_geo = calc_mean_gender_diff(
    geo_bit_word_pos_corpus_df, glove_dists, "bit", "Geography"
)

# %%
mdf_bit = pd.concat([mdf_bit_cs, mdf_scraped_geo])

# %%
mdf_bit

# %%
CS_GEO_PAL = [x for x in SUBJECT_PALETTE if x != "#ffb07c"]

# %%
ax = sns.barplot(
    x="subject",
    y="mean_gender_diff",
    data=mdf_bit.query("POS == 'Noun'"),
    palette=CS_GEO_PAL,
)

ax.set(
    xlabel="Subject", ylabel="Mean gender difference", title="Nouns", ylim=(0, 0.018)
)
plt.savefig(SAVE_FIGS_DIR / "mean_gend_diff_noun_bit.png", dpi=300)

# %%
ax = sns.barplot(
    x="subject",
    y="mean_gender_diff",
    data=mdf_bit.query("POS == 'Verb'"),
    palette=CS_GEO_PAL,
)

ax.set(
    xlabel="Subject", ylabel="Mean gender difference", title="Verbs", ylim=(0, 0.018)
)
plt.savefig(SAVE_FIGS_DIR / "mean_gend_diff_verb_bit.png", dpi=300)

# %%
ax = sns.barplot(
    x="subject",
    y="mean_gender_diff",
    data=mdf_bit.query("POS == 'Adj/Adv'"),
    palette=CS_GEO_PAL,
)

ax.set(
    xlabel="Subject", ylabel="Mean gender difference", title="Adj/Adv", ylim=(0, 0.018)
)
plt.savefig(SAVE_FIGS_DIR / "mean_gend_diff_adjadv_bit.png", dpi=300)

# %%
scraped = scraped_data()
compsci_descr_scraped = list(scraped["CompSci"].values)
drama_descr_scraped = list(scraped["Drama"].values)
geography_descr_scraped = list(scraped["Geography"].values)

# %%
cs_scraped_word_pos_corpus_df = word_pos_corpus_df(
    subject_descs=compsci_descr_scraped,
    text_cleaner=text_cleaner,
    token_tagger=token_tagger,
    subject_label="CS",
    word_or_lemma="word",
)
drama_scraped_word_pos_corpus_df = word_pos_corpus_df(
    subject_descs=drama_descr_scraped,
    text_cleaner=text_cleaner,
    token_tagger=token_tagger,
    subject_label="Drama",
    word_or_lemma="word",
)
geo_scraped_word_pos_corpus_df = word_pos_corpus_df(
    subject_descs=geography_descr_scraped,
    text_cleaner=text_cleaner,
    token_tagger=token_tagger,
    subject_label="Geo",
    word_or_lemma="word",
)

# %%
mdf_scraped_cs = calc_mean_gender_diff(
    cs_scraped_word_pos_corpus_df, glove_dists, "scraped", "CS"
)

# %%
mdf_scraped_drama = calc_mean_gender_diff(
    drama_scraped_word_pos_corpus_df, glove_dists, "scraped", "Drama"
)

# %%
mdf_scraped_geo = calc_mean_gender_diff(
    geo_scraped_word_pos_corpus_df, glove_dists, "scraped", "Geography"
)

# %%
mdf_scraped = pd.concat([mdf_scraped_cs, mdf_scraped_drama, mdf_scraped_geo])

# %%
mdf_scraped

# %%
ax = sns.barplot(
    x="subject",
    y="mean_gender_diff",
    data=mdf_scraped.query("POS == 'Noun'"),
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Subject", ylabel="Mean gender difference", title="Nouns", ylim=(0, 0.018)
)
plt.savefig(SAVE_FIGS_DIR / "mean_gend_diff_noun_scraped.png", dpi=300)

# %%
ax = sns.barplot(
    x="subject",
    y="mean_gender_diff",
    data=mdf_scraped.query("POS == 'Verb'"),
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Subject", ylabel="Mean gender difference", title="Verbs", ylim=(0, 0.018)
)
plt.savefig(SAVE_FIGS_DIR / "mean_gend_diff_verb_scraped.png", dpi=300)

# %%
ax = sns.barplot(
    x="subject",
    y="mean_gender_diff",
    data=mdf_scraped.query("POS == 'Adj/Adv'"),
    palette=SUBJECT_PALETTE,
)

ax.set(
    xlabel="Subject", ylabel="Mean gender difference", title="Adj/Adv", ylim=(0, 0.018)
)

plt.savefig(SAVE_FIGS_DIR / "mean_gend_diff_adjadv_scraped.png", dpi=300)
