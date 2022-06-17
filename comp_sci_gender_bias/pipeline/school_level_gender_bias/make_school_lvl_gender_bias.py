from comp_sci_gender_bias.pipeline.glove_differences.make_differences import (
    GLOVE_DIMENSIONS,
)
from comp_sci_gender_bias.pipeline.glove_differences.process_text_utils import (
    TokenTagger,
    TextCleaner,
    GloveDistances,
)
from comp_sci_gender_bias.getters.scraped_data import scraped_data
from comp_sci_gender_bias.getters.dfe_combined_school_data import (
    dfe_combined_school_data,
)
from comp_sci_gender_bias.getters.urn_website_lookup import urn_website_lookup
from comp_sci_gender_bias.utils.process_pandas import (
    clean_website_col,
    percent_to_float,
)
from comp_sci_gender_bias.utils.io import make_path_if_not_exist
from comp_sci_gender_bias import PROJECT_DIR
from statistics import mean

GLOVE_DIMENSIONS = 300

PERCENTAGE_COLS = [
    "percentage_of_girls_on_roll",
    "percentage_of_boys_on_roll",
    "percentage_pupils_fsm_past_6_years",
    "percentage_boys_strong_9to5_passes_eng_math_gcses",
    "percentage_girls_strong_9to5_passes_eng_math_gcses",
]

MEAN_GENDER_SIM_COLS = [
    "CompSci_Mean_Gender_Difference",
    "Geography_Mean_Gender_Difference",
    "Drama_Mean_Gender_Difference",
]

SUBJECT_COLS = ["CompSci", "Geography", "Drama"]

SCHOOL_LVL_SAVE_DIR = PROJECT_DIR / "outputs/school_level"


def mean_gender_cosine_difference(text: str, lemma: bool = False) -> float:
    """Calculate the mean gender cosine difference of all the words
    in the input text.

    Args:
        text: School GCSE subject text
        lemma: If True will use the lemmas of the words in the text,
            if False will use the words in the text

    Returns:
        Mean gender cosine difference value
    """
    clean_text = text_cleaner.clean(text)
    tags = token_tagger.tag(clean_text)
    word_or_lemma_index = 1 if lemma else 0
    words_list = [tag[word_or_lemma_index].lower() for tag in tags]
    word_male_minus_fem_distances = glove_dists.gender_similarity_difference_word_list(
        words_list
    )
    return mean(word_male_minus_fem_distances.values())


if __name__ == "__main__":
    glove_dists = GloveDistances(glove_d=GLOVE_DIMENSIONS)
    glove_dists.load_glove2word2vec()
    text_cleaner = TextCleaner()
    token_tagger = TokenTagger()

    dfe_data = dfe_combined_school_data()
    urn_web = urn_website_lookup().pipe(clean_website_col, "SchoolWebsite")
    scraped_school_descs = scraped_data().pipe(clean_website_col, "Website")
    school_urn = scraped_school_descs.merge(
        right=urn_web, how="left", left_on="Website", right_on="SchoolWebsite"
    ).drop(columns="SchoolWebsite")

    for mean_gender_sim_col, subject_col in zip(MEAN_GENDER_SIM_COLS, SUBJECT_COLS):
        school_urn[mean_gender_sim_col] = school_urn[subject_col].apply(
            mean_gender_cosine_difference
        )

    school_urn_dfe = school_urn.merge(
        right=dfe_data,
        left_on="URN",
        right_on="school_unique_reference_number",
        how="left",
    )

    for col in PERCENTAGE_COLS:
        school_urn_dfe = school_urn_dfe.pipe(percent_to_float, col)

    make_path_if_not_exist(SCHOOL_LVL_SAVE_DIR)
    school_urn_dfe.to_csv(SCHOOL_LVL_SAVE_DIR / "scraped_schools_urn_dfe.csv")
