from comp_sci_gender_bias.getters.school_data import text_descriptions
from comp_sci_gender_bias.pipeline.glove_differences.process_text_utils import (
    TokenTagger,
    TextCleaner,
    GloveDistances,
    get_word_comparisons,
    word_pos_corpus_df,
)
import pandas as pd
from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.utils.io import make_path_if_not_exist
from comp_sci_gender_bias.getters.scraped_data import scraped_data

GLOVE_DIMENSIONS = 300
SAVE_DIR = PROJECT_DIR / "outputs/differences"
QUERIES = ["POS == 'NOUN'", "POS in ['ADJ', 'ADV']", "POS == 'VERB'"]
LABELS = ["noun", "adjadv", "verb"]


def make_top_freq_word_male_fem_diff(
    sub1_word_pos_corpus_df: pd.DataFrame,
    sub2_word_pos_corpus_df: pd.DataFrame,
    glove_dists: GloveDistances,
) -> pd.DataFrame:
    """Make dataframe containing columns for subject1 - subject2 freq, POS,
    Word freq, Word count, Male - Female

    Args:
        sub1_word_pos_corpus_df: Dataframe containing columns for Word, POS, Corpus
        sub2_word_pos_corpus_df: Dataframe containing columns for Word, POS, Corpus
        glove_dists: GloveDistances class object

    Returns:
        Dataframe containing columns for subject1 - subject2 freq, POS,
            Word freq, Word count, Male - Female
    """
    word_differences_df = get_word_comparisons(
        sub1_word_pos_corpus_df, sub2_word_pos_corpus_df
    )
    glove_sims = glove_dists.gender_similarity_difference_word_list(
        word_differences_df.index.tolist()
    )
    word_differences_df["Male - Female"] = word_differences_df.index.map(glove_sims)
    return word_differences_df


if __name__ == "__main__":
    text_cleaner = TextCleaner()
    token_tagger = TokenTagger()
    glove_dists = GloveDistances(glove_d=GLOVE_DIMENSIONS)
    glove_dists.load_glove2word2vec()

    compsci_descr = text_descriptions(subject="compsci").values()
    geo_descr = text_descriptions(subject="geo").values()

    comp_df = word_pos_corpus_df(
        subject_descs=compsci_descr,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="CS",
        word_or_lemma="word",
    )
    geo_df = word_pos_corpus_df(
        subject_descs=geo_descr,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="Geo",
        word_or_lemma="word",
    )

    comp_geo_word_diffs_df = make_top_freq_word_male_fem_diff(
        comp_df, geo_df, glove_dists
    )
    geo_comp_word_diffs_df = make_top_freq_word_male_fem_diff(
        geo_df, comp_df, glove_dists
    )
    make_path_if_not_exist(SAVE_DIR)
    for query, lbl in zip(QUERIES, LABELS):
        comp_geo_word_diffs_df.query(query).sort_values(
            "CS - Geo freq", ascending=False
        ).head(20).to_csv(SAVE_DIR / f"cs_geo_diff_{lbl}_bit_data.csv")
        geo_comp_word_diffs_df.query(query).sort_values(
            "Geo - CS freq", ascending=False
        ).head(20).to_csv(SAVE_DIR / f"geo_cs_diff_{lbl}_bit_data.csv")

    scraped = scraped_data()
    compsci_descr_scraped = list(scraped["CompSci"].values)
    drama_descr_scraped = list(scraped["Drama"].values)

    comp_scraped_df = word_pos_corpus_df(
        subject_descs=compsci_descr_scraped,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="CS",
        word_or_lemma="word",
    )
    drama_scraped_df = word_pos_corpus_df(
        subject_descs=drama_descr_scraped,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="Drama",
        word_or_lemma="word",
    )

    comp_drama_word_diffs_df = make_top_freq_word_male_fem_diff(
        comp_scraped_df, drama_scraped_df, glove_dists
    )
    drama_comp_word_diffs_df = make_top_freq_word_male_fem_diff(
        drama_scraped_df, comp_scraped_df, glove_dists
    )

    for query, lbl in zip(QUERIES, LABELS):
        comp_drama_word_diffs_df.query(query).sort_values(
            "CS - Drama freq", ascending=False
        ).head(20).to_csv(SAVE_DIR / f"cs_drama_diff_{lbl}_scraped_data.csv")
        drama_comp_word_diffs_df.query(query).sort_values(
            "Drama - CS freq", ascending=False
        ).head(20).to_csv(SAVE_DIR / f"drama_cs_diff_{lbl}_scraped_data.csv")
