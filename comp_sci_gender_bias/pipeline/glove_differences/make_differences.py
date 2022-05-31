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
import pathlib

GLOVE_DIMENSIONS = 300
SAVE_DIR = PROJECT_DIR / "outputs/differences/no_divide"
POS_QUERIES = ["POS == 'NOUN'", "POS in ['ADJ', 'ADV']", "POS == 'VERB'"]
POS_LABELS = ["noun", "adjadv", "verb"]


def make_top_freq_word_male_fem_diff(
    sub1_word_pos_corpus_df: pd.DataFrame,
    sub2_word_pos_corpus_df: pd.DataFrame,
    glove_dists: GloveDistances,
) -> pd.DataFrame:
    """Make dataframe containing columns for subject1 - subject2 freq, POS,
    Word freq, Word count, Male - Female

    Args:
        sub1_word_pos_corpus_df: Dataframe containing columns for Word, POS, Corpus
            for subject 1
        sub2_word_pos_corpus_df: Dataframe containing columns for Word, POS, Corpus
            for subject 2
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


def make_query_save_differences(
    sub1_descriptions: list,
    sub1_lbl: str,
    sub2_descriptions: list,
    sub2_lbl: str,
    word_or_lemma: str,
    glove_dists: GloveDistances,
    source: str,
    pos_queries: list = POS_QUERIES,
    pos_labels: list = POS_LABELS,
    top_n: int = 20,
    save_dir: pathlib.Path = SAVE_DIR,
):
    """Makes dataframes containing columns for subject1 and subject2 for
    subject freq difference, POS, Word freq, Word count, Male - Female.
    Queries dataframes to select for POS, sorts by largest subject frequency
    difference and saves top n to csv.

    Args:
        sub1_descriptions: Subject 1 course descriptions
        sub1_lbl: Label for subject 1 e.g 'CS'
        sub2_descriptions: Subject 2 course descriptions
        sub2_lbl: Label for subject 2 e.g 'GEO'
        word_or_lemma: Use the word or the lemma
        glove_dists: GloveDistances class object
        source: Label to use in csv filename to indicate
            where the data came from
        queries: Queries to filter data by POS
        pos_labels: Label to use in csv filename to indicate
            which POS the file is for
        top_n: Top n results
        save_dir: Directory to save csv files to
    """
    sub1_word_pos_corpus_df = word_pos_corpus_df(
        subject_descs=sub1_descriptions,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label=sub1_lbl,
        word_or_lemma=word_or_lemma,
    )
    sub2_word_pos_corpus_df = word_pos_corpus_df(
        subject_descs=sub2_descriptions,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label=sub2_lbl,
        word_or_lemma=word_or_lemma,
    )
    sub1_geo_word_diffs_df = make_top_freq_word_male_fem_diff(
        sub1_word_pos_corpus_df, sub2_word_pos_corpus_df, glove_dists
    )
    sub2_geo_word_diffs_df = make_top_freq_word_male_fem_diff(
        sub2_word_pos_corpus_df, sub1_word_pos_corpus_df, glove_dists
    )

    make_path_if_not_exist(save_dir)

    for query, lbl in zip(pos_queries, pos_labels):
        sub1_geo_word_diffs_df.query(query).sort_values(
            f"{sub1_lbl} - {sub2_lbl} freq", ascending=False
        ).head(top_n).to_csv(
            SAVE_DIR / f"{sub1_lbl}_{sub2_lbl}_diff_{lbl}_{source}_data.csv"
        )
        sub2_geo_word_diffs_df.query(query).sort_values(
            f"{sub2_lbl} - {sub1_lbl} freq", ascending=False
        ).head(top_n).to_csv(
            SAVE_DIR / f"{sub2_lbl}_{sub1_lbl}_diff_{lbl}_{source}_data.csv"
        )


if __name__ == "__main__":
    text_cleaner = TextCleaner()
    token_tagger = TokenTagger()
    glove_dists = GloveDistances(glove_d=GLOVE_DIMENSIONS)
    glove_dists.load_glove2word2vec()

    # Save files for BIT data
    compsci_descr = text_descriptions(subject="compsci").values()
    geo_descr = text_descriptions(subject="geo").values()

    make_query_save_differences(
        sub1_descriptions=compsci_descr,
        sub1_lbl="CS",
        sub2_descriptions=geo_descr,
        sub2_lbl="Geo",
        word_or_lemma="word",
        glove_dists=glove_dists,
        source="bit",
    )

    # Save files for scraped data
    scraped = scraped_data()
    compsci_descr_scraped = list(scraped["CompSci"].values)
    drama_descr_scraped = list(scraped["Drama"].values)

    make_query_save_differences(
        sub1_descriptions=compsci_descr_scraped,
        sub1_lbl="CS",
        sub2_descriptions=drama_descr_scraped,
        sub2_lbl="Drama",
        word_or_lemma="word",
        glove_dists=glove_dists,
        source="scraped",
    )
