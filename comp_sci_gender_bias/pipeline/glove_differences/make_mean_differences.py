from comp_sci_gender_bias.pipeline.glove_differences.process_text_utils import (
    TextCleaner,
    TokenTagger,
    GloveDistances,
    word_pos_corpus,
)
from comp_sci_gender_bias.getters.school_data import text_descriptions
from comp_sci_gender_bias.getters.scraped_data import scraped_data
import pandas as pd
from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.utils.io import make_path_if_not_exist

MEAN_DIFFERENCES_SAVE_PATH = PROJECT_DIR / "outputs/mean_differences"
GLOVE_DIMENSIONS = 300

POS_QUERIES = ["POS == 'NOUN'", "POS in ['ADJ', 'ADV']", "POS == 'VERB'"]
POS_LABELS = ["Noun", "Adj/Adv", "Verb"]


def calc_mean_gender_diff(
    sub_word_pos_corpus: pd.DataFrame,
    glove_dists: GloveDistances,
    data_source_lbl: str,
    subject_lbl: str,
):
    """Calculate the mean gender difference for each POS
    for all the words in a subject corpus

    Args:
        sub_word_pos_corpus: Dataframe containing each word in corpus
            with associated POS and Corpus label
        glove_dists: GloveDistances class object
        data_source_lbl: Data source label
        subject_lbl: Subject label

    Returns:
        Dataframe with columns for:
            - POS
            - mean_gender_diff
            - subject
            - data_source
    """
    glove_sims = glove_dists.gender_similarity_difference_word_list(
        sub_word_pos_corpus["Word"].tolist()
    )
    sub_word_pos_corpus["Male - Female"] = sub_word_pos_corpus["Word"].map(glove_sims)
    sub_word_pos_corpus = sub_word_pos_corpus.dropna()
    return pd.DataFrame.from_dict(
        {
            "POS": POS_LABELS,
            "mean_gender_diff": [
                sub_word_pos_corpus.query(query)["Male - Female"].mean()
                for query in POS_QUERIES
            ],
            "subject": [subject_lbl] * len(POS_LABELS),
            "data_source": [data_source_lbl] * len(POS_LABELS),
        }
    )


if __name__ == "__main__":
    text_cleaner = TextCleaner()
    token_tagger = TokenTagger()
    glove_dists = GloveDistances(glove_d=GLOVE_DIMENSIONS)
    glove_dists.load_glove2word2vec()

    compsci_descr = text_descriptions(subject="compsci").values()
    geo_descr = text_descriptions(subject="geo").values()

    cs_bit_word_pos_corpus = word_pos_corpus(
        subject_descs=compsci_descr,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="CS",
        lemma=False,
    )
    geo_bit_word_pos_corpus = word_pos_corpus(
        subject_descs=geo_descr,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="Geo",
        lemma=False,
    )

    mgd_bit_cs = calc_mean_gender_diff(cs_bit_word_pos_corpus, glove_dists, "BIT", "CS")
    mgd_bit_geo = calc_mean_gender_diff(
        geo_bit_word_pos_corpus, glove_dists, "BIT", "Geography"
    )

    mgd_bit = pd.concat([mgd_bit_cs, mgd_bit_geo])
    make_path_if_not_exist(MEAN_DIFFERENCES_SAVE_PATH)
    mgd_bit.to_csv(
        MEAN_DIFFERENCES_SAVE_PATH / "mean_differences_pos_bit.csv", index=False
    )

    scraped = scraped_data()
    compsci_descr_scraped = list(scraped["CompSci"].values)
    drama_descr_scraped = list(scraped["Drama"].values)
    geography_descr_scraped = list(scraped["Geography"].values)

    cs_scraped_word_pos_corpus = word_pos_corpus(
        subject_descs=compsci_descr_scraped,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="CS",
        lemma=False,
    )
    drama_scraped_word_pos_corpus = word_pos_corpus(
        subject_descs=drama_descr_scraped,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="Drama",
        lemma=False,
    )
    geo_scraped_word_pos_corpus = word_pos_corpus(
        subject_descs=geography_descr_scraped,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="Geo",
        lemma=False,
    )

    mgd_scraped_cs = calc_mean_gender_diff(
        cs_scraped_word_pos_corpus, glove_dists, "Scraped", "CS"
    )
    mgd_scraped_drama = calc_mean_gender_diff(
        drama_scraped_word_pos_corpus, glove_dists, "Scraped", "Drama"
    )
    mgd_scraped_geo = calc_mean_gender_diff(
        geo_scraped_word_pos_corpus, glove_dists, "Scraped", "Geography"
    )
    mgd_scraped = pd.concat([mgd_scraped_cs, mgd_scraped_drama, mgd_scraped_geo])
    mgd_scraped.to_csv(
        MEAN_DIFFERENCES_SAVE_PATH / "mean_differences_pos_scraped.csv", index=False
    )
