import pandas as pd
from comp_sci_gender_bias.pipeline.glove_differences.process_text_utils import (
    TokenTagger,
    TextCleaner,
    GloveDistances,
    get_word_freq,
    get_word_comparisons,
    combined_pos_freq_and_count,
    subject_from_df,
    word_differences,
    word_pos_corpus_df,
)

geo_word_pos = pd.DataFrame(
    {
        "Word": ["world", "world", "computer"],
        "POS": ["Noun", "Noun", "Noun"],
        "Corpus": ["Geo", "Geo", "Geo"],
    }
)
cs_word_pos = pd.DataFrame(
    {
        "Word": ["computer", "computer", "world"],
        "POS": ["Noun", "Noun", "Noun"],
        "Corpus": ["CS", "CS", "CS"],
    }
)
token_tagger = TokenTagger()
text_cleaner = TextCleaner()


def test_TokenTagger():
    sentence = "London is in England"
    tags = token_tagger.tag(sentence)
    assert len(tags) == 4
    assert tags[0][2] == "NOUN"

    tags = token_tagger.tag(sentence, convert_propn=False)
    assert tags[0][2] == "PROPN"


def test_CleanText():
    cleaned_text = text_cleaner.clean("clean_me!!1")
    assert cleaned_text == "clean me 1"

    # Check "acronyms" aren't being corrected
    cleaned_text = text_cleaner.clean("change thsi but not THSI")
    assert cleaned_text == "change this but not THSI"

    # Check short words aren't corrected
    text = "a cta and dgo"
    assert text_cleaner.clean(text) == text


def test_GloveDistances():

    glove_dists = GloveDistances()
    glove_dists.load_glove2word2vec()

    mother_score = glove_dists.gender_similarity_difference_word_list(["mother"])
    father_score = glove_dists.gender_similarity_difference_word_list(["father"])

    # At the very least 'mother' should be less masculine than 'father'
    assert mother_score["mother"] < father_score["father"]

    not_word_scores = glove_dists.gender_similarity_difference_word_list("notawordkd")

    assert not_word_scores is None

    some_scores = glove_dists.gender_similarity_difference_word_list(
        ["mother", "notawordkd"]
    )
    assert len(some_scores) == 1
    assert "notawordkd" not in some_scores


def test_get_word_freq():

    # frequency of a word / frequency of the specific POS type
    word_pos_df = pd.DataFrame(
        {"Word": ["and", "and", "and", "the"], "POS": ["NOUN", "ADJ", "NOUN", "NOUN"]}
    )
    word_freq = get_word_freq(word_pos_df)
    assert word_freq["and"] == 1


def test_get_word_comparisons():
    word_differences_df = get_word_comparisons(geo_word_pos, cs_word_pos)

    assert len(word_differences_df) == 2
    assert (
        word_differences_df.loc["computer"]["Geo - CS freq"]
        < word_differences_df.loc["world"]["Geo - CS freq"]
    )


def test_combined_pos_freq_and_count():
    (
        all_word_pos,
        all_word_booklet_freq,
        all_word_booklet_count,
    ) = combined_pos_freq_and_count(geo_word_pos, cs_word_pos)

    assert all_word_pos == {"computer": "Noun", "world": "Noun"}
    assert all_word_booklet_freq == {"world": 0.5, "computer": 0.5}
    assert all_word_booklet_count == {"computer": 3, "world": 3}


def test_subject_from_df():
    assert subject_from_df(geo_word_pos) == "Geo"


def test_word_differences():
    sub1_word_freq = get_word_freq(geo_word_pos)
    sub2_word_freq = get_word_freq(cs_word_pos)
    (
        all_word_pos,
        all_word_booklet_freq,
        all_word_booklet_count,
    ) = combined_pos_freq_and_count(geo_word_pos, cs_word_pos)

    assert word_differences(
        sub1_word_freq,
        sub2_word_freq,
        all_word_pos,
        all_word_booklet_freq,
        all_word_booklet_count,
    ) == {
        "computer": (-0.3333333333333333, "Noun", 0.5, 3),
        "world": (0.3333333333333333, "Noun", 0.5, 3),
    }


def test_word_pos_corpus_df():
    cs_descriptions = ["Computer science is good", "Computer science uses computers"]
    cs_word_pos_corpus_df = word_pos_corpus_df(
        subject_descs=cs_descriptions,
        text_cleaner=text_cleaner,
        token_tagger=token_tagger,
        subject_label="CS",
        word_or_lemma="word",
    )
    cs_word_pos_corpus_df_check = pd.DataFrame(
        {
            "Word": [
                "computer",
                "science",
                "is",
                "good",
                "computer",
                "science",
                "uses",
                "computers",
            ],
            "POS": ["NOUN", "NOUN", "AUX", "ADJ", "NOUN", "NOUN", "VERB", "NOUN"],
            "Corpus": ["CS"] * 8,
        }
    )
    assert cs_word_pos_corpus_df.equals(cs_word_pos_corpus_df_check)
