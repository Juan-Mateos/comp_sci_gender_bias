import pandas as pd

import pytest

from comp_sci_gender_bias.pipeline.process_text_utils import (
    TokenTagger,
    TextCleaner,
    GloveDistances,
    get_word_freq,
    get_word_comparisons,
)


def test_TokenTagger():

    token_tagger = TokenTagger()

    sentence = "London is in England"
    tags = token_tagger.tag(sentence)
    assert len(tags) == 4
    assert tags[0][2] == "NOUN"

    tags = token_tagger.tag(sentence, convert_propn=False)
    assert tags[0][2] == "PROPN"


def test_CleanText():

    text_cleaner = TextCleaner()

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

    assert not_word_scores == None

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
    comp_df = pd.DataFrame({"Word": ["computer", "work"], "POS": ["NOUN", "NOUN"]})
    geo_df = pd.DataFrame({"Word": ["geography", "work"], "POS": ["NOUN", "NOUN"]})

    word_differences_df = get_word_comparisons(comp_df, geo_df)

    assert len(word_differences_df) == 3
    assert (
        word_differences_df.loc["computer"]["Geo - CS freq"]
        < word_differences_df.loc["geography"]["Geo - CS freq"]
    )
