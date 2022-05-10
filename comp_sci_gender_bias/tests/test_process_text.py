import pytest

from comp_sci_gender_bias.pipeline.process_text_utils import TokenTagger, TextCleaner


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
