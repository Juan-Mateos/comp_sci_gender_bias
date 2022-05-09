import pytest

from comp_sci_gender_bias.pipeline.process_text_utils import TokenTagger


def test_TokenTagger():

    token_tagger = TokenTagger()

    sentence = "London is in England"
    tags = token_tagger.tag(sentence)
    assert len(tags) == 4
    assert tags[0][2] == "NOUN"

    tags = token_tagger.tag(sentence, convert_propn=False)
    assert tags[0][2] == "PROPN"
