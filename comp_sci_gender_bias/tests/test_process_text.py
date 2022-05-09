import pytest

from comp_sci_gender_bias.pipeline.process_text_utils import TokenTagger


def test_TokenTagger():

    token_tagger = TokenTagger()
    tags = token_tagger.tag("This is a test")

    assert len(tags) == 4
