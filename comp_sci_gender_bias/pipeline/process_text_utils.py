import spacy_udpipe
from hunspell import Hunspell

import re


class TextCleaner:
    def __init__(self, replace_char=" "):
        self.hunspell = Hunspell()
        self.replace_char = replace_char

    def strip_nonalphanumeric(self, text):
        """
        Replace all nonalphanumeric characters in a text string
        with replace_char
        """

        return re.sub("[^0-9a-zA-Z]+", self.replace_char, text)

    def spell_check(self, word):
        """
        Spell check a word unless it is an abbreviation or acronym
        and return first suggestion
        hunspell = Hunspell()
        """
        return (
            word
            if (len(word) <= 3) or word.isupper() or self.hunspell.spell(word)
            else self.hunspell.suggest(word)[0]
        )

    def clean(self, text):
        """
        Apply all the cleaning steps to a text string
        """
        text = self.strip_nonalphanumeric(text)

        return " ".join([self.spell_check(word) for word in text.split()])


class TokenTagger:
    def __init__(self):

        spacy_udpipe.download("en")
        self.nlp = spacy_udpipe.load("en")

    def tag(self, sentence, convert_propn=True):
        """
        Tokenise a single sentence and output a list of tuples
        (text, lemma, POS) for each token

        If convert_propn==True then convert all proper nouns to nouns
        """

        doc = self.nlp(sentence)

        tags = [(token.text, token.lemma_, token.pos_) for token in doc]

        if convert_propn:
            return [(t, l, "NOUN") if p == "PROPN" else (t, l, p) for (t, l, p) in tags]
        else:
            return tags
