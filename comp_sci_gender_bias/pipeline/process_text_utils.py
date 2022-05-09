import spacy_udpipe


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
