import spacy_udpipe


class TokenTagger:
    def __init__(self):

        spacy_udpipe.download("en")
        self.nlp = spacy_udpipe.load("en")

    def tag(self, sentence):
        """
        Tokenise a single sentence and output a list of tuples
        (text, lemma, POS) for each token
        """

        doc = self.nlp(sentence)

        return [(token.text, token.lemma_, token.pos_) for token in doc]
