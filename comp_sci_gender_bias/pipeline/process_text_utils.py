import spacy_udpipe
from hunspell import Hunspell
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd

import os
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
        and return first suggestion (if any given)
        hunspell = Hunspell()
        """
        if (len(word) <= 3) or word.isupper() or self.hunspell.spell(word):
            return word
        else:
            spelling_suggestions = self.hunspell.suggest(word)
            if spelling_suggestions:
                return spelling_suggestions[0]
            else:
                return ""

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


class GloveDistances:
    def __init__(
        self,
        masc_comparisons=["man", "he", "his", "masculine", "male"],
        fem_comparisons=["woman", "she", "her", "feminine", "female"],
        glove_d=100,
    ):

        self.glove_path = os.environ.get("GLOVE_PATH")
        self.masc_comparisons = masc_comparisons
        self.fem_comparisons = fem_comparisons
        self.glove_d = glove_d  # 50, 100, 200, 300
        self.glove_txt_file = f"glove.6B.{self.glove_d}d.txt"

    def load_glove2word2vec(self):
        output_file = os.path.join(self.glove_path, "gensim_" + self.glove_txt_file)
        if not os.path.exists(output_file):
            print("Gensim file being prepared...")
            input_file = os.path.join(self.glove_path, self.glove_txt_file)
            glove2word2vec(input_file, word2vec_output_file=output_file)

        self.model = KeyedVectors.load_word2vec_format(output_file, binary=False)

    def gender_similarity_difference_word_list(self, word_list):
        """
        Input a word or a list of words and the output will be a dictionary
        of the masculine - feminine cosine similarity averages for each word.

        Not all words will be in the corpus, so only words found will be in the output dictionary.
        """
        if not isinstance(word_list, list):
            word_list = [word_list]

        # There will be an error if the word isn't in the vocab
        # also all words need to be lower case to be found in vocab
        word_list = [
            word.lower() for word in word_list if word in self.model.index_to_key
        ]

        if word_list:
            word_vecs = self.model[word_list]

            masc_av_similarities = np.array(
                [
                    self.model.cosine_similarities(self.model[word], word_vecs)
                    for word in self.masc_comparisons
                ]
            ).mean(axis=0)
            fem_av_similarities = np.array(
                [
                    self.model.cosine_similarities(self.model[word], word_vecs)
                    for word in self.fem_comparisons
                ]
            ).mean(axis=0)

            return dict(
                zip(word_list, list(masc_av_similarities - fem_av_similarities))
            )
        else:
            return None


def get_word_freq(word_pos_df):
    """
    Get the word frequencies for a corpus
    Input:
        word_pos_df: DataFrame with 2 columns ["Word", "POS"]
        Each row is every word in the corpus along with it's POS tag
    Output:
        A dictionary of the frequency of words in the corpus
        as computed by frequency of a word in all corpus / frequency of
        the specific part of speech type in all corpus
    """

    pos_freq = (word_pos_df.groupby("POS").count() / len(word_pos_df))["Word"].to_dict()
    word_pos = (
        word_pos_df.groupby("Word")["POS"].agg(lambda x: pd.Series.mode(x)[0])
    ).to_dict()
    word_corpus_freq = (word_pos_df.groupby("Word").count() / len(word_pos_df))[
        "POS"
    ].to_dict()

    return {
        word: freq / pos_freq[word_pos[word]] for word, freq in word_corpus_freq.items()
    }


def get_word_comparisons(comp_df, geo_df):
    """
    Calculate the word frequency differences between Geography course
    data and the CS course data.
    If multiple POS tags are given for a particular word then the most
    commonly occuring one is used
    (e.g. 'students' is NOUN 585 times and a VERB 3 times)

    Input: comp_df and geo_df are the two DataFrames of each
        word and POS tag in the CS and the geography corpuses

    Output: A DataFrame where for each word there is the
        geo-CS word frequencies, as well as the most common POS tag
        for this word, and the word frequency and word count in both
        corpuses combined.
    """

    # Get frequency information for the two corpuses separately
    comp_word_freq = get_word_freq(comp_df)
    geo_word_freq = get_word_freq(geo_df)

    # Get frequency information for the two corpuses combined
    pos_all_words = pd.concat([comp_df, geo_df]).reset_index(drop=True)
    all_word_pos = (
        pos_all_words.groupby("Word")["POS"].agg(lambda x: pd.Series.mode(x)[0])
    ).to_dict()
    word_counts = pos_all_words.groupby("Word").count()
    all_word_booklet_freq = (word_counts / len(pos_all_words))["POS"].to_dict()
    all_word_booklet_count = word_counts["POS"].to_dict()

    # Calculate the differences between CS and Geography
    word_differences = {}
    for word in set(comp_word_freq.keys()).union(set(geo_word_freq.keys())):
        word_differences[word] = (
            geo_word_freq.get(word, 0) - comp_word_freq.get(word, 0),
            all_word_pos[word],
            all_word_booklet_freq[word],
            all_word_booklet_count[word],
        )

    word_differences_df = pd.DataFrame(
        word_differences, index=["Geo - CS freq", "POS", "Word freq", "Word count"]
    ).T

    return word_differences_df
