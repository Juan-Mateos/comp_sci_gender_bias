import spacy_udpipe
from hunspell import Hunspell
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from typing import Tuple
import os
import re
from dotenv import load_dotenv

load_dotenv()


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


def get_word_freq(word_pos_df: pd.DataFrame, divide_by_pos_freq: bool = False) -> dict:
    """
    Get the word frequencies for a corpus.

    Args:
        word_pos_df: DataFrame with 2 columns ["Word", "POS"].
            Each row is every word in the corpus along with it's POS tag
        divide_by_pos_freq: Whether to divide by frequency of the specific
            part of speech type in all corpus.

    Returns:
        A dictionary of the frequency of words in the corpus.
        If divide_by_pos_freq is True, frequency is divided by
        frequency of the specific part of speech type in all corpus.
    """

    pos_freq = (word_pos_df.groupby("POS").count() / len(word_pos_df))["Word"].to_dict()
    word_pos = (
        word_pos_df.groupby("Word")["POS"].agg(lambda x: pd.Series.mode(x)[0])
    ).to_dict()
    word_corpus_freq = (word_pos_df.groupby("Word").count() / len(word_pos_df))[
        "POS"
    ].to_dict()
    word_corpus_freq_div_by_pos_freq = {
        word: freq / pos_freq[word_pos[word]] for word, freq in word_corpus_freq.items()
    }
    return word_corpus_freq_div_by_pos_freq if divide_by_pos_freq else word_corpus_freq


def combined_pos_freq_and_count(
    sub1_word_pos: pd.DataFrame, sub2_word_pos: pd.DataFrame
) -> Tuple[dict, dict, dict]:
    """For each word, calculate POS label, frequency and counts
    across two corpuses combined

    Args:
        sub1_word_pos: Dataframe containing columns
            for 'Word', 'POS', 'Corpus'
        sub2_word_pos: Dataframe containing columns
            for 'Word', 'POS', 'Corpus'

    Returns:
        A tuple of dictionaries:
            - word: most common POS label across both subjects
            - word: frequency across both subjects
            - word: count across both subjects
    """

    pos_all_words = pd.concat([sub1_word_pos, sub2_word_pos]).reset_index(drop=True)
    all_word_pos = (
        pos_all_words.groupby("Word")["POS"].agg(lambda x: pd.Series.mode(x)[0])
    ).to_dict()
    word_counts = pos_all_words.groupby("Word").count()
    all_word_booklet_freq = (word_counts / len(pos_all_words))["POS"].to_dict()
    all_word_booklet_count = word_counts["POS"].to_dict()
    return all_word_pos, all_word_booklet_freq, all_word_booklet_count


def subject_from_df(sub_word_pos: pd.DataFrame) -> str:
    """Return the subject relating to the course descriptions"""
    return sub_word_pos.Corpus.values[0]


def word_differences(
    sub1_word_freq: dict,
    sub2_word_freq: dict,
    all_word_pos: dict,
    all_word_booklet_freq: dict,
    all_word_booklet_count: dict,
) -> dict:
    """Create dictionary containing frequency difference between the two
    subjects, POS label, word frequency across the combined subjects, word
    count across the combined subjects

    Args:
        sub1_word_freq: Frequency of words in subject1
        sub2_word_freq: Frequency of words in subject2
        all_word_pos: Most common POS label across both subjects
            for each word
        all_word_booklet_freq: Frequency of words across
            both subjects
        all_word_booklet_count: Count of words across
            both subjects

    Returns:
        Dictionary in the format
            word: (sub1 - sub2 frequency, POS, Word freq, Word count)
    """
    return {
        word: (
            sub1_word_freq.get(word, 0) - sub2_word_freq.get(word, 0),
            all_word_pos[word],
            all_word_booklet_freq[word],
            all_word_booklet_count[word],
        )
        for word in set(sub1_word_freq.keys()).union(set(sub2_word_freq.keys()))
    }


def get_word_comparisons(
    sub1_word_pos: pd.DataFrame, sub2_word_pos: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the word frequency differences between
    two subjects. If multiple POS tags are given for a
    particular word then the most commonly occuring one is used
    (e.g. 'students' is NOUN 585 times and a VERB 3 times)

    Args:
        sub1_word_pos: Dataframe containing columns
            for 'Word', 'POS', 'Corpus'
        sub2_word_pos: Dataframe containing columns
            for 'Word', 'POS', 'Corpus'

    Returns:
        DataFrame containing columns for:
            - Word (index)
            - subject1 - subject2 frequency difference
            - POS label
            - Word frequency (in both corpuses combined)
            - Word count
    """
    # Get frequency information for the two corpuses separately
    sub1_word_freq = get_word_freq(sub1_word_pos)
    sub2_word_freq = get_word_freq(sub2_word_pos)

    # Get POS, frequency and count information for the two corpuses combined
    (
        all_word_pos,
        all_word_booklet_freq,
        all_word_booklet_count,
    ) = combined_pos_freq_and_count(sub1_word_pos, sub2_word_pos)

    # Calculate the differences between the subjects
    word_diffs = word_differences(
        sub1_word_freq,
        sub2_word_freq,
        all_word_pos,
        all_word_booklet_freq,
        all_word_booklet_count,
    )

    return pd.DataFrame(
        word_diffs,
        index=[
            f"{subject_from_df(sub1_word_pos)} - {subject_from_df(sub2_word_pos)} freq",
            "POS",
            "Word freq",
            "Word count",
        ],
    ).T


def word_pos_corpus(
    subject_descs: list,
    text_cleaner: TextCleaner,
    token_tagger: TokenTagger,
    subject_label: str,
    word_or_lemma: str,
) -> pd.DataFrame:
    """Turn subject descriptions into a dataframe containing
    columns for Word, POS, Corpus

    Args:
        subject_descs: List of subject descriptions
        text_cleaner: Class to clean text
        token_tagger: Class to part of speech tag text
        subject_label: Subject label that the descriptions are from
            e.g Geo, CS, Drama
        word_or_lemma: Whether to use the text from the descriptions
            or the lemmatized text

    Returns:
        Dataframe containing columns for Word, POS, Corpus
    """
    clean_text = [text_cleaner.clean(text) for text in subject_descs]
    clean_tagged = [token_tagger.tag(text) for text in clean_text]
    clean_tagged_flatten = [
        clean_tag for sublist in clean_tagged for clean_tag in sublist
    ]
    world_or_lemma_index = 0 if word_or_lemma == "word" else 1
    return pd.DataFrame(
        {
            "Word": [
                tags[world_or_lemma_index].lower() for tags in clean_tagged_flatten
            ],
            "POS": [tags[2] for tags in clean_tagged_flatten],
            "Corpus": [subject_label] * len(clean_tagged_flatten),
        }
    )
