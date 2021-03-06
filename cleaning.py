"""
Module provides functions for cleaning up text. Separate functions are
written for if stop words or porter stemming is required so no args are
need to be passed and each may be called by pd.Dataframe.apply()
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

english_words = set(nltk.corpus.words.words() + ['australia', 'australian'])
stop_words = set(stopwords.words("english"))


def process_text(text, stop=False, stem=False):
    """Removes punctuation and uppercase letters. Returns list of words"""
    cleaned_text = re.sub('(\n|[^a-zA-Z])', ' ', text)
    words = cleaned_text.lower()
    words = word_tokenize(words)
    if stop:
        words = [w for w in words if not w in stop_words]
    if stem:
        ps = PorterStemmer()
        words = [ps.stem(w) for w in words]
    return words


def stop_process_text(text, stop=True, stem=False):
    """
    Removes punctuation, uppercase letters and stop words.
    Returns list of words
    """
    cleaned_text = re.sub('(\n|[^a-zA-Z])', ' ', text)
    words = cleaned_text.lower()
    words = word_tokenize(words)
    if stop:
        words = [w for w in words if not w in stop_words]
    if stem:
        ps = PorterStemmer()
        words = [ps.stem(w) for w in words]
    return words


def stem_process_text(text, stop=True, stem=True):
    """
    Removes punctuation, uppercase letters and stop words.
    Performs porter stemming on remaining words.
    Returns list of words
    """
    cleaned_text = re.sub('(\n|[^a-zA-Z])', ' ', text)
    words = cleaned_text.lower()
    words = word_tokenize(words)
    words = [w for w in words if w in english_words] 
    if stop:
        words = [w for w in words if not w in stop_words]
    if stem:
        ps = PorterStemmer()
        words = [ps.stem(w) for w in words]
    return " ".join(words)

def process_text_doc2vec(text, stop=False, stem=False):
    """Removes punctuation and uppercase letters. Returns list of words"""
    cleaned_text = re.sub('(\n|[^a-zA-Z])', ' ', text)
    words = cleaned_text.lower()
    words = word_tokenize(words)
    words = [w for w in words if w in english_words]

    return words
