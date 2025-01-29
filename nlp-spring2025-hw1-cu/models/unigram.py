"""
Unigram Language Model Implementation
Created: 2025-01-27
Updated: 2025-01-27
Author: Rama Chaganti

This module implements a basic unigram language model using various text corpora
from NLTK. It provides functionality to calculate unigram probabilities for words
using different corpora (Brown, Reuters, or Webtext).
"""

from nltk.corpus import brown, webtext, reuters

def get_corpus(corpus_name='brown'):
    """
    Returns the specified corpus.
    Args:
        corpus_name (str): Name of corpus ('brown', 'webtext', or 'reuters')
    Returns:
        list: Words from the selected corpus
    """
    corpus_dict = {
        'brown': brown.words(),
        'webtext': webtext.words(),
        'reuters': reuters.words()
    }
    return corpus_dict.get(corpus_name, brown.words())

def unigram(word, corpus_name='brown'):
    corpus = get_corpus(corpus_name)
    prob = corpus.count(word) / len(corpus)
    return prob



