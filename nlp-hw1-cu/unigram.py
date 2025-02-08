"""
Unigram Language Model Implementation
Created: 2025-01-27
Updated: 2025-02-08
Author: Rama Chaganti

This module implements a basic unigram language model using various text corpora
from NLTK. It provides functionality to calculate unigram probabilities for words
using different corpora (Brown, Reuters, or Webtext).

Usage:
    python unigram.py                     # Run with default example
    python unigram.py --word this         # Calculate probability for specific word
    python unigram.py --word this --corpus webtext  # Use different corpus
"""

from nltk.corpus import brown, webtext, reuters
import argparse

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
    """
    Calculate unigram probability of a word in the specified corpus.
    Args:
        word (str): Word to calculate probability for
        corpus_name (str): Name of corpus to use
    Returns:
        float: Probability of the word
    """
    corpus = get_corpus(corpus_name)
    return corpus.count(word) / len(corpus)

# print(unigram("very", "brown"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate unigram probability of a word')
    parser.add_argument('--word', type=str, default='this',
                      help='word to calculate probability for (default: "this")')
    parser.add_argument('--corpus', type=str, default='brown',
                      choices=['brown', 'webtext', 'reuters'],
                      help='corpus to use (default: brown)')
    
    args = parser.parse_args()
    
    prob = unigram(args.word, args.corpus)
    
    print("\nUnigram Probability Example")
    print("-" * 50)
    print(f'In: prob("{args.word}", {args.corpus})')
    print(f"Out: {prob}")