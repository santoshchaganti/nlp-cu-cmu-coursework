"""
Data Module
Created: 2025-01-27
Updated: 2025-01-27
Author: Rama Chaganti

This module provides functionality to download and load text corpora from NLTK.
"""

import nltk

# Download the corpora
nltk.download("brown")
nltk.download("webtext")
nltk.download("reuters")


# Load the corpora Example no need to run this
from nltk.corpus import brown, webtext, reuters
brown_corpus = brown.sents()
brown_corpus = [" ".join(sentence) for sentence in brown_corpus]
brown_corpus = ["<s> " + sentence + " </s>" for sentence in brown_corpus][:5000]
webtext_corpus = webtext.sents()
webtext_corpus = [" ".join(sentence) for sentence in webtext_corpus]
webtext_corpus = ["<s> " + sentence + " </s>" for sentence in webtext_corpus][:5000]
reuters_corpus = reuters.sents()
reuters_corpus = [" ".join(sentence) for sentence in reuters_corpus]
reuters_corpus = ["<s> " + sentence + " </s>" for sentence in reuters_corpus][:5000]
