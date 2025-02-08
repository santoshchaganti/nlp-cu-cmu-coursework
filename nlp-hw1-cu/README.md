# <center> Homework 1 CSCI 5832 <center/>
## N-gram Language Models Implementation

This repository contains implementations of Unigram, Bigram, and Trigram language models using NLTK corpora (Brown, Webtext, and Reuters). The models can calculate word probabilities, predict next words, and generate sentences based on the learned patterns from the training data.

## Getting Started
### Clone the Repository
```console
$ git clone https://github.com/santoshchaganti/nlp-cu-cmu-coursework.git
$ cd nlp-hw1-cu
```
## Requirements
- Python 3.12
- NLTK library
- Required NLTK corpora (brown, webtext, reuters)

## Installation
```console
$ pip install nltk
$ python -c "import nltk; nltk.download('brown'); nltk.download('webtext'); nltk.download('reuters')"
```

## Project Structure


    nlp-hw1-cu/
    ├── unigram.py
    ├── bigram.py                    
    ├── trigram.py                     
    ├── evaluation.py                 
    ├── Report.pdf
    └── README.md


## Usage Instructions

### 1. Unigram Model
The unigram model calculates the probability of individual words in a corpus.
```bash
# Run default example
$ python unigram.py

# Calculate probability for a specific word
$ python unigram.py --word market

# Use different corpus (brown, webtext, or reuters)
$ python unigram.py --word market --corpus webtext
```

### 2. Bigram Model
The bigram model provides four main functionalities:
- Get bigrams from a sentence
- Calculate conditional probabilities
- Predict next word
- Generate sentences

```bash
# Get bigrams for a sentence
$ python bigram.py --get-bigrams "The quick brown fox"

# Calculate conditional probability
$ python bigram.py --word1 the --word2 quick

# Predict next word
$ python bigram.py --predict the

# Generate sentence (default limit=10)
$ python bigram.py --sentence "This is"

# Generate sentence with custom length
$ python bigram.py --sentence "This is" --limit 6
```

### 3. Trigram Model
The trigram model extends the bigram model with similar functionalities but uses three-word sequences:

```bash
# Get trigrams for a sentence
$ python trigram.py --get-trigrams "The quick brown fox"

# Calculate conditional probability
$ python trigram.py --word1 the --word2 quick --word3 brown

# Predict next word
$ python trigram.py --predict1 the --predict2 quick

# Generate sentence (default limit=10)
$ python trigram.py --sentence "This is good"

# Generate sentence with custom length
$ python trigram.py --sentence "This is good" --limit 6
```

### 4. Model Evaluation
To evaluate and compare the models:

```bash
python evaluation.py
```
This will:
- Compare Brown bigram and trigram models
- Test models on Reuters data
- Show perplexity scores
- Generate example predictions

## Notes
- Models are trained on the first 5000 sentences of each corpus by default
- All sentences are preprocessed with start and end tokens (```<s> and </s>```)
- Laplace smoothing is applied for probability calculations