"""
Bigram Language Model Implementation
Created: 2025-01-27
Updated: 2025-02-08
Author: Rama Chaganti

This module implements a basic bigram language model using various text corpora
from NLTK. It provides functionality to calculate bigram probabilities for words
using different corpora (Brown, Reuters, or Webtext).

Usage:
    python bigram.py                           # Run default examples for all functions
    python bigram.py --get-bigrams "The quick brown fox"  # Get bigrams for sentence
    python bigram.py --word1 the --word2 quick # Calculate conditional probability
    python bigram.py --predict the             # Predict next word
    python bigram.py --sentence "This is"      # Predict full sentence (default limit=10)
    python bigram.py --sentence "This is" --limit 6  # Predict sentence with custom limit
"""
from nltk.corpus import brown
from collections import Counter
import argparse

# test_sentence = "The quick brown fox jumps over the lazy dog"
# test_sentence = "The smart student reads the interesting book because the student loves readin"
brown_corpus = brown.sents()
brown_corpus = [" ".join(sentence) for sentence in brown_corpus]
brown_corpus = ["<s> " + sentence + " </s>" for sentence in brown_corpus]

class BigramModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bigrams = self.get_bigrams()
        self.bigram_counts = Counter(self.bigrams)
        self.word1_counts = Counter(word for word, _ in self.bigrams)
        self.V, self.unique_words = self.get_V()
        
    def get_bigrams(self):
        list_tuples = []
        for sentence in self.corpus:
            words = sentence.split()
            list_tuples.extend(list(zip(words[:-1], words[1:])))
        return list_tuples
    
    # Todo 
    # case sensitive case get_V

    def get_V(self):
        unique_words = {word for tuple_pair in self.bigrams for word in tuple_pair}
        return len(unique_words), unique_words
    
    def conditional_prob_bg(self, word1, word2):
        bigram_count = self.bigram_counts[(word1, word2)]
        word1_count = self.word1_counts[word1]
        return (bigram_count + 1) / (word1_count + self.V + 1)
    
    def predict_next_word(self, word1):
        probabilities = {}
        for word in self.unique_words:
            probabilities[word] = self.conditional_prob_bg(word1, word)
        return max(probabilities, key=probabilities.get)
    
    def predict_sentence(self, sentence, limit_total_words = 10):
        words = sentence.split()
        if len(words) >= limit_total_words:
            warning_msg = (f"Warning: Input sentence has {len(words)} words, "
                         f"which exceeds or meets the limit of {limit_total_words}. "
                         f"Truncating to first {limit_total_words} words.")
            print(f"\033[93m{warning_msg}\033[0m")
            return ' '.join(words[:limit_total_words])
        
        for _ in range(limit_total_words - len(words)):
            words.append(self.predict_next_word(words[-1]))
        
        return f"\033[92m{' '.join(words)}\033[0m"
            

# model = BigramModel(brown_corpus)
# print(model.predict_sentence('</s> This is very', limit_total_words = 10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bigram model operations')
    parser.add_argument('--get-bigrams', type=str,
                      help='sentence to get bigrams from')
    parser.add_argument('--word1', type=str,
                      help='first word for conditional probability')
    parser.add_argument('--word2', type=str,
                      help='second word for conditional probability')
    parser.add_argument('--predict', type=str,
                      help='word to predict next word for')
    parser.add_argument('--sentence', type=str,
                      help='initial sequence for sentence prediction')
    parser.add_argument('--limit', type=int, default=10,
                      help='word limit for sentence prediction')
    
    args = parser.parse_args()
    
    # Initialize model with first 5000 sentences
    model = BigramModel(brown_corpus[:5000])
    
    print("\nBigram Model")
    print("-" * 50)
    
    if not any(vars(args).values()):
        test_sentence = "The quick brown fox jumps over the lazy dog"
        temp_model = BigramModel()  # Create temporary model for this sentence
        print("\n1. Get Bigrams Example:")
        print(f'In: "{test_sentence}"')
        print(f"Out: {temp_model.bigrams}")
        
        print("\n2. Conditional Probability Example:")
        print('In: conditional_prob_bg("one", "this", bigrams)')
        prob = model.conditional_prob_bg("one", "this")  # Use main model
        print(f"Out: {prob:.2f}")
        
        print("\n3. Predict Next Word Example:")
        print('In: predict_next_word("this", bigrams)')
        next_word = model.predict_next_word("this")  # Use main model
        print(f'Out: "{next_word}"')
        
        initial_seq = "<s> This is"
        print("\n4. Predict Sentence Example:")
        print(f'In: predict_sentence("{initial_seq}", bigrams, limit=6)')
        predicted = model.predict_sentence(initial_seq, 6)  # Use main model with limit 6
        print(f'Out: {predicted}')
    
    else:
        if args.get_bigrams:
            temp_model = BigramModel([args.get_bigrams])
            print(f'In: "{args.get_bigrams}"')
            print(f"Out: {temp_model.bigrams}")
            
        if args.word1 and args.word2:
            prob = model.conditional_prob_bg(args.word1, args.word2)
            print(f'In: conditional_prob_bg("{args.word1}", "{args.word2}", bigrams)')
            print(f"Out: {prob}")
            
        if args.predict:
            next_word = model.predict_next_word(args.predict)
            print(f'In: predict_next_word("{args.predict}", bigrams)')
            print(f'Out: "{next_word}"')
            
        if args.sentence:
            predicted = model.predict_sentence(args.sentence, args.limit)
            print(f'In: predict_sentence("{args.sentence}", bigrams, limit={args.limit})')
            print(f'Out: {predicted}')
