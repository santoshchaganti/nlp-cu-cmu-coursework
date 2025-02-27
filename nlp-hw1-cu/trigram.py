"""
Trigram Language Model Implementation
Created: 2025-01-27
Updated: 2025-02-08
Author: Rama Chaganti

This module implements a trigram language model using Brown corpus.
It provides functionality to find trigrams, calculate conditional probabilities,
and predict next words in sequences.

Usage:
    python trigram.py                                      # Run default examples for all functions
    python trigram.py --get-trigrams "The quick brown fox"  # Get trigrams for sentence
    python trigram.py --word1 the --word2 quick --word3 brown  # Calculate conditional probability
    python trigram.py --predict1 the --predict2 quick          # Predict next word
    python trigram.py --sentence "This is good"               # Predict full sentence (default limit=10)
    python trigram.py --sentence "This is good" --limit 6     # Predict sentence with custom limit
"""

from nltk.corpus import brown
from collections import Counter
import argparse

brown_corpus = brown.sents()
brown_corpus = [" ".join(sentence) for sentence in brown_corpus]
brown_corpus = ["<s> " + sentence + " </s>" for sentence in brown_corpus]

class TrigramModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.trigrams = self.get_trigrams()
        self.trigram_counts = Counter(self.trigrams)
        self.bigram_counts = Counter((t[0], t[1]) for t in self.trigrams)
        self.V, self.unique_words = self.get_V()
        
    def get_trigrams(self):
        list_tuples = []
        for sentence in self.corpus:
            words = sentence.split()
            list_tuples.extend(list(zip(words[:-2], words[1:-1], words[2:])))
        return list_tuples
    
    def get_V(self):
        unique_words = {word for tuple_pair in self.trigrams for word in tuple_pair}
        return len(unique_words), unique_words
    
    def conditional_prob_tg(self, word1, word2, word3):
        trigram_count = self.trigram_counts[(word1, word2, word3)]
        bigram_count = self.bigram_counts[(word1, word2)]
        return (trigram_count + 1) / (bigram_count + self.V + 1)
    
    def predict_next_word(self, word1, word2):
        probabilities = {}
        for word in self.unique_words:
            probabilities[word] = self.conditional_prob_tg(word1, word2, word)
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
            words.append(self.predict_next_word(words[-2], words[-1]))
        
        return f"\033[92m{' '.join(words)}\033[0m"
    
# model = TrigramModel(brown_corpus)
# # print(model.predict_sentence('</s> I am very', limit_total_words = 20))
# print(model.predict_sentence('</s> This is good', limit_total_words = 20))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trigram model operations')
    parser.add_argument('--get-trigrams', type=str,
                      help='sentence to get trigrams from')
    parser.add_argument('--word1', type=str,
                      help='first word for conditional probability')
    parser.add_argument('--word2', type=str,
                      help='second word for conditional probability')
    parser.add_argument('--word3', type=str,
                      help='third word for conditional probability')
    parser.add_argument('--predict1', type=str,
                      help='first word for prediction')
    parser.add_argument('--predict2', type=str,
                      help='second word for prediction')
    parser.add_argument('--sentence', type=str,
                      help='initial sequence for sentence prediction')
    parser.add_argument('--limit', type=int, default=10,
                      help='word limit for sentence prediction')
    
    args = parser.parse_args()
    
    # Initialize model with first 5000 sentences
    model = TrigramModel(brown_corpus[:5000])
    
    print("\nTrigram Model Examples")
    print("-" * 50)
    
    if not any(vars(args).values()):
        
        test_sentence = "The quick brown fox jumps over"
        temp_model = TrigramModel([test_sentence])
        print("\n1. Get Trigrams Example:")
        print(f'In: "{test_sentence}"')
        print(f"Out: {temp_model.trigrams}")
        
        print("\n2. Conditional Probability Example:")
        print('In: conditional_prob_tg("the", "quick", "brown", trigrams)')
        prob = model.conditional_prob_tg("the", "quick", "brown")
        print(f"Out: {prob:.2f}")
        
        print("\n3. Predict Next Word Example:")
        print('In: predict_next_word("the", "quick", trigrams)')
        next_word = model.predict_next_word("the", "quick")
        print(f'Out: "{next_word}"')
        
        initial_seq = "<s> This is good"
        print("\n4. Predict Sentence Example:")
        print(f'In: predict_sentence("{initial_seq}", trigrams, limit=6)')
        predicted = model.predict_sentence(initial_seq, 6)
        print(f'Out: {predicted}')
    
    else:
        if args.get_trigrams:
            temp_model = TrigramModel([args.get_trigrams])
            print(f'In: "{args.get_trigrams}"')
            print(f"Out: {temp_model.trigrams}")
            
        if args.word1 and args.word2 and args.word3:
            prob = model.conditional_prob_tg(args.word1, args.word2, args.word3)
            print(f'In: conditional_prob_tg("{args.word1}", "{args.word2}", "{args.word3}", trigrams)')
            print(f"Out: {prob:.2f}")
            
        if args.predict1 and args.predict2:
            next_word = model.predict_next_word(args.predict1, args.predict2)
            print(f'In: predict_next_word("{args.predict1}", "{args.predict2}", trigrams)')
            print(f'Out: "{next_word}"')
            
        if args.sentence:
            predicted = model.predict_sentence(args.sentence, args.limit)
            print(f'In: predict_sentence("{args.sentence}", trigrams, limit={args.limit})')
            print(f'Out: {predicted}')