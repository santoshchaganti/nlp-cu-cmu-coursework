"""
Evaluation Module
Created: 2025-01-27
Updated: 2025-01-27
Author: Rama Chaganti

This module provides functionality to evaluate the performance of bigram and trigram models using perplexity metrics.
"""

def find_next_word_probabilities(sentence, target_word):
    # Convert sentence to lowercase and split into words
    words = sentence.lower().split()
    target_word = target_word.lower()
    
    # Get vocabulary (unique words) and vocabulary size (V)
    vocabulary = set(words)
    V = len(vocabulary)
    
    # Count how many times the target word appears (denominator for basic probability)
    target_count = sum(1 for word in words if word == target_word)
    
    # Store probabilities for each possible next word
    probabilities = {}
    laplace_probabilities = {}
    
    # Go through each unique word and calculate probabilities
    for next_word in vocabulary:
        # Count bigram occurrences
        bigram_count = 0
        for i in range(len(words)-1):
            if words[i] == target_word and words[i+1] == next_word:
                bigram_count += 1
        
        # Calculate regular conditional probability
        if target_count > 0:  # Avoid division by zero
            prob = bigram_count / target_count
        else:
            prob = 0
        probabilities[next_word] = prob
        
        # Calculate Laplace smoothed probability
        laplace_prob = (bigram_count + 1) / (target_count + V)
        laplace_probabilities[next_word] = laplace_prob
    
    return probabilities, laplace_probabilities

# Let's test it with your sentence
test_sentence = "The quick brown fox jumps over the lazy dog"
target = "the"

regular_probs, laplace_probs = find_next_word_probabilities(test_sentence, target)

# Print results in a nicely formatted way
print(f"\nRegular Conditional Probabilities P(word|{target}):")
for word, prob in sorted(regular_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"P({word}|{target}) = {prob:.3f}")

print(f"\nLaplace Smoothed Probabilities P(word|{target}):")
for word, prob in sorted(laplace_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"P({word}|{target}) = {prob:.3f}")