"""
Bigram Language Model Implementation
Created: 2025-01-27
Updated: 2025-01-28
Author: Rama Chaganti

This module implements a basic bigram language model using various text corpora
from NLTK. It provides functionality to calculate bigram probabilities for words
using different corpora (Brown, Reuters, or Webtext).
"""

# test_sentence = "The quick brown fox jumps over the lazy dog"
test_sentence = "The smart student reads the interesting book because the student loves readin"

# a function that finds all bigrams in a sentence and returns them as a list of tuples
def bigram_list_of_tuples(test_sentence):
    words = test_sentence.split()
    list_tuples = list(zip(words[:-1], words[1:]))
    # print(list_tuples)
    return list_tuples

# bigram_list_of_tuples(test_sentence)

# Todo 
# case sensitive case get_V

# function to get the number of unique words in the list of tuples
def get_V(list_tuples):
    # list_tuples = bigram_list_of_tuples(test_sentence)
    unique_words = {word for tuple_pair in list_tuples for word in tuple_pair}
    # print(unique_words)
    return len(unique_words), unique_words

# function to calculate the conditional probability of a bigram
def conditional_prob_bg(word1, word2, test_sentence):
    list_tuples = bigram_list_of_tuples(test_sentence)
    V, unique_words = get_V(list_tuples)
    # print(f"V: {V}")
    def count_bigram(w1, w2):
        count = 0
        for i in range(len(list_tuples)):
            if list_tuples[i] == (w1, w2):
                count += 1
        return count
    def count_word(w):
        count = 0
        for i in range(len(list_tuples)):
            if list_tuples[i][0] == w:
                count += 1
        return count
    bigram_count = count_bigram(word1, word2)
    # print(f"bigram_count: {bigram_count}")
    word1_count = count_word(word1)
    # print(f"word1_count: {word1_count}" )
    probability = (bigram_count + 1) / (word1_count + V + 1)
    # print(f"probability: {probability}")
    return probability

# print(f"The probability of 'brown' following 'quick' is {conditional_prob_bg('quick', 'brown', test_sentence)}")

def predict_next_word(word1, test_sentence):
    list_tuples = bigram_list_of_tuples(test_sentence)
    V, unique_words = get_V(list_tuples)
    probabilities = {}
    for word in unique_words:
        probabilities[word] = conditional_prob_bg(word1, word, test_sentence)
    max_prob_word = max(probabilities, key=probabilities.get)
    print(max_prob_word)
    return max_prob_word

# predict_next_word('The', test_sentence)

def predict_sentence(sentence, test_sentence, limit = 6):
    words = sentence.split()
    words_list = []
    

predict_sentence('The brown santosh', test_sentence)


# for i in range(len(words) - 1):
#         words_list.append(predict_next_word(words[i], test_sentence))
#         # test_sentence = test_sentence + " " + predict_next_word(words[i], test_sentence)
#         if i == limit:
#             break
#     print(words_list)
#     return words_list