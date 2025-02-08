"""
Evaluation Module
Created: 2025-01-27
Updated: 2025-02-08
Author: Rama Chaganti

This module provides functionality to evaluate the performance of bigram and trigram models using perplexity metrics.
"""

import math
from bigram import BigramModel
from trigram import TrigramModel
from unigram import unigram
from data import brown_corpus, webtext_corpus, reuters_corpus

def calculate_first_word_probability(word, corpus, vocab_size):
    """Calculate probability of first word using unigram model with Laplace smoothing"""
    first_word_count = 0
    total_words = 0
    
    # Count word occurrences and total words
    for sentence in corpus:
        words = sentence.split()
        total_words += len(words)
        for w in words:
            if w == word:
                first_word_count += 1
    
    prob = (first_word_count + 1) / (total_words + vocab_size + 1)
    return prob

def perplexity_bg(sentence, bigrams, corpus):
    """Calculate perplexity using bigram model"""
    words = sentence.split()
    n = len(words)
    prob_product = 1.0
    
    # Handle first word using unigram probability
    first_word_prob = calculate_first_word_probability(words[0], corpus, bigrams.V)
    prob_product *= first_word_prob
    
    # Calculate conditional probabilities for remaining words
    for i in range(1, n):
        prev_word = words[i-1]
        curr_word = words[i]
        prob = bigrams.conditional_prob_bg(prev_word, curr_word)
        prob_product *= prob
    
    # Calculate perplexity using nth root
    perplexity = (1/prob_product) ** (1/n)
    return perplexity

def perplexity_tri(sentence, trigrams, bigrams):
    """Calculate perplexity using trigram model"""
    words = sentence.split()
    n = len(words)
    prob_product = 1.0
    
    # Handle first word using unigram probability
    first_word_prob = calculate_first_word_probability(words[0], bigrams.corpus, bigrams.V)
    prob_product *= first_word_prob
    
    # Handle second word using bigram probability
    if n > 1:
        prob = bigrams.conditional_prob_bg(words[0], words[1])
        prob_product *= prob
    
    # Calculate conditional probabilities for remaining words
    for i in range(2, n):
        prev_prev_word = words[i-2]
        prev_word = words[i-1]
        curr_word = words[i]
        prob = trigrams.conditional_prob_tg(prev_prev_word, prev_word, curr_word)
        prob_product *= prob
    
    # Calculate perplexity using nth root
    perplexity = (1/prob_product) ** (1/n)
    return perplexity



class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    from bigram import BigramModel
    from trigram import TrigramModel
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}Question 1: Brown Corpus Bigram vs Trigram Analysis{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 50}{Colors.ENDC}")
    
    # Create models using Brown corpus
    brown_bigram = BigramModel(brown_corpus[:5000])
    brown_trigram = TrigramModel(brown_corpus[:5000])
    
    # Test sentences from Brown corpus
    test_sentences_brown = [
        "<s> The quick brown fox jumps over the lazy dog </s>",
        "<s> I am going to the market to buy some groceries </s>",
        "<s> The weather is beautiful today and the sun is shining </s>",
        "<s> She opened the book and started reading the first chapter </s>",
        "<s> The students are studying hard for their final exams </s>"
    ]
    
    print(f"\n{Colors.BOLD}Perplexity Comparison on Brown Corpus:{Colors.ENDC}")
    bg_perps_brown = []
    tg_perps_brown = []
    
    for i, sentence in enumerate(test_sentences_brown):
        bg_perp = perplexity_bg(sentence, brown_bigram, brown_corpus[:5000])
        tg_perp = perplexity_tri(sentence, brown_trigram, brown_bigram)
        bg_perps_brown.append(bg_perp)
        tg_perps_brown.append(tg_perp)
        
        print(f"\n{Colors.CYAN}Sentence {i+1}:{Colors.ENDC} {sentence}")
        print(f"{Colors.YELLOW}Bigram Perplexity:{Colors.ENDC} {bg_perp:.2f}")
        print(f"{Colors.YELLOW}Trigram Perplexity:{Colors.ENDC} {tg_perp:.2f}")
        
        # Generate next word predictions
        words = sentence.split()
        bg_pred = brown_bigram.predict_next_word(words[-2])
        tg_pred = brown_trigram.predict_next_word(words[-3], words[-2])
        print(f"{Colors.GREEN}Bigram next word prediction:{Colors.ENDC} {bg_pred}")
        print(f"{Colors.GREEN}Trigram next word prediction:{Colors.ENDC} {tg_pred}")
    
    print(f"\n{Colors.BOLD}Average Perplexity on Brown Corpus:{Colors.ENDC}")
    print(f"{Colors.YELLOW}Bigram:{Colors.ENDC} {sum(bg_perps_brown)/len(bg_perps_brown):.2f}")
    print(f"{Colors.YELLOW}Trigram:{Colors.ENDC} {sum(tg_perps_brown)/len(tg_perps_brown):.2f}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}Question 2: Brown vs Webtext on Reuters Data{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 50}{Colors.ENDC}")
    
    # Create separate models for Brown and Webtext
    brown_bigram = BigramModel(brown_corpus[:5000])
    webtext_bigram = BigramModel(webtext_corpus[:5000])
    
    # Test on Reuters sentences
    test_sentences_reuters = reuters_corpus[:25]
    
    brown_perps = []
    webtext_perps = []
    
    for i, sentence in enumerate(test_sentences_reuters):
        brown_perp = perplexity_bg(sentence, brown_bigram, brown_corpus[:5000])
        webtext_perp = perplexity_bg(sentence, webtext_bigram, webtext_corpus[:5000])
        
        brown_perps.append(brown_perp)
        webtext_perps.append(webtext_perp)
        
        if i < 5:  
            print(f"\n{Colors.CYAN}Reuters sentence {i+1}:{Colors.ENDC} {sentence}")
            print(f"{Colors.YELLOW}Brown Model Perplexity:{Colors.ENDC} {brown_perp:.2f}")
            print(f"{Colors.YELLOW}Webtext Model Perplexity:{Colors.ENDC} {webtext_perp:.2f}")
    
    print(f"\n{Colors.BOLD}Average Perplexity on Reuters:{Colors.ENDC}")
    print(f"{Colors.YELLOW}Brown Model:{Colors.ENDC} {sum(brown_perps)/len(brown_perps):.2f}")
    print(f"{Colors.YELLOW}Webtext Model:{Colors.ENDC} {sum(webtext_perps)/len(webtext_perps):.2f}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}Question 3: Impact of Training Data Size{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 50}{Colors.ENDC}")
    
    # Test with different training sizes
    sizes = [1000, 2000, 5000]
    test_sentence = "<s> The market is expected to improve </s>"
    
    print(f"\n{Colors.CYAN}Testing on sentence:{Colors.ENDC} {test_sentence}")
    print(f"\n{Colors.BOLD}Bigram Model Results:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 20}{Colors.ENDC}")
    for size in sizes:
        bigram_model = BigramModel(brown_corpus[:size])
        bg_perp = perplexity_bg(test_sentence, bigram_model, brown_corpus[:size])
        words = test_sentence.split()
        bg_pred = bigram_model.predict_next_word(words[-2])
        print(f"\n{Colors.YELLOW}Training size:{Colors.ENDC} {size} sentences")
        print(f"{Colors.YELLOW}Perplexity:{Colors.ENDC} {bg_perp:.2f}")
        print(f"{Colors.GREEN}Predicted next word:{Colors.ENDC} {bg_pred}")
    
    print(f"\n{Colors.BOLD}Trigram Model Results:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 20}{Colors.ENDC}")
    for size in sizes:
        bigram_model = BigramModel(brown_corpus[:size])
        trigram_model = TrigramModel(brown_corpus[:size])
        tg_perp = perplexity_tri(test_sentence, trigram_model, bigram_model)
        words = test_sentence.split()
        tg_pred = trigram_model.predict_next_word(words[-3], words[-2])
        print(f"\n{Colors.YELLOW}Training size:{Colors.ENDC} {size} sentences")
        print(f"{Colors.YELLOW}Perplexity:{Colors.ENDC} {tg_perp:.2f}")
        print(f"{Colors.GREEN}Predicted next word:{Colors.ENDC} {tg_pred}")