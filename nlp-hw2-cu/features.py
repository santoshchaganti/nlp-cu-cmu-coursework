"""
Feature Engineering for Sentiment Analysis
Created: 2025-02-18
Updated: 2025-02-18
Author: Rama Chaganti

This module implements feature extraction and normalization for sentiment analysis.
It creates a 6-dimensional feature vector from text and normalizes features to [0,1] range.
"""

import math


def featurize_text(text):
    
    with open("positive-words.txt", "r") as f:
        positive_words = set(f.read().splitlines())
    
    with open("negative-words.txt", "r") as f:
        negative_words = set(f.read().splitlines())  
    
    words = text.lower().split()

    x1 = sum(1 for word in words if word in positive_words)
    x2 = sum(1 for word in words if word in negative_words)
    x3 = 1 if "no" in words else 0
    first_second_person = {"i", "me", "my", "mine", "we", "us", "our", "ours", 
                          "you", "your", "yours"}
    x4 = sum(1 for word in words if word in first_second_person)   
    x5 = 1 if "!" in text else 0
    x6 = math.log(len(words))
    
    return [x1, x2, x3, x4, x5, x6]


def normalize(feature_vectors):
    normalized = [vector.copy() for vector in feature_vectors]
    
    n_features = len(feature_vectors[0])
    
    for j in range(n_features):
        feature_values = [vector[j] for vector in feature_vectors]
        
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        if max_val == min_val:
            normalized_value = 1 if max_val > 0 else 0
            for i in range(len(normalized)):
                normalized[i][j] = round(normalized_value, 3)
        else:
            for i in range(len(normalized)):
                normalized[i][j] = round(
                    (feature_vectors[i][j] - min_val) / (max_val - min_val), 
                    3
                )
    
    return normalized