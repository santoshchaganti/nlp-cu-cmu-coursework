"""
Evaluation Metrics for Sentiment Analysis
Created: 2025-02-18
Updated: 2025-02-18
Author: Rama Chaganti

This module implements evaluation metrics (precision, recall, F1) for binary classification
tasks, specifically focused on sentiment analysis. The metrics are calculated based on
true positives, false positives, and false negatives in the predicted vs true labels.
"""

def precision(predicted_labels, true_labels):
    predicted_labels = list(predicted_labels)
    true_labels = list(true_labels)
    true_positives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 1)
    false_positives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 0)
    return true_positives / (true_positives + false_positives)


def recall(predicted_labels, true_labels):
    predicted_labels = list(predicted_labels)
    true_labels = list(true_labels)
    true_positives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 1)
    false_negatives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 0 and t == 1)
    return true_positives / (true_positives + false_negatives)


def f1(predicted_labels, true_labels):
    p = precision(predicted_labels, true_labels)
    r = recall(predicted_labels, true_labels)
    return 2 * (p * r) / (p + r)