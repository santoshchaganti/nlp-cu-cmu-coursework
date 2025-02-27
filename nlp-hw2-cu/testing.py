# """
# Sentiment Analysis for Hotel Reviews
# Created: 2025-02-18
# Updated: 2025-02-18
# Author: Rama Chaganti

# This module implements a binary sentiment classifier for hotel reviews using logistic regression.
# The model takes feature vectors extracted from text reviews and predicts positive/negative sentiment.
# """

# from util import load_train_data, load_test_data
# from sklearn.model_selection import train_test_split
# from features import featurize_text, normalize
# from evaluation_metrics import precision, recall, f1
from typing import List, Tuple, Dict
# from tqdm import tqdm
# import numpy as np
# import torch
# import random


# def analyze_train_data_distribution() -> Tuple[List[str], List[int]]:
#     """
#     Analyze and print the distribution of positive and negative samples in training data.
    
#     Returns:
#         Tuple containing:
#             - List of training texts
#             - List of corresponding labels (0 for negative, 1 for positive)
#     """
#     train_texts, train_labels = load_train_data("hotelPosT-train.txt", "hotelNegT-train.txt")
    
#     positive_count = sum(train_labels)
#     negative_count = len(train_labels) - positive_count

#     print(f"\nTotal Training samples: {len(train_texts)}")
#     print(f"Positive samples: {positive_count}")
#     print(f"Negative samples: {negative_count}")
    
#     return train_texts, train_labels


def split_train_data(train_texts: List[str], train_labels: List[int]) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Split training data into training and development sets with stratification.
    
    Args:
        train_texts: List of training text samples
        train_labels: List of corresponding labels
    
    Returns:
        Tuple containing:
            - Training texts (80% of data)
            - Development texts (20% of data)
            - Training labels
            - Development labels
    """
    x_train_texts, x_dev_texts, y_train_labels, y_dev_labels = train_test_split(
        train_texts, train_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_labels
    )
    
    print("\nData Split Information:")
    print(f"Training Set Size: {len(x_train_texts)}")
    print(f"Dev Set Size: {len(x_dev_texts)}")
    
    print(f"\nTraining Set Distribution:")
    print(f"Positive samples: {sum(y_train_labels)}")
    print(f"Negative samples: {len(y_train_labels) - sum(y_train_labels)}")
    
    print(f"\nDev Set Distribution:")
    print(f"Positive samples: {sum(y_dev_labels)}")
    print(f"Negative samples: {len(y_dev_labels) - sum(y_dev_labels)}")
    
    return x_train_texts, x_dev_texts, y_train_labels, y_dev_labels


# def analyze_test_data_distribution() -> Tuple[List[str], List[int]]:
#     """
#     Analyze and print the distribution of positive and negative samples in test data.
    
#     Returns:
#         Tuple containing:
#             - List of test texts
#             - List of corresponding labels
#     """
#     test_texts, test_labels = load_test_data("HW2-testset.txt")
    
#     positive_count = sum(test_labels)
#     negative_count = len(test_labels) - positive_count

#     print(f"\nTotal Test samples: {len(test_texts)}")
#     print(f"Positive samples: {positive_count}")
#     print(f"Negative samples: {negative_count}")

#     return test_texts, test_labels


# class SentimentClassifier(torch.nn.Module):
#     """
#     Binary logistic regression classifier for sentiment analysis.
    
#     The model takes a 6-dimensional feature vector as input and outputs
#     a probability between 0 and 1 indicating positive sentiment.
#     """
    
#     def __init__(self, input_dim: int = 6, output_size: int = 1):
#         """
#         Initialize the classifier.
        
#         Args:
#             input_dim: Dimension of input feature vectors (default: 6)
#             output_size: Dimension of output (default: 1 for binary classification)
#         """
#         super(SentimentClassifier, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_size)
    
#     def forward(self, feature_vec: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the model.
        
#         Args:
#             feature_vec: Input feature vectors
            
#         Returns:
#             Probability of positive sentiment (between 0 and 1)
#         """
#         z = self.linear(feature_vec)
#         return torch.sigmoid(z)
    
#     @staticmethod
#     def logprob2label(log_prob: float) -> int:
#         """
#         Convert probability to binary label.
        
#         Args:
#             log_prob: Probability from model output
            
#         Returns:
#             1 for positive sentiment (prob > 0.5), 0 for negative
#         """
#         return 1 if log_prob > 0.5 else 0


# def train_and_evaluate(
#     train_vectors: List[List[float]], 
#     train_labels: List[int],
#     dev_vectors: List[List[float]], 
#     dev_labels: List[int],
#     num_epochs: int = 100,
#     batch_size: int = 16,
#     learning_rate: float = 0.1
# ) -> Tuple[SentimentClassifier, List[float], List[float]]:
#     """
#     Train the sentiment classifier and evaluate on development set.
    
#     Args:
#         train_vectors: Normalized feature vectors for training
#         train_labels: Training labels (0 or 1)
#         dev_vectors: Normalized feature vectors for development
#         dev_labels: Development labels (0 or 1)
#         num_epochs: Number of training epochs
#         batch_size: Size of training batches
#         learning_rate: Learning rate for optimizer
    
#     Returns:
#         Tuple containing:
#             - Trained model
#             - List of training losses per epoch
#             - List of development losses per epoch
#     """
#     # Initialize model and training components
#     model = SentimentClassifier()
#     loss_function = torch.nn.BCELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
#     # Convert data to PyTorch tensors
#     train_vectors = torch.FloatTensor(train_vectors)
#     train_labels = torch.FloatTensor(train_labels)
#     dev_vectors = torch.FloatTensor(dev_vectors)
#     dev_labels = torch.FloatTensor(dev_labels)
    
#     # Track losses
#     train_losses = []
#     dev_losses = []
    
#     for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
#         model.train()
#         samples = list(zip(train_vectors, train_labels))
#         random.shuffle(samples)
#         batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        
#         epoch_train_losses = []
#         for batch in tqdm(batches, desc=f"Epoch {epoch+1} Batches", leave=False):
#             batch_vectors, batch_labels = zip(*batch)
#             batch_vectors = torch.stack(list(batch_vectors))
#             batch_labels = torch.stack(list(batch_labels)).reshape(-1, 1)
            
#             model.zero_grad()
#             log_probs = model(batch_vectors)
#             loss = loss_function(log_probs, batch_labels)
#             loss.backward()
#             optimizer.step()
            
#             epoch_train_losses.append(loss.item())
        
#         avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
#         train_losses.append(avg_train_loss)
        
#         # Evaluation phase
#         model.eval()
#         with torch.no_grad():
#             dev_log_probs = model(dev_vectors)
#             dev_loss = loss_function(dev_log_probs, dev_labels.reshape(-1, 1))
#             dev_losses.append(dev_loss.item())
        
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch {epoch+1}:')
#             print(f'  Training Loss: {avg_train_loss:.4f}')
#             print(f'  Development Loss: {dev_loss.item():.4f}')
    
#     return model, train_losses, dev_losses


# def evaluate_model(
#     model: SentimentClassifier, 
#     feature_vectors: List[List[float]], 
#     true_labels: List[int]
# ) -> Tuple[float, float, float]:
#     """
#     Evaluate model using precision, recall, and F1 score.
    
#     Args:
#         model: Trained SentimentClassifier
#         feature_vectors: Normalized feature vectors
#         true_labels: True labels (0 or 1)
    
#     Returns:
#         Tuple of (precision, recall, f1) scores
#     """
#     model.eval()
#     with torch.no_grad():
#         vectors = torch.FloatTensor(feature_vectors)
#         probs = model(vectors)
#         predicted_labels = [model.logprob2label(p.item()) for p in probs]
        
#         prec = precision(predicted_labels, true_labels)
#         rec = recall(predicted_labels, true_labels)
#         f1_score = f1(predicted_labels, true_labels)
        
#         return prec, rec, f1_score


# def experiment_with_hyperparameters(
#     train_vectors: List[List[float]], 
#     train_labels: List[int],
#     dev_vectors: List[List[float]], 
#     dev_labels: List[int]
# ) -> Tuple[SentimentClassifier, Dict]:
#     """
#     Experiment with different hyperparameter settings to find the best model.
    
#     Args:
#         train_vectors: Training feature vectors
#         train_labels: Training labels
#         dev_vectors: Development feature vectors
#         dev_labels: Development labels
    
#     Returns:
#         Tuple containing:
#             - Best performing model
#             - Dictionary of best hyperparameters
#     """
#     settings = [
#         {'epochs': 100, 'batch_size': 16, 'learning_rate': 0.1},    # Original default
#         {'epochs': 200, 'batch_size': 16, 'learning_rate': 0.05},   # More epochs, moderate lr
#         {'epochs': 150, 'batch_size': 8, 'learning_rate': 0.1},     # Smaller batch
#         {'epochs': 150, 'batch_size': 64, 'learning_rate': 0.1},    # Larger batch
#         {'epochs': 200, 'batch_size': 32, 'learning_rate': 0.01},   # More epochs, smaller lr
#         {'epochs': 100, 'batch_size': 16, 'learning_rate': 0.2},    # Larger learning rate
#         {'epochs': 300, 'batch_size': 32, 'learning_rate': 0.05},   # Many epochs
#         {'epochs': 150, 'batch_size': 16, 'learning_rate': 0.15},   # Moderate increase in lr
#     ]
#     all_results = []
#     best_f1 = 0
#     best_model = None
#     best_params = None
    
#     for params in settings:
#         print(f"\nTrying parameters: {params}")
#         model, train_losses, dev_losses = train_and_evaluate(
#             train_vectors, train_labels, dev_vectors, dev_labels,
#             num_epochs=params['epochs'],
#             batch_size=params['batch_size'],
#             learning_rate=params['learning_rate']
#         )
        
#         # Get final losses
#         final_train_loss = train_losses[-1]
#         final_dev_loss = dev_losses[-1]
        
#         # Evaluate on dev set
#         prec, rec, f1_score = evaluate_model(model, dev_vectors, dev_labels)
        
#         # Store results
#         result = {
#             'params': params,
#             'train_loss': final_train_loss,
#             'dev_loss': final_dev_loss,
#             'precision': prec,
#             'recall': rec,
#             'f1': f1_score,
#             'model': model
#         }
#         all_results.append(result)
        
#         if f1_score > best_f1:
#             best_f1 = f1_score
#             best_model = model
#             best_params = params
#     # Print results table
#     print("\n" + "="*100)
#     print("Experiment Results:")
#     print("="*100)
#     print(f"{'Epochs':^10} | {'Batch':^10} | {'LR':^8} | {'Train Loss':^12} | {'Dev Loss':^12} | {'Precision':^10} | {'Recall':^10} | {'F1':^10} |")
#     print("-"*100)
    
#     for result in all_results:
#         params = result['params']
#         print(f"{params['epochs']:^10} | {params['batch_size']:^10} | {params['learning_rate']:^8.3f} | "
#               f"{result['train_loss']:^12.4f} | {result['dev_loss']:^12.4f} | "
#               f"{result['precision']:^10.4f} | {result['recall']:^10.4f} | {result['f1']:^10.4f} |")
    
#     print("-"*100)
#     print(f"Best Model Parameters: {best_params}")
#     print(f"Best F1 Score (dev set): {best_f1:.4f}")
#     print("="*100)
#     return best_model, best_params


# def main():
#     """Main function to run the sentiment analysis pipeline."""
#     # Load and analyze data
#     train_texts, train_labels = analyze_train_data_distribution()
#     x_train_texts, x_dev_texts, y_train_labels, y_dev_labels = split_train_data(train_texts, train_labels)
    
#     # Prepare feature vectors
#     train_vectors = [featurize_text(text) for text in x_train_texts]
#     train_vectors = normalize(train_vectors)
#     dev_vectors = [featurize_text(text) for text in x_dev_texts]
#     dev_vectors = normalize(dev_vectors)
    
#     # Train and evaluate initial model
#     model, train_losses, dev_losses = train_and_evaluate(train_vectors, y_train_labels, dev_vectors, y_dev_labels)
    
#     # Evaluate on dev set
#     prec, rec, f1_score = evaluate_model(model, dev_vectors, y_dev_labels)
#     print(f"\nDevelopment Set Metrics:")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall: {rec:.4f}")
#     print(f"F1 Score: {f1_score:.4f}")
    
#     # Experiment with different settings
#     best_model, best_params = experiment_with_hyperparameters(train_vectors, y_train_labels, dev_vectors, y_dev_labels)
    
#     # Final evaluation on test set
#     test_texts, test_labels = analyze_test_data_distribution()
#     test_vectors = [featurize_text(text) for text in test_texts]
#     test_vectors = normalize(test_vectors)
#     prec, rec, f1_score = evaluate_model(best_model, test_vectors, test_labels)
#     print(f"\nTest Set Metrics (Best Model):")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall: {rec:.4f}")
#     print(f"F1 Score: {f1_score:.4f}")
#     print(f"Best Parameters: {best_params}")


# if __name__ == "__main__":
#     main()
    


# """
# Feature Engineering for Sentiment Analysis
# Created: 2025-02-18
# Updated: 2025-02-18
# Author: Rama Chaganti

# This module implements feature extraction and normalization for sentiment analysis.
# It creates a 6-dimensional feature vector from text and normalizes features to [0,1] range.
# """

# import math
# from typing import List


# def featurize_text(text: str) -> List[float]:
#     """
#     Create a 6-dimensional feature vector from input text.
    
#     Features:
#         x1: Count of positive lexicon words
#         x2: Count of negative lexicon words
#         x3: Binary indicator for presence of "no"
#         x4: Count of first and second person pronouns
#         x5: Binary indicator for presence of "!"
#         x6: Natural log of word count
    
#     Args:
#         text: Input text to featurize
    
#     Returns:
#         List[float]: 6-dimensional feature vector [x1, x2, x3, x4, x5, x6]
#     """
#     # Load lexicons
#     with open("positive-words.txt", "r") as f:
#         positive_words = set(f.read().splitlines())  # Using set for faster lookup
    
#     with open("negative-words.txt", "r") as f:
#         negative_words = set(f.read().splitlines())  # Using set for faster lookup
    
#     # Preprocess text
#     words = text.lower().split()
    
#     # x1: Count of positive words
#     x1 = sum(1 for word in words if word in positive_words)
    
#     # x2: Count of negative words
#     x2 = sum(1 for word in words if word in negative_words)
    
#     # x3: Binary feature for presence of "no"
#     x3 = 1 if "no" in words else 0
    
#     # x4: Count of first and second person pronouns
#     first_second_person = {"i", "me", "my", "mine", "we", "us", "our", "ours", 
#                           "you", "your", "yours"}
#     x4 = sum(1 for word in words if word in first_second_person)
    
#     # x5: Binary feature for exclamation mark
#     x5 = 1 if "!" in text else 0
    
#     # x6: Natural log of word count
#     x6 = math.log(len(words))
    
#     return [x1, x2, x3, x4, x5, x6]


# def normalize(feature_vectors: List[List[float]]) -> List[List[float]]:
#     """
#     Normalize each feature across all vectors to range [0,1].
    
#     For each feature i, applies the formula:
#     x'ᵢ = (xᵢ - min(xᵢ)) / (max(xᵢ) - min(xᵢ))
    
#     Args:
#         feature_vectors: List of feature vectors, where each vector is [x1, x2, x3, x4, x5, x6]
    
#     Returns:
#         List[List[float]]: Normalized feature vectors with values rounded to 3 decimal places
    
#     Example:
#         >>> vectors = [[3, 2, 1, 3, 0, 4.19], [2, 1, 0, 2, 1, 3.91]]
#         >>> normalize(vectors)
#         [[1.0, 1.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
#     """
#     # Create a copy to avoid modifying the original vectors
#     normalized = [vector.copy() for vector in feature_vectors]
    
#     # Get number of features (should be 6)
#     n_features = len(feature_vectors[0])
    
#     # For each feature index (0 to 5)
#     for j in range(n_features):
#         # Get all values for this feature across all vectors
#         feature_values = [vector[j] for vector in feature_vectors]
        
#         # Find min and max for this feature
#         min_val = min(feature_values)
#         max_val = max(feature_values)
        
#         # Handle case where all values are the same
#         if max_val == min_val:
#             normalized_value = 1 if max_val > 0 else 0
#             for i in range(len(normalized)):
#                 normalized[i][j] = round(normalized_value, 3)
#         else:
#             # Normalize each value for this feature
#             for i in range(len(normalized)):
#                 normalized[i][j] = round(
#                     (feature_vectors[i][j] - min_val) / (max_val - min_val), 
#                     3
#                 )
    
#     return normalized


# """
# Evaluation Metrics for Sentiment Analysis
# Created: 2025-02-18
# Updated: 2025-02-18
# Author: Rama Chaganti

# This module implements evaluation metrics (precision, recall, F1) for binary classification
# tasks, specifically focused on sentiment analysis. The metrics are calculated based on
# true positives, false positives, and false negatives in the predicted vs true labels.
# """

# from typing import List, Union


# def precision(predicted_labels: Union[List[int], List[bool]], 
#              true_labels: Union[List[int], List[bool]]) -> float:
#     """
#     Calculate precision score for binary classification.
#     Precision = TP / (TP + FP)
    
#     Args:
#         predicted_labels: List of predicted labels (0 or 1)
#         true_labels: List of true labels (0 or 1)
    
#     Returns:
#         float: Precision score between 0 and 1
        
#     Raises:
#         ValueError: If lengths of predicted and true labels don't match
#     """
#     predicted_labels = list(predicted_labels)
#     true_labels = list(true_labels)
    
#     if len(predicted_labels) != len(true_labels):
#         raise ValueError("Length of predicted and true labels must match")
    
#     # Calculate true positives and false positives
#     true_positives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 1)
#     false_positives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 0)
    
#     # Handle division by zero case
#     if true_positives + false_positives == 0:
#         return 0.0
    
#     return true_positives / (true_positives + false_positives)


# def recall(predicted_labels: Union[List[int], List[bool]], 
#           true_labels: Union[List[int], List[bool]]) -> float:
#     """
#     Calculate recall score for binary classification.
#     Recall = TP / (TP + FN)
    
#     Args:
#         predicted_labels: List of predicted labels (0 or 1)
#         true_labels: List of true labels (0 or 1)
    
#     Returns:
#         float: Recall score between 0 and 1
        
#     Raises:
#         ValueError: If lengths of predicted and true labels don't match
#     """
#     predicted_labels = list(predicted_labels)
#     true_labels = list(true_labels)
    
#     if len(predicted_labels) != len(true_labels):
#         raise ValueError("Length of predicted and true labels must match")
    
#     # Calculate true positives and false negatives
#     true_positives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 1)
#     false_negatives = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 0 and t == 1)
    
#     # Handle division by zero case
#     if true_positives + false_negatives == 0:
#         return 0.0
    
#     return true_positives / (true_positives + false_negatives)


# def f1(predicted_labels: Union[List[int], List[bool]], 
#        true_labels: Union[List[int], List[bool]]) -> float:
#     """
#     Calculate F1 score for binary classification.
#     F1 = 2 * (precision * recall) / (precision + recall)
    
#     The F1 score is the harmonic mean of precision and recall, providing a single
#     score that balances both metrics.
    
#     Args:
#         predicted_labels: List of predicted labels (0 or 1)
#         true_labels: List of true labels (0 or 1)
    
#     Returns:
#         float: F1 score between 0 and 1
#     """
#     p = precision(predicted_labels, true_labels)
#     r = recall(predicted_labels, true_labels)
    
#     # Handle division by zero case
#     if p + r == 0:
#         return 0.0
    
#     return 2 * (p * r) / (p + r)