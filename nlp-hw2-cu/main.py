"""
Sentiment Analysis for Hotel Reviews
Created: 2025-02-18
Updated: 2025-02-18
Author: Rama Chaganti

This module implements a binary sentiment classifier for hotel reviews using logistic regression.
The model takes feature vectors extracted from text reviews and predicts positive/negative sentiment.
"""

from util import load_train_data, load_test_data
from sklearn.model_selection import train_test_split
from features import featurize_text, normalize
from evaluation_metrics import precision, recall, f1
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
import torch
import random


def analyze_train_data_distribution():
    train_texts, train_labels = load_train_data("hotelPosT-train.txt", "hotelNegT-train.txt")
    
    positive_count = sum(train_labels)
    negative_count = len(train_labels) - positive_count

    print(f"\nTotal Training samples: {len(train_texts)}")
    print(f"Positive samples: {positive_count}")
    print(f"Negative samples: {negative_count}")
    
    return train_texts, train_labels


def split_train_data(train_texts, train_labels):
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


def analyze_test_data_distribution():
    
    test_texts, test_labels = load_test_data("HW2-testset.txt")
    
    positive_count = sum(test_labels)
    negative_count = len(test_labels) - positive_count

    print(f"\nTotal Test samples: {len(test_texts)}")
    print(f"Positive samples: {positive_count}")
    print(f"Negative samples: {negative_count}")

    return test_texts, test_labels


class SentimentClassifier(torch.nn.Module):
    def __init__(self, input_dim=6, output_size=1):
        super(SentimentClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_size)
    
    def forward(self, feature_vec):
        z = self.linear(feature_vec)
        return torch.sigmoid(z)
    
    @staticmethod
    def logprob2label(log_prob):
        return 1 if log_prob > 0.5 else 0


def train_and_evaluate(
    train_vectors, 
    train_labels,
    dev_vectors, 
    dev_labels,
    num_epochs=100,
    batch_size=16,
    learning_rate=0.1
):
    model = SentimentClassifier()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_vectors = torch.FloatTensor(train_vectors)
    train_labels = torch.FloatTensor(train_labels)
    dev_vectors = torch.FloatTensor(dev_vectors)
    dev_labels = torch.FloatTensor(dev_labels)
    
    train_losses = []
    dev_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        samples = list(zip(train_vectors, train_labels))
        random.shuffle(samples)
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        
        epoch_train_losses = []
        for batch in tqdm(batches, desc=f"Epoch {epoch+1} Batches", leave=False):
            batch_vectors, batch_labels = zip(*batch)
            batch_vectors = torch.stack(list(batch_vectors))
            batch_labels = torch.stack(list(batch_labels)).reshape(-1, 1)
            
            model.zero_grad()
            log_probs = model(batch_vectors)
            loss = loss_function(log_probs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        
        model.eval()
        with torch.no_grad():
            dev_log_probs = model(dev_vectors)
            dev_loss = loss_function(dev_log_probs, dev_labels.reshape(-1, 1))
            dev_losses.append(dev_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}:')
            print(f'  Training Loss: {avg_train_loss:.4f}')
            print(f'  Development Loss: {dev_loss.item():.4f}')
    
    return model, train_losses, dev_losses


def evaluate_model(model, feature_vectors, true_labels):
    model.eval()
    with torch.no_grad():
        vectors = torch.FloatTensor(feature_vectors)
        probs = model(vectors)
        predicted_labels = [model.logprob2label(p.item()) for p in probs]
        
        prec = precision(predicted_labels, true_labels)
        rec = recall(predicted_labels, true_labels)
        f1_score = f1(predicted_labels, true_labels)
        
        return prec, rec, f1_score


def experiment_with_hyperparameters(train_vectors, train_labels, dev_vectors, dev_labels):
    settings = [
        {'epochs': 100, 'batch_size': 16, 'learning_rate': 0.1},    
        {'epochs': 200, 'batch_size': 16, 'learning_rate': 0.05},  
        {'epochs': 150, 'batch_size': 8, 'learning_rate': 0.1},    
        {'epochs': 150, 'batch_size': 64, 'learning_rate': 0.1},    
        {'epochs': 200, 'batch_size': 32, 'learning_rate': 0.01},   
        {'epochs': 100, 'batch_size': 16, 'learning_rate': 0.2},    
        {'epochs': 300, 'batch_size': 32, 'learning_rate': 0.05},   
        {'epochs': 150, 'batch_size': 16, 'learning_rate': 0.15},   
    ]
    all_results = []
    best_f1 = 0
    best_model = None
    best_params = None
    
    for params in settings:
        print(f"\nTrying parameters: {params}")
        model, train_losses, dev_losses = train_and_evaluate(
            train_vectors, train_labels, dev_vectors, dev_labels,
            num_epochs=params['epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate']
        )
        
        
        final_train_loss = train_losses[-1]
        final_dev_loss = dev_losses[-1]
        
        prec, rec, f1_score = evaluate_model(model, dev_vectors, dev_labels)
        
        result = {
            'params': params,
            'train_loss': final_train_loss,
            'dev_loss': final_dev_loss,
            'precision': prec,
            'recall': rec,
            'f1': f1_score,
            'model': model
        }
        all_results.append(result)
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model = model
            best_params = params
    print("\n" + "="*100)
    print("Experiment Results:")
    print("="*100)
    print(f"{'Epochs':^10} | {'Batch':^10} | {'LR':^8} | {'Train Loss':^12} | {'Dev Loss':^12} | {'Precision':^10} | {'Recall':^10} | {'F1':^10} |")
    print("-"*100)
    
    for result in all_results:
        params = result['params']
        print(f"{params['epochs']:^10} | {params['batch_size']:^10} | {params['learning_rate']:^8.3f} | "
              f"{result['train_loss']:^12.4f} | {result['dev_loss']:^12.4f} | "
              f"{result['precision']:^10.4f} | {result['recall']:^10.4f} | {result['f1']:^10.4f} |")
    
    print("-"*100)
    print(f"Best Model Parameters: {best_params}")
    print(f"Best F1 Score (dev set): {best_f1:.4f}")
    print("="*100)
    return best_model, best_params


def main():
    train_texts, train_labels = analyze_train_data_distribution()
    x_train_texts, x_dev_texts, y_train_labels, y_dev_labels = split_train_data(train_texts, train_labels)
    
    train_vectors = [featurize_text(text) for text in x_train_texts]
    train_vectors = normalize(train_vectors)
    dev_vectors = [featurize_text(text) for text in x_dev_texts]
    dev_vectors = normalize(dev_vectors)
    
    model, train_losses, dev_losses = train_and_evaluate(train_vectors, y_train_labels, dev_vectors, y_dev_labels)
    
    prec, rec, f1_score = evaluate_model(model, dev_vectors, y_dev_labels)
    print(f"\nDevelopment Set Metrics:")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    best_model, best_params = experiment_with_hyperparameters(train_vectors, y_train_labels, dev_vectors, y_dev_labels)
    
    test_texts, test_labels = analyze_test_data_distribution()
    test_vectors = [featurize_text(text) for text in test_texts]
    test_vectors = normalize(test_vectors)
    prec, rec, f1_score = evaluate_model(best_model, test_vectors, test_labels)
    print(f"\nTest Set Metrics (Best Model):")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Best Parameters: {best_params}")


if __name__ == "__main__":
    main()