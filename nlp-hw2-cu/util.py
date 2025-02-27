"""
Provided by the instructor
Utility functions for loading and processing data
"""
from typing import List, Tuple

def load_train_data(
    positive_filepath: str,
    negative_filepath: str
) -> Tuple[List[str], List[int]]:
    """Load the training data, producing Lists of text and labels

    Args:
        filepath (str): Path to the training file

    Returns:
        Tuple[List[str], List[int]]: The texts and labels
    """

    def _read(filename: str):
        texts = []
        with open(filename,"r", encoding="utf-8") as f:
            for line in f:
                _id, text = line.rstrip().split("\t")
                texts.append(text)
        return texts

    texts = []
    labels = []
    for text in _read(positive_filepath):
        texts.append(text)
        labels.append(1)

    for text in _read(negative_filepath):
        texts.append(text)
        labels.append(0)

    return texts, labels


def load_test_data(filepath: str) -> List[str]:
    """Load the test data, producing a List of texts

    Args:
        filepath (str): Path to the training file

    Returns:
        List[str]: The texts and labels
    """
    
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            idx, text, label = line.rstrip().split("\t")
            text = text.strip('"')
            texts.append(text)
            if label == 'POS':
                label = 1
            else:
                label = 0
            labels.append(label)

    return texts, labels

