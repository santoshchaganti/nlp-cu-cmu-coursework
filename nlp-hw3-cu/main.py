'''
CSCI 5832 Assignment 3
Spring 2025
Implementation of data preprocessing, feedforward neural network training,
and embedding analysis.
'''

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import gensim.downloader as api

nltk.download('wordnet')
nltk.download('punkt')

print("=== Starting Data Preprocessing ===")

# Load the Rotten Tomatoes dataset
def load_rt_dataset():
    reviews = []
    for sentiment in ['pos', 'neg']:
        path = f'rt-polarity.{sentiment}'
        file = open(path)
        for line in file.readlines():
            review = line.strip()
            reviews.append({'review': review, 'sentiment': sentiment})
    return pd.DataFrame(reviews)

reviews = load_rt_dataset()
print("Sample of loaded reviews:")
print(reviews.head())

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in lemmatized_tokens if token.strip()]
    return tokens

# Preprocess all documents
preprocessed_documents = reviews['review'].apply(preprocess)
print(f'\nNumber of preprocessed documents: {len(preprocessed_documents)}')
print(preprocessed_documents.head())

# Make a token2index dictionary and a index2token dictionary and convert the documents to sequences of indices
token2index = {}
index2token = {}
index = 1  # reserve 0 for padding
for document in preprocessed_documents:
    for token in document:
        if token not in token2index:
            token2index[token] = index
            index2token[index] = token
            index += 1

token2index['[PAD]'] = 0
index2token[0] = '[PAD]'

print(f'Vocabulary size: {len(token2index)}')

# Convert the dataset into sequences of indices
def document_to_sequence(document: str) -> list:
    return [token2index[token] for token in document]

sequences = preprocessed_documents.apply(document_to_sequence)
print(sequences.head())

# Truncate the sequences
def pad_sequence(sequence: list, max_length: int, padding_token: int = 0) -> list:
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [padding_token] * (max_length - len(sequence))

max_length = 40
truncated_sequences = sequences.apply(lambda x: pad_sequence(x, max_length))
print(truncated_sequences.head())

# Split the dataset into training and testing sets
X_train, X_test_and_val, y_train, y_test_and_val = train_test_split(truncated_sequences, reviews['sentiment'], test_size=0.2, random_state=123)
X_test, X_val, y_test, y_val = train_test_split(X_test_and_val, y_test_and_val, test_size=0.5, random_state=42)

# Encode the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train.tolist())
y_val = label_encoder.transform(y_val.tolist())
y_test = label_encoder.transform(y_test.tolist())

# Convert the vectorized reviews to numpy arrays
X_train = torch.tensor(X_train.tolist())
X_val = torch.tensor(X_val.tolist())
X_test = torch.tensor(X_test.tolist())
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)

# Define the dataset class and dataloader
batch_size = 16
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=len(test_data))

print([index2token[i.item()] for i in train_data.tensors[0][0]])

print("\n=== Starting Model Training ===")

class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, 
                 num_embeddings: int, embedding_dim: int, padding_idx: int = 0, 
                 pretrained_embeddings: torch.Tensor = None, freeze_embeddings: bool = False):
        super(Net, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=padding_idx, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Hyperparameters that worked for TA on a CPU:
d = 50 
h = 50  
num_epochs = 10
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = Net(
    input_size=d,
    output_size=1,
    hidden_size=h,
    num_embeddings=len(token2index),
    embedding_dim=d,
    padding_idx=0
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopper = EarlyStopper(3, 0.01)

print("\nTraining custom embedding model:")
# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(train_loader):
        inputs, targets = batch
        targets = targets.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)

    # Validation
    model.eval()
    val_losses = []
    for input_, target in val_loader:
        with torch.no_grad():
            targets = target.float().unsqueeze(1)
            outputs = model(input_)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses)
    
    if early_stopper.early_stop(val_loss):
        print("Early stopping triggered")
        break
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f'Training loss: {avg_train_loss:.4f}')
    print(f'Validation loss: {val_loss:.4f}')

model.eval()
for batch in test_loader:
    test_inputs, test_targets = batch
with torch.no_grad():
    test_outputs = model(test_inputs)

predictions = test_outputs.view(-1)
predictions = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])
print(f'accuracy: {accuracy_score(test_targets, predictions)}')
print(f'precision: {precision_score(test_targets, predictions)}')
print(f'recall: {recall_score(test_targets, predictions)}')
print(f'f1: {f1_score(test_targets, predictions)}')

# Train with GloVe embeddings
print("\nTraining with GloVe embeddings:")
word_vectors = api.load(f"glove-wiki-gigaword-{d}")

embedding_matrix = torch.zeros(len(token2index), d)
for token, index in token2index.items():
    if token in word_vectors:
        embedding_matrix[index] = torch.tensor(word_vectors[token])

pretrained_model = Net(
    input_size=d,
    output_size=1,
    hidden_size=h,
    num_embeddings=len(token2index),
    embedding_dim=d,
    padding_idx=0,
    pretrained_embeddings=embedding_matrix,
    freeze_embeddings=False
)

optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=learning_rate)
early_stopper = EarlyStopper(3, 0.01)

# Training loop for GloVe model
for epoch in range(num_epochs):
    pretrained_model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        inputs, targets = batch
        targets = targets.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = pretrained_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation
    pretrained_model.eval()
    val_losses = []
    for input_, target in val_loader:
        with torch.no_grad():
            targets = target.float().unsqueeze(1)
            outputs = pretrained_model(input_)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses)
    
    if early_stopper.early_stop(val_loss):
        print("Early stopping triggered")
        break
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f'Training loss: {avg_train_loss:.4f}')
    print(f'Validation loss: {val_loss:.4f}')

print("\n=== Starting Embedding Analysis ===")

def k_nearest_neighbors(embeddings, token2index, token, k: int = 5):
    token_index = token2index[token]
    token_embedding = embeddings[token_index]
    similarities = cosine_similarity([token_embedding], embeddings)[0]
    nearest_neighbors = np.argsort(similarities)[-k-1:][::-1]
    top_indices = [index for index in nearest_neighbors if index != token_index][:k]
    index2token = {index: token for token, index in token2index.items()}
    neighbors = [index2token[index] for index in top_indices]
    return neighbors

def plot_embeddings_tsne(embeddings, num_embeddings_to_plot: int = 2000, pca_n_components: int = 50):
    # Don't plot stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    non_stop_words = [token for token in list(token2index.keys()) if token not in stop_words]
    indices_to_take = [token2index[token] for token in non_stop_words[:num_embeddings_to_plot]]
    subset_of_embeddings = embeddings[indices_to_take, :]

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    random_state = 42
    vocab = [index2token[index] for index in indices_to_take]
    colors = ['black' for i in vocab]
    # reduction with PCA
    pca = PCA(n_components=pca_n_components, random_state=random_state)
    X = pca.fit_transform(subset_of_embeddings)
    # t-SNE:
    tsne = TSNE(
        n_components=2,
        init='random',
        learning_rate='auto',
        random_state=random_state)
    tsnemat = tsne.fit_transform(X)
    # Plot values:
    xvals = tsnemat[: , 0]
    yvals = tsnemat[: , 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,30))
    ax.plot(xvals, yvals, marker='', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        try:
            ax.annotate(word, (x, y), fontsize=8, color=color)
        except UnicodeDecodeError:
            pass
    plt.axis('off')
    plt.show()
    
def plot_embeddings_3d(embeddings, num_embeddings_to_plot: int = 250):

    # Don't plot stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    non_stop_words = [token for token in list(token2index.keys()) if token not in stop_words]
    indices_to_take = [token2index[token] for token in non_stop_words[:num_embeddings_to_plot]]
    subset_of_embeddings = embeddings[indices_to_take, :]

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(subset_of_embeddings)

    # Create a 3D scatter plot of the projection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(components[:, 0], components[:, 1], components[:, 2], s=10)

    for i, index_in_dict in enumerate(indices_to_take):
        token = index2token[index_in_dict]
        ax.text(components[i, 0], components[i, 1], components[i, 2], token)
        
    plt.show()


# Analyze embeddings
print("\nAnalyzing Custom Embeddings:")
words_to_analyze = ['good', 'bad', 'excellent', 'terrible', 'movie']
for word in words_to_analyze:
    neighbors = k_nearest_neighbors(model.embedding.weight.data, token2index, word, k=5)
    print(f"Nearest neighbors for '{word}': {neighbors}")

print("\nAnalyzing GloVe Embeddings:")
for word in words_to_analyze:
    neighbors = k_nearest_neighbors(pretrained_model.embedding.weight.data, token2index, word, k=5)
    print(f"Nearest neighbors for '{word}': {neighbors}")

# Visualize embeddings
print("Visualizing Custom Embeddings:")
plot_embeddings_tsne(model.embedding.weight.data.numpy())
plot_embeddings_3d(model.embedding.weight.data.numpy())

print("\nVisualizing GloVe Embeddings:")
plot_embeddings_tsne(pretrained_model.embedding.weight.data.numpy())
plot_embeddings_3d(pretrained_model.embedding.weight.data.numpy())
