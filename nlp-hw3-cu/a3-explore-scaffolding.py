'''
CSCI 5832 Assignment 3
Spring 2025
Use this code as a starting point or reference for the embedding
exploration portion of your assignment.
'''
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Get nearest neighbors using cosine similarity
def k_nearest_neighbors(embeddings, token2index, token, k: int = 5):
    # your code here
    # Add error handling
    if token not in token2index:
        raise ValueError(f"Token '{token}' not found in vocabulary")
    if k >= len(embeddings):
        raise ValueError(f"k ({k}) must be less than vocabulary size ({len(embeddings)})")
        
    token_index = token2index[token]
    token_embedding = embeddings[token_index]
    similarities = cosine_similarity([token_embedding], embeddings)[0]
    nearest_neighbors = np.argsort(similarities)[-k-1:][::-1]
    top_indices = [index for index in nearest_neighbors if index != token_index][:k]
    index2token = {index: token for token, index in token2index.items()}
    neighbors = [index2token[index] for index in top_indices]
    return neighbors

# Get the nearest neighbors of the word 'good'
k_nearest_neighbors(embeddings, token2index, 'good', k=10)

# The plot_embeddings functions are provided for you, but 
# you can play around with the details if you want to.
# for instance, a larger figure size, different colors, etc.

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
    plt.title("3D PCA Projection of Word Embeddings")
    plt.show()

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
    plt.title("t-SNE Projection of Word Embeddings")
    plt.show()

# compare custom vs GloVe embeddings
words_to_analyze = ['good', 'bad', 'excellent', 'terrible', 'movie']

print("Custom Embeddings Analysis:")
for word in words_to_analyze:
    neighbors = k_nearest_neighbors(model.embedding.weight.data, token2index, word, k=5)
    print(f"Nearest neighbors for '{word}': {neighbors}")

print("\nGloVe Embeddings Analysis:")
for word in words_to_analyze:
    neighbors = k_nearest_neighbors(pretrained_model.embedding.weight.data, token2index, word, k=5)
    print(f"Nearest neighbors for '{word}': {neighbors}")

# Visualize both custom and GloVe embeddings
print("Visualizing Custom Embeddings:")
plot_embeddings_tsne(model.embedding.weight.data.numpy())
plot_embeddings_3d(model.embedding.weight.data.numpy())

print("\nVisualizing GloVe Embeddings:")
plot_embeddings_tsne(pretrained_model.embedding.weight.data.numpy())
plot_embeddings_3d(pretrained_model.embedding.weight.data.numpy())