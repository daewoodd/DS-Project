import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load test data
X_test = pd.read_csv("models/X_test_scaled.csv")
y_test = pd.read_csv("models/y_test.csv").values.ravel()

def get_projection(method: str = "pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    else:
        raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

    reduced = reducer.fit_transform(X_test)
    projection = [{"x": float(x), "y": float(y), "label": label} 
                  for (x, y), label in zip(reduced, y_test)]

    return projection
