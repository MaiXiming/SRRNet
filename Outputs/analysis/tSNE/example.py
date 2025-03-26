import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize the data

# Use only a subset of the data for speed
indices = np.random.choice(range(X.shape[0]), size=3000, replace=False)
X_subset = X[indices]
y_subset = y[indices]

# Create the t-SNE transformation
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X_subset)

# Plot the result
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_subset.astype(int), cmap='jet', alpha=0.6)
plt.colorbar()
# plt.show()
plt.savefig('example.png')
