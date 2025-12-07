from sklearn import datasets, manifold
import numpy as np

print("Loading data...")
digits = datasets.load_digits()
print("Loading data - finished")

X, y = digits.data, digits.target
n_samples, n_features = X.shape

print("Computing 2D representation...")
data_2d = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
x_min, x_max = np.min(data_2d, axis=0), np.max(data_2d, axis=0)
data_2d_norm = (data_2d - x_min) / (x_max - x_min)
print("Computing 2D representation finished")

print("Saving data...")
np.savetxt("./data/in/digits/digits.csv", data_2d_norm, delimiter=",")
print("Saving data finished")