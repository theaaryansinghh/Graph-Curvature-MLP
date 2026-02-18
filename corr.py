import numpy as np

X = np.load("data/X_train.npy")
y = np.load("data/y_train.npy").flatten()

for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    print(f"Feature {i} correlation with delta: {corr}")
