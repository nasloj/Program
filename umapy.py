import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap  # pip install umap-learn
from sklearn.datasets import fetch_openml

import Utils


def main():
    # Load the MNIST data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    # X, y = Utils.read_data()  # cancer data
    # randomly select 800 samples from dataset
    np.random.seed(100)
    subsample_idc = np.random.choice(X.shape[0], 800, replace=False)
    X = X[subsample_idc, :]
    y = y[subsample_idc]
    y = np.array([int(lbl) for lbl in y])

    n_components = 2
    ump = umap.umap(
        n_neighbors=20,
        min_dist=0.1,
        n_components=n_components,
        metric = 'euclidean')
    umap_result = ump.fit_transform(X)
    print(umap_result.shape)  # 2d for visualization

    plt.scatter(umap_result[:, 0], umap_result[:, 1], s=5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show()


if __name__ == "__main__":
    sys.exit(int(main() or 0))
