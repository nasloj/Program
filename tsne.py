import sys
# ------------------tsne.py----------------
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns
import Utils


def main():


    # Load the MNIST data
    # X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame = False)

    X, y = Utils.read_data()  # cancer data

    # randomly select 800 samples from dataset
    np.random.seed(100)
    subsample_idc = np.random.choice(X.shape[0], 800, replace=False)
    X = X[subsample_idc, :]
    y = y[subsample_idc]
    y = np.array([int(lbl) for lbl in y])

    n_components = 2
    tsne = TSNE(n_components, perplexity=30)
    tsne_result = tsne.fit_transform(X)
    print(tsne_result.shape)

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, c=y, cmap = 'Spectral')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))