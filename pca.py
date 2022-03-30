import sys
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import Utils


def main():


    # Load the MNIST data
    #X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame = False)
    X, y = Utils.read_data()  # cancer data

    # randomly select 800 samples from dataset
    np.random.seed(100)
    subsample_idc = np.random.choice(X.shape[0], 800, replace=False)
    X = X[subsample_idc, :]
    y = y[subsample_idc]
    y = np.array([int(lbl) for lbl in y])

    n_components = 2
    X = StandardScaler().fit_transform(X)  # subtract mean, divide by var
    pca = PCA(n_components)
    pca_result = pca.fit_transform(X)
    print(pca_result.shape)  # 2d for visualization

    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11) -
    0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))