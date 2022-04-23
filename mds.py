

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

def main():

    dataset = pd.read_csv('/Users/naseemdavis/Desktop/CPSC552 copy/TCGA-PANCAN-HiSeq-801x20531/data.csv')

    X = dataset.iloc[:, 1:31].values

    Y = dataset.iloc[:, 31].values

    dataset.head()

    print("Cancer data set dimensions : {}".format(dataset.shape))

    dataset.groupby('diagnosis').size()

    #Visualization of data

    dataset.groupby('diagnosis').hist(figsize=(12, 12))
    
    mds = MDS(n_components, perplexity=30)
    mds_result = mds.fit_transform(X)
    print(tsne_result.shape)

    plt.scatter(mds_result[:, 0], mds_result[:, 1], s=5, c=y, cmap = 'Spectral')
    
    plt.mds
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show()

   