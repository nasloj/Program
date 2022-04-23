#-----------GMMCars.py---------------------
import sys
import pandas as pd
import math
import numpy as np
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.mixture import GaussianMixture

def compute_gaussian_probab(x, mu, cov, d): # d is the number of dimensions in data xmu = np.matrix(x - mu)
    exponent = np.exp(-0.5 * xmu*np.linalg.inv(cov)*xmu.T)
    denom = 1 / np.sqrt(((2*np.pi)**d) * np.linalg.det(cov))
    res = denom * exponent
        #distribution = multivariate_normal(mu, cov)
        #res2 = distribution.pdf(x)
    return res
def E_Step(xdata,mu,cov,d, kc, phi, gamma):
    #-----------E step-----------------
    N = xdata.shape[0]
    for i in range(0,N):
        denom = 0
        for j in range(0,kc):
            denom = denom + (phi[j] * compute_gaussian_probab(xdata[i,:],mu[j],cov[j],d)) 
        for k in range(0,kc):
            gamma[i,k] = (phi[k] *
compute_gaussian_probab(xdata[i,:],mu[k],cov[k],d))/denom
def M_Step(xdata, mu, cov, d, kc, phi, gamma):
    #-----------M step-----------------
    N = xdata.shape[0]
    phi = np.mean(gamma,axis=0)
    sumgk = np.sum(gamma, axis=0)
    for k in range(0,kc):

        mu[k] = np.zeros((d))
        for i in range(0,N):
            mu[k] = mu[k] + (xdata[i,:] * gamma[i,k])
        mu[k] = mu[k]/sumgk[k]
    for k in range(0,kc):
        cov[k] = np.zeros((d,d))
        for i in range(0,N):
            xmu = np.matrix(xdata[i,:] - mu[k])
            cov[k] = cov[k] + ((xmu.T*xmu) * gamma[i,k])
        cov[k] = cov[k]/sumgk[k]
def plot_cars(xdata, clusters):
    plt.figure(figsize=(10, 6))
    plt.title('GMM Clusters')
    plt.scatter( # since we
       xdata[:,0],
       xdata[:,4],
        c=clusters,
        cmap=plt.cm.get_cmap('brg'),
        marker='.')
    plt.tight_layout()
    plt.show()
def main():
    #---------------car Dataset-------------
    df = pd.read_csv("/Users/naseemdavis/Downloads/cardata.csv")
    df.columns=['buying','maint','doors','persons','lug_boot','safety','class']
    print(df.head)
    df = df.replace(['low'],0)
    df = df.replace(['med'],1)
    df = df.replace(['high'],2)
    df = df.replace(['vhigh'],3)
    df = df.replace(['small'],0)
    df = df.replace(['big'],2)
    df = df.replace(['unacc'],0)
    df = df.replace(['acc'],1)
    df = df.replace(['good'],2)
    df = df.replace(['vgood'],3)
    df = df.replace(['more'],5)
    df = df.replace(['5more'],5)
    print(df.head)
    df = df.dropna()
    print(df.describe())
    df1 = df.iloc[:,0:6].astype(float)
    print(df1.info())
    #---separate out the last column
    df2 = df.iloc[:,6]
    xdata = df1.values
    xdata = xdata[:,0:6]
    ydata = df2.values
    print(xdata.shape)
    #--------------GMM N-D Algorithm-----------------
    kc = 4  # cluster count
    d = 6   # dimensionality of data
    N = xdata.shape[0]


    np.random.seed(43)
    mu = np.zeros((kc,d))
    cov = np.zeros((kc,d,d))
    #-----------initialization step------------
    phi = np.full(kc,1/kc)
    random_row = np.random.randint(low=0, high=N-1, size=kc)
    mu = np.array([xdata[row,:] for row in random_row ])
    mean_data = np.mean(xdata,axis=0) # mean of entire data (column wise) xmu = np.matrix(xdata-mean_data)
    cov = np.array([xmu.T*xmu/N for k in range(kc)])
    print(phi)
    print(mu)
    #print(cov)
    N = len(dfrandom)
    gamma = np.zeros((N,kc))
    print(gamma.shape)
    num_iterations = 10
    for n in range(0,num_iterations):
        E_Step(xdata, mu, cov, d, kc, phi, gamma)
        M_Step(xdata, mu, cov, d, kc, phi, gamma)
        print('----------------iteration =',n)
    #---------final result--------
    print(mu)
    print(gamma)
    print(phi)
    #-----compute accuracy GMM-ND--------
    # uncomment following to use your own implementation
    #preds = []
    #for i in range(0,N):
    #    pred = np.argmax(np.multiply(gamma[i,:],phi))
    #    preds.append(pred)
    #print(preds)
    #---------GMM from sklearn library---------
    gm = GaussianMixture(n_components=4, max_iter=50)
    gm.fit(xdata)
    preds = gm.predict(xdata)
    plot_cars(xdata, preds)
    #----since GMM is unsupervised and assigns clusters on its own, we need # to determine which cluster number is assigned to which class
    # One way is to determine the mode of the class in each cluster
    acc = 0
    class0_clusters = []
    class0_cluster_num = 0
    class1_clusters = []
    class1_cluster_num = 0
    class2_clusters = []
    class2_cluster_num = 0
    class3_clusters = []
    class3_cluster_num = 0
    for i in range(0, len(preds)):


        if ydata[i] == 0:
            class0_clusters.append(preds[i])
        if ydata[i] == 1:
            class1_clusters.append(preds[i])
        if ydata[i] == 2:
            class2_clusters.append(preds[i])
        if ydata[i] == 3:
            class3_clusters.append(preds[i])
    class0_cluster_num = mode(class0_clusters)
        # remove cluster num for class 0 before taking mode for class 1 cluster
    class1_clusters = [x for x in class1_clusters if x != class0_cluster_num[0]] 
    class1_cluster_num = mode(class1_clusters)
    # remove cluster num for class 0,1 before taking mode for class 2 cluster 
    class2_clusters = [x for x in class2_clusters if x != class0_cluster_num[0] and x!= class1_cluster_num[0]]
    class2_cluster_num = mode(class2_clusters)
        # remove cluster num for class 0,1,2 before taking mode for class 3 cluster
    class3_clusters = [x for x in class3_clusters if x != class0_cluster_num[0] and x!= class1_cluster_num[0] and x!= class2_cluster_num[0]]
    class3_cluster_num = mode(class3_clusters)
    print('-------cluster assignments--------')
    print(class0_cluster_num, ' ',class1_cluster_num, ' ', class2_cluster_num, ' ',
    class3_cluster_num)
    acc = 0
    for i in range(0, len(preds)):
        if ydata[i] == 0 and preds[i] == class0_cluster_num[0]:
            acc = acc + 1
        if ydata[i] == 1 and preds[i] == class1_cluster_num[0]:
            acc = acc + 1
        if ydata[i] == 2 and preds[i] == class2_cluster_num[0]:
            acc = acc + 1
        if ydata[i] == 3 and preds[i] == class3_cluster_num[0]:
            acc = acc + 1
    print('accuracy = ',acc/len(preds)*100)

if __name__ == "__main__":
    sys.exit(int(main() or 0))