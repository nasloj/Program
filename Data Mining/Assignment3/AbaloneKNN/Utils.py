#-----------Utils.py------------- 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import * 
 
def get_dataset(): 
    url = ( 
        "https://archive.ics.uci.edu/ml/machine-learning-databases" 
        "/abalone/abalone.data" 
    ) 
    abalone = pd.read_csv(url, header=None) 
    abalone.columns = [ 
        "Sex", 
        "Length", 
        "Diameter", 
        "Height", 
        "Whole weight", 
        "Shucked weight", 
        "Viscera weight", 
        "Shell weight", 
        "Rings", 
    ] 
    return abalone 
2 
 
 
 
def get_train_test_data(X, y): 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=12345) 
    return X_train, X_test, y_train, y_test 
 
def plot_predicted_vs_actual(ypred, y): 
    mean_error = sum(abs(ypred-y))/len(y) 
    step_size = 20 
    a = [ypred[i] for i in range(0,len(ypred)) if i%step_size==0] 
    b = [y[i] for i in range(0,len(ypred)) if i%step_size==0] 
    t = linspace(0, len(a), len(a)) 
 
    plt.plot(t, a, 'red',linestyle='dashed', label='predicted')  
    plt.plot(t, b, 'blue',label='actual')  
    plt.scatter(t, a,marker='o', s=10, color="red", label="predicted") 
    plt.scatter(t, b, s=10, color="blue", label="actual") 
    plt.legend() 
    plt.title('mean error ='+str(mean_error)) #title 
    plt.show() 