import sys
import numpy as np
import pandas as pd
import math

def compute_gaussian_probab(x, mean, var):
    res = 1
    for i in range(0,len(x)):
        exponent = math.exp(-((x[i]-mean[i])**2 / (2 * var[i] ))) 
        res *= (1 / (math.sqrt(2 * math.pi * var[i]))) * exponent
    return res

def main():
    df = pd.read_csv("/Users/naseemdavis/Downloads/wheat-seeds.csv")

    dfrandom = df.sample(frac=1, random_state=1119).reset_index(drop=True) 
    
    df1 = dfrandom.iloc[:,0:7].astype(float)
    df2 = dfrandom.iloc[:,7]

    dfrandom = pd.concat([df1,df2],axis=1) 
    print(dfrandom)

    dftest = dfrandom.iloc[0:,:] 
    print(dftest)

    dfone = dfrandom[dfrandom['class'] == 1] 
    print(dfone)
    dftwo = dfrandom[dfrandom['class'] == 2] 
    print(dftwo)
    dfthree = dfrandom[dfrandom['class'] == 3] 
    print(dfthree)

    mean_one = dfone.iloc[:,0:7].mean(axis=0) 
    print('mean one\n',mean_one)
    mean_two = dftwo.iloc[:,0:7].mean(axis=0)
    print('mean two\n',mean_two) 
    mean_three = dfthree.iloc[:,0:7].mean(axis=0) 
    print('mean three\n',mean_three)

    var_one = dfone.iloc[:,0:7].var(axis=0) 
    print('var one\n',var_one)
    var_two = dftwo.iloc[:,0:7].var(axis=0) 
    print('var two\n',var_two) 
    var_three = dfthree.iloc[:,0:7].var(axis=0) 
    print('var three\n',var_three)

    count_correct = 0 
    print(len(dftest))

    for i in range(0,len(dftest)):
        x = dftest.iloc[i,0:7].values
        probC1 = compute_gaussian_probab(x,mean_one.values,var_one.values) 
        probC2 = compute_gaussian_probab(x,mean_two.values,var_two.values) 
        probC3 = compute_gaussian_probab(x,mean_three.values,var_three.values) 
        probs = np.array([probC1,probC2,probC3])
        maxindex = probs.argmax(axis=0)

        if dftest.iloc[i,7] == 1: 
            index = 0
        if dftest.iloc[i,7] == 2: 
            index = 1
        if dftest.iloc[i,7] == 3: 
            index = 2
        if maxindex == index:
            count_correct = count_correct + 1

    print('classification accuracy =', count_correct/len(dftest)*100) 

if __name__ == "__main__":
    sys.exit(int(main() or 0))