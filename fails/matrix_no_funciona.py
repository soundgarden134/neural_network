import numpy as np
import pandas as pd
import sys
import math
import matplotlib
import random
from sklearn import preprocessing
import time
        
        
if __name__ == "__main__":  

    def sigmoid(matrix):
        a = []
    #    print(matrix)
        for row in matrix:
            aux = []
            for column in row:
                aux.append(1/(1+math.exp(-column)))
            a.append(aux)
        
        a = np.array(a)
        return a
    
    def mean_squared_error(a, b):
        J = 0
        for i in range(len(a)):
            J += (a[i]-b[i])**2
        
        error = 0
        for value in J:
            error += value
        return error
    
    start = time.time()
    DATA_PATH = 'fashion-1.csv'
    data = pd.read_csv(DATA_PATH)
    labels = data['label']
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    data.drop(['label'], axis = 1)
    data = preprocessing.normalize(data)
    theta1 = np.random.rand(786, 16)
    theta2 = np.random.rand(17, 10)

    X = data[1:4,:] #input de la fila 1
    ones = np.array((1,1,1))
    X = np.transpose(X)
    X = np.vstack((ones,X))
    y = labels[1:4]
    y = np.array(y)
    y = np.transpose(y)
    a1 = X
    z2 = np.dot(theta1.T, a1)
    a2 = sigmoid(z2)
    a2 = np.vstack((ones, a2))
    z3 = np.dot(theta2.T,a2)
    a3 = sigmoid(z3)
    
    J = mean_squared_error(a3, y)
    
    end = time.time()
    print(end - start)   
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    