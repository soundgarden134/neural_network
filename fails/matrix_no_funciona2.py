import numpy as np
import pandas as pd
import sys
import math
import matplotlib
import random
from sklearn.utils import shuffle
from sklearn import preprocessing
import time
        
        
if __name__ == "__main__":  

    def sigmoid(matrix):
        a = []
    #    print(matrix)
        if matrix.size > 1:
            for value in matrix:
            
                a.append(1/(1+math.exp(-value)))
        
            a = np.array(a)
            return a
        else:
            return (1/(1+math.exp(-matrix)))
        
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    
    def derivative(matrix):
        a = []
        for value in matrix:
            der = sigmoid(value)*(1-sigmoid(value))
            a.append(der)
        a = np.array(a)
        return a
    
    
    def mean_squared_error(a, b):
        J = 0
        for i in range(len(a)):
            J += (a[i]-b[i])**2
        
        return J
    
    start = time.time()
    DATA_PATH = 'seeds_dataset.csv'
    data = pd.read_csv(DATA_PATH)
    data = shuffle(data)
    labels = data['y']
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    data.drop(['y'], axis = 1)
    data = preprocessing.normalize(data)
    data = np.array(data)
    theta1 = np.random.rand(8, 6)
    theta2 = np.random.rand(6, 3)
    aux = np.mean(theta2)
    aux2 = np.mean(theta1)
    print("theta2: " + str(aux))
    print("Theta1: " + str(aux2))
    for j in range(40):
        i=0
        for i in range(200):
            X = data[i,:] #input de la fila 1
            X = np.transpose(X)
        #    X = np.hstack((1,X))
            y = labels[i]
            y = np.array(y)
            y = np.transpose(y)
            a1 = X
            z2 = np.dot(theta1.T, a1)
         #   z2 = np.hstack((1,z2))
            a2 = sigmoid(z2)
            z3 = np.dot(theta2.T,a2)
            a3 = softmax(z3)
        
            J = mean_squared_error(a3, y)
            if i % 1000 == 0:
          #      print(a3)
                print(J)
            #    print(y)
            
    
            delta2 = (a3-y)
            a1 = np.array([a1])
            a2 = np.array([a2])
            der2 = delta2*a2.T
            delta1 = np.dot(theta2,delta2)*derivative(z2)
            delta1 = np.array([delta1])
            delta1 = np.transpose(delta1)
            der1 = a1.T*delta1.T
            theta2 -= 0.01*theta2 * der2
            theta1 -= 0.01*theta1 * der1

        
    aux = np.mean(theta2)
    aux2 = np.mean(theta1)
    print("")
    print("")
    print("theta2: " + str(aux))
    print("Theta1: " + str(aux2))
    #test
    X = data[100, :]
    X = np.transpose(X)
#    X = np.hstack((1,X))
    a1 = X
    z2 = np.dot(theta1.T, a1)
    a2 = sigmoid(z2)
 #   a2 = np.hstack((1, a2))
    z3 = np.dot(theta2.T,a2)
    a3 = softmax(z3)
    
    print(a3)
    print(labels[100])
    print(mean_squared_error(a3, labels[100]))
    end = time.time()
  #  print(end - start)   
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    