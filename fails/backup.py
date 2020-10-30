import numpy as np
import pandas as pd
import sys
import math
import matplotlib
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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
    
    def predict(i, theta1, theta2):
        X = data[i, :]
        X = np.transpose(X)
    #    X = np.hstack((1,X))
        a1 = X
        z2 = np.dot(theta1.T, a1)
        a2 = sigmoid(z2)
     #   a2 = np.hstack((1, a2))
        z3 = np.dot(theta2.T,a2)
        a3 = softmax(z3)
        
        return a3
    
    def compare(prediction, y):
        predicted_value = np.argmax(prediction)
        target_value = np.argmax(y)
        
        if predicted_value == target_value:
            return 1
        else:
            return 0
        
    
    start_time = time.time()
    DATA_PATH = 'fashion-1.csv'
    data = pd.read_csv(DATA_PATH)
    data_aux = data

  #  data_aux = data_aux[data_aux.label <= 1]

   # data = data_aux
    data = data.sample(frac=1) #con esto se desordenan las filas de datos
    n = data.shape[0]
    training_data, test_data = train_test_split(data, test_size = 0.3)
    
    labels = training_data['label']

    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    training_data.drop(['label'], axis = 1)
    training_data = preprocessing.normalize(data)
    training_data = np.array(data)
    theta1 = np.random.rand(785, 20)/785
    theta2 = np.random.rand(20, 10)
    aux = np.mean(theta2)
    aux2 = np.mean(theta1)
    print("theta2: " + str(aux))
    print("Theta1: " + str(aux2))
    avg_error = 0

    for j in range(500):
        J = 0
        print("Epoch number: " + str(j))
        for i in range(5000):
            n = random.randint(0, int(data.size/785)-1)
            X = data[n,:] #input de la fila 1
            X = np.transpose(X)
        #    X = np.hstack((1,X))
            y = labels[n]
            y = np.array(y)
            y = np.transpose(y)
            a1 = X
            z2 = np.dot(theta1.T, a1)
         #   z2 = np.hstack((1,z2))
            a2 = sigmoid(z2)
            z3 = np.dot(theta2.T,a2)
            a3 = softmax(z3)
        
            J+= mean_squared_error(a3,y)
    
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
        avg_error = J/5000
        print("Avg error: " + str(avg_error))


        
    aux = np.mean(theta2)
    aux2 = np.mean(theta1)
    print("")
    print("")
    print("theta2: " + str(aux))
    print("Theta1: " + str(aux2))
    #test
    accuracy = 0
    for i in range(500):
        prediction = predict(i, theta1, theta2)
        accuracy+= compare(prediction, labels[i])

    accuracy = accuracy/500
    print("Accuracy: " + str(accuracy))
    print("--- %s seconds to build and train ---" % (time.time() - start_time))
        

    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    