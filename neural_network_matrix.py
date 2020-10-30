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

    def sigmoid(matrix): #sigmoide en la primera capa
        a = []
    #    print(matrix)
        if matrix.size > 1:
            for value in matrix:
            
                a.append(1/(1+math.exp(-value)))
        
            a = np.array(a)
            return a
        else:
            return (1/(1+math.exp(-matrix)))
        
    def softmax(x):  #activacion de la ultima capa, esta funcion hace que la suma de todos los output sea 1
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    
    def derivative(matrix): #derivada de sigmoide
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
    
    def predict(X, theta1, theta2):
        X = np.transpose(X)
    #    X = np.hstack((1,X))
        a1 = X
        z2 = np.dot(theta1.T, a1)
        a2 = sigmoid(z2)
     #   a2 = np.hstack((1, a2))
        z3 = np.dot(theta2.T,a2)
        a3 = softmax(z3)
        
        return a3
    
    def compare(prediction, y):   #arroja 1 si la prediccion es correcta (se queda con el valor mayor del array de prediccion)
        predicted_value = np.argmax(prediction)
        target_value = np.argmax(y)
        
        if predicted_value == target_value:
            return 1
        else:
            return 0
        
    
    start_time = time.time()
    DATA_PATH = 'fashion-1.csv'
    data = pd.read_csv(DATA_PATH)
   
    data = data.sample(frac=1) #con esto se desordenan las filas de datos
    n = data.shape[0]
    data_labels = data['label']
    data_to_normalize = data.drop('label', axis = 1) #normalizamos todo menos el output
    data_to_normalize = preprocessing.normalize(data_to_normalize) 
    
    normalized_data = np.column_stack((data_to_normalize, data_labels))
    training_data, test_data = train_test_split(normalized_data, test_size = 0.3)
    
    labels = training_data[:,784] #se selecciona la ultima columna que ahora contiene la etiqueta
    labels = pd.get_dummies(labels)  #one hot encoder para las etiquetas
    labels = labels.to_numpy()
    training_data = np.delete(training_data, 784, axis = 1)


    #se hace lo mismo para el test set
    test_labels = test_data[:,784]
    test_labels = pd.get_dummies(test_labels)
    test_labels = test_labels.to_numpy()
    test_data = np.delete(test_data, 784, axis = 1)
    test_data = np.array(test_data)
    
    #inicializamos los pesos
    np.random.seed(42) #para ser consistentes con los random
    theta1 = np.random.rand(784, 20)/784
    theta2 = np.random.rand(20, 10)
    aux = np.mean(theta2) 
    aux2 = np.mean(theta1)
    #Se printean el valor promedio de los pesos de cada capa
    print("Theta2 inicial: " + str(aux))
    print("Theta1 inicial: " + str(aux2))
    avg_error = 0

    for j in range(500):  #numero de epochs
        J = 0
        print("Epoch number: " + str(j))
        for i in range(5000): #5000 veces se elige una instancia aleatoria para entrenar
            n = random.randint(0, int(training_data.shape[0])-1)
            X = training_data[n,:] #input de la fila 1
            X = np.transpose(X)
        #    X = np.hstack((1,X)) #con bias no funciono bien, por aplicar mal el gradiente del bias creo
            y = labels[n]
            y = np.array(y)
            y = np.transpose(y)
            a1 = X
            z2 = np.dot(theta1.T, a1)
         #   z2 = np.hstack((1,z2))
            a2 = sigmoid(z2)
            z3 = np.dot(theta2.T,a2)
            a3 = softmax(z3)
        
            J+= mean_squared_error(a3,y)  #se calcula el promedio de error cuadratico medio de cada instancia
            
            #BACKPROPAGATION
            
            delta2 = (a3-y)
            a1 = np.array([a1])
            a2 = np.array([a2])
            der2 = delta2*a2.T     #derivada 
            delta1 = np.dot(theta2,delta2)*derivative(z2)
            delta1 = np.array([delta1])  #se tiene que transformar en un array 2d para usar transpose
            delta1 = np.transpose(delta1)
            der1 = a1.T*delta1.T
            theta2 -= 0.01*theta2 * der2
            theta1 -= 0.01*theta1 * der1
            
        avg_error = J/5000
        print("Avg error: " + str(avg_error)) #Error promedio de una epoch

    
        
    aux = np.mean(theta2)
    aux2 = np.mean(theta1)
    print("")
    print("")
    #valores finales de los pesos
    print("Theta2 final: " + str(aux))
    print("Theta1 final: " + str(aux2))
    #test
    accuracy = 0
    #se prueba con la data para el testeo
    
    for i in range(len(test_data)):
        prediction = predict(test_data[i], theta1, theta2)
        accuracy+= compare(prediction, test_labels[i]) #1 si es correcta, 0 si es incorrecta

    accuracy = accuracy/len(test_data) #promedio de precision
    print("Accuracy: " + str(accuracy))
    print("--- %s seconds to build and train ---" % (time.time() - start_time))
        

    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    