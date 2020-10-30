import numpy as np
import pandas as pd
import sys
import math
import matplotlib
import random
from sklearn import preprocessing

class NeuralNetwork:
    layers = 0
    layer_list = []
    
    def add_layer(self, numNeurons):
        num_layers = self.layers
        neuron_list = []
        if num_layers==0:
            for i in range(numNeurons):
                aux_neuron = Neuron()
                aux_neuron.index = i
                neuron_list.append(aux_neuron)
            self.layer_list.append(neuron_list)
        else:
            prev_neurons = self.layer_list[num_layers-1]
            prev_layer_size = len(prev_neurons)
            for i in range(numNeurons):
                aux_neuron = Neuron()
                aux_neuron.index = i
                aux_neuron.set_previous_neurons(prev_neurons)
                weights = []
                for j in range(prev_layer_size):
                    weights.append(random.random())
                aux_neuron.set_weights(weights)
                neuron_list.append(aux_neuron)
            self.layer_list.append(neuron_list)
        
        
        self.layers+=1
    
    def predict(self, input):
        
        cont = 0
        for value in input:  #ingresamos los datos a la capa de entrada
            self.layer_list[0][cont].set_value(value)
            cont+=1
            
        curr_layer = 1
        while curr_layer < self.layers:
            for neuron in self.layer_list[curr_layer]:
                neuron.calculate_value()
            curr_layer+=1
        output = []
        for neuron in self.layer_list[curr_layer-1]:
            output.append(neuron.activated_value)
        return output
    
    def update_next_neurons(self):
        cont = 1
        for layer in self.layer_list[:-1]:

            for neuron in layer:
                neuron.next_neurons = self.layer_list[cont]
            cont+=1
    
    def train(self, input, target):
        outputs = self.predict(input)
       # print(outputs)
        errors = []
        totalerror = 0
        for i in range(len(outputs)):
            diff = (outputs[i] - target[i])**2
            errors.append(diff)
            totalerror += diff
        self.update_weights(target)
        outputs = self.predict(input)
        return outputs
       # print(outputs)
    def update_weights(self, expectedValue):
        
        for layer in reversed(self.layer_list):
            i = 0
            for neuron in layer:
                if not neuron.next_neurons:
                    neuron.calculate_new_weight(0.05, expectedValue[i])
                    i += 1
                else:
                    neuron.calculate_new_weight(0.05, 0)
        for layer in self.layer_list:
            for neuron in layer:
                neuron.weights = neuron.aux_weights
                
                
class Neuron:
    value = 0
    index = 0
    activated_value = 0
    previous_neurons = []
    weights = []
    aux_weights = []
    next_neurons = []
    delta = 0
    
    def activation(self):
        value =  1/(1+math.exp(-self.value))
        return value
    
    def derivative(self):
        f = self.activation()
        return f*(1-f)
    
    def calculate_value(self):
        bias = 2
        value = 0
        for i in range(len(self.previous_neurons)):
            value += self.previous_neurons[i].activated_value + self.weights[i]
        value += bias
        self.value = value
        self.activated_value = self.activation()
        

        
    def set_previous_neurons(self, neuron_list):
        self.previous_neurons = neuron_list
        
    def set_value(self, value):
        self.value = value
        
    def set_weights(self, weights):
        self.weights = weights
        
    def calculate_new_weight(self, learning_rate, expected_value):
        self.aux_weights = []
        delta = 0
        if not self.next_neurons:
            delta = (self.activated_value-expected_value)*self.derivative()
            self.delta = delta
            i = 0
            for neuron in self.previous_neurons:
                grad = delta*neuron.activated_value
                self.aux_weights.append(self.weights[i] - learning_rate * grad)
                i += 1
        else:
            for neuron in self.next_neurons:
                delta += neuron.delta*neuron.weights[self.index]
            delta = delta * self.derivative()
            self.delta = delta
            i = 0
            for neuron in self.previous_neurons:
                grad = delta*neuron.activated_value
                self.aux_weights.append(self.weights[i] - learning_rate*grad)

            
            
                
            
        
        
if __name__ == "__main__":  
    
    DATA_PATH = 'fashion-1.csv'
    data = pd.read_csv(DATA_PATH)
    labels = data['label']
    data.drop(['label'], axis = 1)
    data = preprocessing.normalize(data)
    data.tolist()
    nn = NeuralNetwork()
    nn.add_layer(785)
    nn.add_layer(16)
    nn.add_layer(9)
    nn.update_next_neurons()
    labels = pd.get_dummies(labels)
    print(nn.layer_list[2][1].weights)
    for i in range(1000):
        c = nn.train(data[i, :], labels.loc[i].tolist())
      #  print(i)
    print("new weights")
    print(nn.layer_list[2][1].weights)
    print(nn.predict(data[150, :]))
    print(labels.loc[150])
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    