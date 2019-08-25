import numpy as np
import math
from statistics import mean 

class NeuralNetwork():
    
    def __init__(self, X = None , y = None, hiddenLayers = 2, neuronsEachLayer = 2, learning_rate = 0.01, 
                epochs = 5, method = 'Linear', tol = 0.1, batch_size = 32):
        self.weights = None
        self.activationHidden = self.sigmoid
        if method == 'Linear':
            self.activationOut = self.linear
            self.derivate_out = self.linear_der
        elif method == 'Logistic':
            self.activationOut = self.sigmoid
            self.derivate_out = self.sigmoid_der
        self.derivate_rest = self.sigmoid_der
        self.X = X
        self.Y = y
        self.hiddenLayers = hiddenLayers
        self.neuronsEachLayer = neuronsEachLayer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.batch_size = batch_size

    def weightsInitialisation(self):
        #Initialising a numpy array of dim(hiddenlayers, neurons) to store weights
        self.weights = np.empty((self.hiddenLayers, self.neuronsEachLayer), dtype = object)
        
        for i in range(self.hiddenLayers):
            for j in range(self.neuronsEachLayer):
                #first hidden layer
                if i == 0:
                    self.weights[i,j] = np.random.normal(0,0.4, size = 1 + self.X.shape[1])
                #rest hidden layers
                else:
                    self.weights[i,j] = np.random.normal(0,0.4, size = 1 + self.neuronsEachLayer)
        #Weights for the final output layer
        self.outputLayerWeights =  np.random.normal(0,0.4, size = 1 + self.neuronsEachLayer)
    
    def gradientInitialisation(self):
        self.gradient = np.empty((self.hiddenLayers, self.neuronsEachLayer), dtype = object)
        for i in range(self.hiddenLayers):
            for j in range(self.neuronsEachLayer):
                #first hidden layer
                if i == 0:
                    self.gradient[i,j] = np.zeros(1 + self.X.shape[1])
                #rest hidden layers
                else:
                    self.gradient[i,j] =  np.zeros(1 + self.neuronsEachLayer)
        self.gradientOutputLayer = [0] * len(self.outputLayerWeights)
    
    def sigmoid(self,x):
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))
    
    def linear(self,x):
        return x
    
    def sigmoid_der(self,x):  
        return self.sigmoid(x) *(1 - self.sigmoid(x))

    def linear_der(self, x):
        return 1.0
    
    def squareErrorLoss(self,x,y):
        return (self.feedForward(x) - y)**2
    
    def error(self, X, y):
        pred= []
        for i in X:
            pred.append(self.feedForward(i))
        return mean([(a_i - b_i)**2 for a_i, b_i in zip(pred, y)])
    
    def predict(self,X):
        pred = []
        for i in X:
            pred.append(self.feedForward(i))
        return pred

    def feedForward(self, x):
        self.x = np.append(x, 1.0)
        self.out = np.empty(shape = (self.hiddenLayers, self.neuronsEachLayer))
        for i in range(self.hiddenLayers + 1):
            outputFromCurrLayer = []
            #For first Layer
            if i == 0:
                for j in range(self.neuronsEachLayer):
                    z = self.activationHidden(np.dot(self.weights[i,j],self.x))
                    self.out[i,j] = z
                    outputFromCurrLayer.append(z)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()
            #Output Layer
            elif i == self.hiddenLayers:
                return self.activationOut(np.dot(self.outputLayerWeights,outputFromPrevLayer))
            #Rest all Layers
            else:
                for j in range(self.neuronsEachLayer):
                    z = self.activationHidden(np.dot(self.weights[i,j],outputFromPrevLayer))
                    self.out[i,j] = z
                    outputFromCurrLayer.append(z)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()

    def backProp(self, pred, n, actual):
        #Weight updation for Output Layer
        delta = []      
        der_outter_layer = self.derivate_out(np.dot(np.append(self.out[self.hiddenLayers -1], 1.0) , self.outputLayerWeights))
        for i in range(len(self.outputLayerWeights)):
            if i == len(self.outputLayerWeights) - 1:
                self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + ((2.0 / n) * (pred- actual) * der_outter_layer * 1)
            else :
                d = (2.0 / n) * (pred- actual) * der_outter_layer * self.outputLayerWeights[i]
                self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + ((2.0 / n) * (pred- actual) * der_outter_layer * self.out[self.hiddenLayers -1, i])
                delta.append(d) 
        #For all other Layers
        for l in reversed(range(self.hiddenLayers)):
            delta_forward = delta.copy()
            delta = [0] * self.neuronsEachLayer
            #For the first layer
            if l == 0 :
                for j in range(self.neuronsEachLayer):
                    der_layer = self.derivate_rest(np.dot(self.x , self.weights[l,j]))
                    for i in range(len(self.weights[l,j])):
                        if i == len(self.weights[l,j]) - 1:
                            self.gradient[l,j][i] = self.gradient[l,j][i] + ((1.0 / n) * delta_forward[j] * der_layer * 1.0)
                        else :
                            self.gradient[l,j][i] = self.gradient[l,j][i] +  ((1.0 / n) * delta_forward[j] * der_layer * self.x[i])
            #Rest all the layers
            else :
                for j in range(self.neuronsEachLayer):
                    der_layer = self.derivate_rest(np.dot(np.append(self.out[l - 1], 1.0) , self.weights[l,j]))
                    for i in range(len(self.weights[l,j])):
                        if i == len(self.weights[l,j]) - 1:
                            self.gradient[l,j][i] = self.gradient[l,j][i] + ((1.0 / n) * delta_forward[j] * der_layer * 1.0)
                        else :
                            d = (1.0 / n) * delta_forward[j] * der_layer * self.weights[l,j][i]
                            delta[i] = delta[i] + d
                            self.gradient[l,j][i] = self.gradient[l,j][i] + (delta_forward[j] * der_layer * self.out[l - 1, i])
    
    def updateWeights(self, n):
        for i in range(len(self.outputLayerWeights)):
                self.outputLayerWeights[i] = self.outputLayerWeights[i] - (self.learning_rate * (1.0 / n) * self.gradientOutputLayer[i])
        #For all other Layers
        for l in reversed(range(self.hiddenLayers)):
                for j in range(self.neuronsEachLayer):
                    for i in range(len(self.weights[l,j])):
                        self.weights[l,j][i] = self.weights[l,j][i] - (self.learning_rate * (1.0 / n) * self.gradient[l,j][i])

    
    def fit(self,X,y,X_val = None, Y_val = None):
        self.X = X
        self.y = y
        self.weightsInitialisation()
        self.gradientInitialisation()
        i = 0
        error_val_old = -1
        tol_count = 0
        while i < self.epochs:
            for j in range(len(X)):
                if j%self.batch_size ==0 and j != 0 or j == len(X) -1:
                    if j == len(X) -1:
                        self.updateWeights(j%self.batch_size)
                    else:
                        self.updateWeights(self.batch_size)
                    self.gradientInitialisation()
                    p = self.feedForward(X[j])
                    self.backProp(p,1,y[j])
                else:
                    p = self.feedForward(X[j])
                    self.backProp(p,1,y[j])
            if X_val is not None and Y_val is not None:
                error_curr_val = self.error(X_val, Y_val)
                print("Epoch : {} and MSE_Train : {} and MSE_Val : {}".format(i, self.error(X,y), error_curr_val))
                if abs(error_val_old -error_curr_val) < self.tol :
                    tol_count = tol_count + 1
                    error_val_old = error_curr_val
                    if tol_count >1 :
                        print("Stopping as validation error did not improve more than tol = {} for 2 iterations".format(self.tol))
                        break
                else:
                    tol_count = 0
                    error_val_old = error_curr_val
            else:
                print("Epoch : {} and MSE : {}".format(i, self.error(X,y)))
            i = i+1