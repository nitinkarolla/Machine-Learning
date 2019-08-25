import numpy as np
import math
from statistics import mean 

class NeuralNetwork():
    
    def __init__(self, X = None , y = None, layers = [5, 2], learning_rate = 0.01, 
                epochs = 5, method = 'Linear', tol = 0.1, batch_size = 32):
        self.weights = None
        self.X = X
        self.y = y
        self.activationHidden = self.sigmoid
        self.method = method
        if self.method == 'Linear':
            self.activationOut = self.linear
            self.derivate_out = self.linear_der
        elif self.method == 'Classification' and len(np.unique(self.y)) == 2:
            self.out_class = 'Binary'
            self.activationOut = self.sigmoid
            self.derivate_out = self.sigmoid_der
        elif self.method == 'Classification' and len(np.unique(self.y)) > 2:
            self.out_class = 'MultiClass'
        self.layers = layers
            #self.activationOut = self.softmax
            #self.derivate_out = self.softmax_der
        self.derivate_rest = self.sigmoid_der
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.batch_size = batch_size
        

    def weightsInitialisation(self):
        #Initialising a numpy array of dim(hiddenlayers, neurons) to store weights
        self.weights = []
        for i in range(len(self.layers)):
            temp = []
            for j in range(self.layers[i]):
                #first hidden layer
                if i == 0:
                    temp.append(np.random.normal(0,0.4, size = 1 + self.X.shape[1]))
                #rest hidden layers
                else:
                    temp.append(np.random.normal(0,0.4, size = 1 + self.layers[i-1]))
            self.weights.append(temp)
        #Weights for the final output layer
        if self.out_class == 'MultiClass':
            self.outputLayerWeights =  np.random.normal(0,0.4, size = ( len(np.unique(self.y)), 1 + self.layers[-1]))
        else:
            self.outputLayerWeights =  np.random.normal(0,0.4, size = 1 + self.layers[-1])
    
    def gradientInitialisation(self):
        self.gradient = []
        for i in range(len(self.layers)):
            temp = []
            for j in range(self.layers[i]):
                #first hidden layer
                if i == 0:
                    temp.append(np.zeros(1 + self.X.shape[1]))
                #rest hidden layers
                else:
                    temp.append(np.zeros(1 + self.layers[i-1]))
            self.gradient.append(temp)
        if self.out_class == 'MultiClass':
            self.gradientOutputLayer = np.zeros(shape = (len(np.unique(self.y)), 1 + self.layers[-1]))
        else:
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
    
    def softmax(self,x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
    
    def squareErrorLoss(self,x,y):
        return (self.feedForward(x) - y)**2
    
    def error(self, X, y):
        if self.out_class == 'Linear':
            pred= []
            for i in X:
                pred.append(self.feedForward(i))
            return mean([(a_i - b_i)**2 for a_i, b_i in zip(pred, y)])
        elif self.out_class == 'Binary':
            error = 0
            for i in range(len(X)):
                prob = self.feedForward(X[i])
                if (prob <0.5 and y[i] == 1) or (prob >=0.5 and y[i] == 0):
                    error = error + 1
            return error/X.shape[0]
        elif self.out_class == 'MultiClass':
            error = 0
            for i in range(len(X)):
                prob = self.feedForward(X[i])
                class_pred = list(prob).index(max(prob))
                if class_pred != list(y[i]).index(1):
                    error = error + 1
            return error/X.shape[0]


    def predict(self,X):
        pred = []
        for i in X:
            pred.append(self.feedForward(i))
        return pred

    def predict_row(self,X):
        out = self.feedForward(X)
        if self.out_class == 'Linear':
            return out
        elif self.out_class == 'Binary':
            if out >= 0.5:
                return 1
            else:
                return 0
        elif self.out_class == 'MultiClass':
            return list(out).index(max(out))
    
    def loss(self, pred, actual):
        if self.method == 'Linear' or self.out_class == 'Binary':
            return 2.0 * (pred- actual)
        #elif self.out_class == 'Binary':
            #return 
        elif self.out_class == 'MultiClass':
            p = np.dot(pred,actual)
            return (-1/math.log(p))
        
    def softmax_der(self, pred, actual, l):
        if actual[l] == 1:
            return pred[l]*(1 - pred[l])
        else:
            i = list(actual).index(1)
            return -1*pred[l]*pred[i]

    def onehotencoding(self, y):
        out = np.zeros((len(y),len(np.unique(y))))
        for i in range(len(y)):
            out[i][y[i]] = 1
        return out

    def feedForward(self, x):
        self.x = np.append(x, 1.0)
        self.out = []
        for i in range(len(self.layers) + 1):
            outputFromCurrLayer = []
            #For first Layer
            if i == 0:
                for j in range(self.layers[i]):
                    z = self.activationHidden(np.dot(self.weights[i][j],self.x))
                    outputFromCurrLayer.append(z)
                temp = outputFromCurrLayer.copy()
                self.out.append(temp)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()
            #Output Layer
            elif i == len(self.layers) and self.out_class == 'MultiClass':
                return self.softmax(np.matmul(self.outputLayerWeights, outputFromPrevLayer))
            elif i == len(self.layers):
                return self.activationOut(np.dot(self.outputLayerWeights,outputFromPrevLayer))
            #Rest all Layers
            else:
                for j in range(self.layers[i]):
                    z = self.activationHidden(np.dot(self.weights[i][j],outputFromPrevLayer))
                    outputFromCurrLayer.append(z)
                temp = outputFromCurrLayer.copy()
                self.out.append(temp)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()

    def backProp(self, pred, actual):
        #Weight updation for Output Layer
        if self.out_class == 'Linear' or self.out_class == 'Binary':
            delta = []
            der_outter_layer = self.derivate_out(np.dot(np.append(self.out[len(self.layers) -1], 1.0) , self.outputLayerWeights))
            for i in range(len(self.outputLayerWeights)):
                if i == len(self.outputLayerWeights) - 1:
                    self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + (self.loss(pred, actual) * der_outter_layer * 1)
                else :
                    d = self.loss(pred, actual) * der_outter_layer * self.outputLayerWeights[i]
                    self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + (self.loss(pred, actual) * der_outter_layer * self.out[len(self.layers) -1][i])
                    delta.append(d) 
        elif self.out_class == 'MultiClass':
            delta = [0] * self.layers[-1]
            for l in range(len(self.outputLayerWeights)):
                der_outter_layer = self.softmax_der(pred,actual, l)
                for i in range(len(self.outputLayerWeights[l])):
                    if i == len(self.outputLayerWeights[l]) - 1:
                        self.gradientOutputLayer[l][i] = self.gradientOutputLayer[l][i] + (self.loss(pred, actual) * der_outter_layer * 1)
                    else:
                        d = self.loss(pred, actual) * der_outter_layer * self.outputLayerWeights[l][i]
                        delta[i] = delta[i] + d
                        self.gradientOutputLayer[l][i] = self.gradientOutputLayer[l][i] + (self.loss(pred, actual) * der_outter_layer * self.out[len(self.layers) - 1][i])       

        #For all other Layers
        for l in reversed(range(len(self.layers))):
            delta_forward = delta.copy()
            delta = [0] * self.layers[l-1]
            #For the first layer
            if l == 0 :
                for j in range(self.layers[l]):
                    der_layer = self.derivate_rest(np.dot(self.x , self.weights[l][j]))
                    for i in range(len(self.weights[l][j])):
                        if i == len(self.weights[l][j]) - 1:
                            self.gradient[l][j][i] = self.gradient[l][j][i] +  (delta_forward[j] * der_layer * 1.0)
                        else :
                            self.gradient[l][j][i] = self.gradient[l][j][i] +   (delta_forward[j] * der_layer * self.x[i])
            #Rest all the layers
            else :
                for j in range(self.layers[l]):
                    der_layer = self.derivate_rest(np.dot(np.append(self.out[l - 1], 1.0) , self.weights[l][j]))
                    for i in range(len(self.weights[l][j])):
                        if i == len(self.weights[l][j]) - 1:
                            self.gradient[l][j][i] = self.gradient[l][j][i] +  (delta_forward[j] * der_layer * 1.0)
                        else :
                            d = delta_forward[j] * der_layer * self.weights[l][j][i]
                            delta[i] = delta[i] + d
                            self.gradient[l][j][i] = self.gradient[l][j][i] + (delta_forward[j] * der_layer * self.out[l - 1][i])
    
    def updateWeights(self, n):
        if self.out_class == 'Linear' and self.out_class == 'Binary':
            for i in range(len(self.outputLayerWeights)):
                self.outputLayerWeights[i] = self.outputLayerWeights[i] - (self.learning_rate *  self.gradientOutputLayer[i]/n)
        elif self.out_class =='MultiClass':
            for l in range(len(self.outputLayerWeights)):
                for i in range(len(self.outputLayerWeights[l])):
                    self.outputLayerWeights[l][i] = self.outputLayerWeights[l][i] - (self.learning_rate *  self.gradientOutputLayer[l][i]/n)
        #For all other Layers
        for l in reversed(range(len(self.layers))):
            for j in range(self.layers[l]):
                for i in range(len(self.weights[l][j])):
                    self.weights[l][j][i] = self.weights[l][j][i] - (self.learning_rate *  self.gradient[l][j][i] /n)

    
    def fit(self,X,y,X_val = None, Y_val = None):
        self.X = X
        self.y = y
        if self.out_class == 'MultiClass':
            y = self.onehotencoding(y)
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
                    self.backProp(p,y[j])
                else:
                    p = self.feedForward(X[j])
                    self.backProp(p,y[j])
            #print(nn.weights)
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


### TESTING ####
X = np.random.normal(loc = 0, scale = 1, size = (1000,3))
y = np.random.randint(2, size=1000)
nn= NeuralNetwork(X = X, y = y, epochs= 1000, method= 'Classification', layers = [5,5], learning_rate= 0.003 )
nn.fit(X,y)
# nn.weightsInitialisation()
#nn.gradientInitialisation()
# p = nn.feedForward(X[0])
# print(p)
#print(nn.weights)
#print(nn.outputLayerWeights)
# #print(nn.x)
# nn.gradientInitialisation()
#print(nn.gradient)
#print(nn.gradientOutputLayer)
# nn.backProp(pred= p, actual= [1.0, 0 , 0, 0, 0])
# print(nn.weights)
# print(nn.outputLayerWeights)
# nn.updateWeights(1)
# print(nn.weights)
# print(nn.outputLayerWeights)
#print(nn.gradient)
#print(nn.gradientOutputLayer)
# nn.updateWeights(1)
# print(nn.weights)
#print(nn.error(x,y))
# from sklearn.datasets import load_iris

# data = load_iris()
# X, y = data.data, data.target
# # iris = load_iris()
# # X = iris.data[:, (2, 3)] 
# # y = (iris.target==0).astype(np.int8)
# nn= NeuralNetwork(X= X, y = y, epochs= 100, method = 'Classification', layers = [2,2,2], learning_rate= 0.01, tol = 0.0001)
# nn.fit(X = X, y = y)



### XOR Data

# X = np.array([[1,1],
#     [0,0],
#     [1,0],
#     [0,1]])

# y = [0,0,1,1]
# nn= NeuralNetwork(epochs= 10, method = 'Logistic', hiddenLayers= 2, neuronsEachLayer= 2, learning_rate= 0.1, tol = 0.00001)
# nn.fit(X,y,X,y)
# print(nn.predict(X))
