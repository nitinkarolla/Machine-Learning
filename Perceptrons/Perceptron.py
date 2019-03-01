import numpy as np
import pandas as pd

def data_generator(data_size, feature_count, parameter_e = 0.1):
    X = {}
    for i in range(feature_count - 1):
        X['X'+ str(i)] = np.random.normal(loc = 0, scale = 1, size = data_size)
    df = pd.DataFrame(data = X)
    
    last_feature_dist = np.random.exponential(scale = 1, size = data_size)

    X_last = []
    Y = []
    for i in last_feature_dist:
        p = np.random.binomial(n = 1, p = 0.5, size = 1)
        if p ==0:
            X_last.append(i + parameter_e)
            Y.append(1)
        else:
            X_last.append(-1 * (i + parameter_e))
            Y.append(-1)

    df['X'+str(feature_count)] = X_last
    df['Y'] = Y
    return df


class Perceptron_Learning_Algorithm():

    def __init__(self, weight_vector = None ,data = None, target = None, store_error = None, learning_rate = 1 ):
        self.weight_vector = weight_vector
        self.data = data
        self.target = target
        self.weight_update_counter = 0
        self.row_iterated_counter = 0
        self.while_loop_counter = 0
        self.store_error = store_error
        self.learning_rate = learning_rate

    
    def _initialize_weights(self, dimension):
        self.weight_vector = np.zeros(shape = dimension)

    def _update_weights(self, weight_vector, x, y):
        self.weight_vector = weight_vector + self.learning_rate * y * x 
    
    def _intialize_data(self):
        df = self.data.copy()
        # Adding Bias with values 1 in front of data frame
        df.insert(loc=0, column='Bias', value=[1.0 for i in range(df.shape[0])])  
        # Extracting values to numpy array
        Y = df['Y'].values
        X = df.drop('Y', axis = 1).values
        return X,Y
        
    def fit(self, data, target):
        self.data = data
        self.target = target
        X,Y = self._intialize_data()

        if self.weight_vector == None :
            self._initialize_weights(X.shape[1])
            
        if self.store_error == None:
            self.store_error = [[],[]]
        #### Fitting the data
        count = 0
        while count <= len(X):
            for i in range(len(X)):
                if np.dot(self.weight_vector, X[i]) * Y[i] <= 0 :
                    self._update_weights(self.weight_vector,  X[i], Y[i])
                    count = 0
                    self.weight_update_counter  = self.weight_update_counter + 1
                    self.store_error[0].append(self.weight_update_counter)
                    self.store_error[1].append(self.training_error())
                        
                count = count + 1
                self.row_iterated_counter = self.row_iterated_counter + 1
            self.while_loop_counter = self.while_loop_counter + 1
            if self.while_loop_counter >= 4:
                break

    def predict(self, data):
        data.insert(loc=0, column='Bias', value=[1.0 for i in range(data.shape[0])])
        pred_y = [1 if np.dot(self.weight_vector, i) > 0 else -1 for i in data.values]
        return pred_y

    def predict_value(self, data):
        data.insert(loc=0, column='Bias', value=[1.0 for i in range(data.shape[0])])
        pred_y = [np.dot(self.weight_vector, i) for i in data.values]
        return pred_y

    def training_error(self):
        predicted_y = self.predict(self.data.drop('Y', axis = 1))
        return (1 -sum(self.data['Y'] == predicted_y)/ len(self.data))

            
       

        
        

data = data_generator(100,5)

model = Perceptron_Learning_Algorithm(learning_rate= 1)

model.fit(data,'Y')

print(model.training_error())

print(model.weight_vector)

#print(model.predict_value(data.drop('Y', axis = 1)))
    




    