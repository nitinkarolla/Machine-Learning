import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def data_generation (m = 1000):
    X = []
    col_names = []
    for i in range(15):
        x = np.random.normal(0,1,m)
        X.append(x)
        if i < 10:
            col_names.append('X' + str(i+1))
        else:
            col_names.append('X' + str(i+6))

    data = pd.DataFrame(data = np.array(X).T, columns= col_names)
    data['X11'] = data['X1'] + data['X2'] + np.random.normal(0, np.sqrt(0.1), m)
    data['X12'] = data['X3'] + data['X4'] + np.random.normal(0, np.sqrt(0.1), m)
    data['X13'] = data['X4'] + data['X5'] + np.random.normal(0, np.sqrt(0.1), m)
    data['X14'] = 0.1 * data['X7'] + np.random.normal(0, np.sqrt(0.1), m)
    data['X15'] = data['X2'].apply(lambda x : (2*x) - 10) + np.random.normal(0, np.sqrt(0.1), m)
    
    columns = ['X' + str(i) for i in range(1,21)]
    data = data[columns]
    
    point_six = [0.6**i for i in range(1,11)]
    Y = []
    for index, row in data.iterrows():
        y = 10 + sum(np.multiply(np.array(row[(data.columns[:10])]), np.array(point_six))) + np.random.normal(0, np.sqrt(0.1), 1)
        Y.append(float(y))
    
    data['Y'] = Y
    
    return data

class LinearRegression():
    
    def __init__(self, method = None, lambda_value = 0.1):
        self.method = method
        self.lambda_value = lambda_value
        
    def prepare_data(self, data, target):
        data['Bias'] = 1
        self.variables = data.drop(target, axis = 1).columns
        self.X = data.drop(target, axis = 1).values
        data.drop('Bias', axis = 1, inplace = True)
        self.Y = data[target].values
        
    def fit(self, data, target):
        self.data = data
        self.target = target
        self.prepare_data(self.data, self.target)
        
        if self.method == None :
            self.weights = np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), np.matmul(self.X.T, self.Y))
        elif self.method == "Ridge" :
            self.weights = np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X) + (self.lambda_value * np.identity(self.X.shape[1]))), np.matmul(self.X.T, self.Y))
        elif self.method == "Lasso":
            #print("Working..")
            count_weight = self.X.shape[1]
            self.weights = [0 for i in range(count_weight)]
            while True:
                old_weights = self.weights.copy()
                for i in range(len(self.weights)):
                    denom_value = np.matmul(self.X[:,i].T, self.X[:,i])
                    actual_value = (self.Y - np.matmul(self.X,self.weights))
                    cal_x_upper = (np.matmul((-1 * self.X[:,i].T), actual_value) + (self.lambda_value/2))/ denom_value
                    cal_x_lower = (np.matmul((-1 * self.X[:,i].T), actual_value) - (self.lambda_value/2))/ denom_value
                    if  cal_x_upper < self.weights[i] :
                       self.weights[i] = self.weights[i] + (np.matmul((self.X[:,i].T),actual_value) - (self.lambda_value/2))/ denom_value
                    elif cal_x_lower > self.weights[i] :
                        self.weights[i] = self.weights[i] +(np.matmul((self.X[:,i].T),actual_value) + (self.lambda_value/2))/ denom_value
                    else:
                        self.weights[i] = 0
                #Stopping criteria
                updates = [k - l for k, l in zip(old_weights, self.weights)]
                if max(updates) < 1e-4 and abs(min(updates)) < 1e-4:
                    break


    def predict_row(self, row):
        y_pred = np.sum(np.multiply(self.weights, row))
        return y_pred
        
    def predict(self,test):
        test['bias'] = 1
        y_predicted = []
        for index,row in test.iterrows():
            y_predicted.append(self.predict_row(row))
        return y_predicted
    
    def training_error(self):
        predicted_y = self.predict(self.data.drop('Y', axis = 1))
        mse = []
        for i in range(len(predicted_y)):
            err = ((predicted_y[i] - self.Y[i])**2)
            mse.append(err)
        return sum(mse)/len(mse)
    
    def error(self, test):
        predicted_y = self.predict(test.drop('Y', axis = 1))
        mse = []
        for i in range(len(predicted_y)):
            err = ((predicted_y[i] - test.Y[i])**2)
            mse.append(err)
        return sum(mse)/len(mse)
            
    def plot_weights(self):
        plt.figure(figsize = (10,6))
        sns.barplot(self.variables[:-1], self.weights[:-1])
        
    def plot_weights_comparision(self):
        original_weights = [0.6**i for i in range(1,11)] + [0 for i in range(10)]
        plt.figure(figsize = (10,6))
        w_b = pd.DataFrame({'columns' : self.variables[:-1], 'weights' : self.weights[:-1]})
        w_b['org_cat'] = 'Calculated'
        temp = pd.DataFrame({'columns' : self.variables[:-1], 'weights' : original_weights})
        temp['org_cat'] = 'Actual'
        w_b = w_b.append(temp)
        plt.figure(figsize = (10,6))
        sns.barplot(data = w_b, x = "columns", y = "weights", hue = "org_cat")
        plt.title("Comparision of Weights")
        
    def plot_bias_comparision(self):
        plt.figure(figsize = (10,6))
        sns.barplot(x = ['Calculated', 'Actual'], y= [self.weights[-1], 10 ])
        plt.title("Comparision of Bias")


df = data_generation()

lasso = LinearRegression(method = 'Lasso', lambda_value = 10)
lasso.fit(df,'Y')
print(lasso.weights)