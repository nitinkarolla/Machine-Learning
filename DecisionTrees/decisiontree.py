import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def data_generator(m,k):
    #Generating the X
    X = [] 
    for j in range(m):
        y = []
        a = np.random.binomial(1, p=0.5, size= 1)[0]
        y.append(a)
        for i in range(k-1):
            x = np.random.binomial(1, p=0.75, size= 1)[0]
            if x == 1:
                y.append(a)
            else:
                y.append(1-a)
                a = 1-a
        X.append(y)
        
    #Generating the Weights
    j = 0.9*0.9
    s = j * (1 - (0.9 ** (k-1)))/ (1- 0.9)
    w = [0]
    for i in range(k-1):
        w_i = 0.9**(i+2)/ s
        w.append(w_i)
    
    #Generating the Y
    Y = []
    for i in range(m):
        y_i = [a*b for a,b in zip(X[i],w)]
        if sum(y_i) >= 0.5:
            Y.append(X[i][0])
        else:
            Y.append(1-X[i][0])
    
    #Generate column names
    colnames = [ 'X'+ str(i)  for i in range(1,k+1)]
    
    df = pd.DataFrame(data= X, columns= colnames)
    df['Y'] = pd.Series(data = Y)
    return(df)


data = pd.read_csv("E:\Semester - 2\MachineLearning\Decision Trees\data.csv")


class Node():
    def __init__(self,
                 data = None,
                 rule = None,
                 error = None,
                 child = None
                ):
        self.data = data
        self.rule = rule
        self.error = error
        self.child = child




class Decision_Tree():
    
    def __init__(self, root = None):
        self.root = root
        
        
    def entropy(self, data, variable):
        # Calculate the entropy of target variable
        entropy_value = sum([(-data[variable].value_counts()[i]/ data[variable].count()) * np.log2((data[variable].value_counts()[i]/ data[variable].count()) + 0.00000001) for i in data[variable].unique()])
        return entropy_value
    
    
    def information_gain(self, data, variable, target):
        infomation_content = sum([data[variable].value_counts()[i]/data[variable].count() * self.entropy(data[data[variable]== i], target) for i in data[variable].unique()])
        info_gain = self.entropy(data, target) - infomation_content
        return(info_gain)
    
    
    def split_variable_identification(self, data, target):
        #loop through all features and calculate information gain for each feature
        variable_ig_required = list(data.columns)
        variable_ig_required.remove('Y')
        ig_values = [(i, self.information_gain(data,i,'Y')) for i in variable_ig_required]
        if len(ig_values) != 0:
            split_variable = max(ig_values, key = lambda item : item[1])
        else:
            split_variable = (0,0)
        return(split_variable)
    

    def split_data(self, data, split_variable): 
        splitted_data = [Node(data = (data[data[split_variable] == i].drop(split_variable,1)), rule = (split_variable,i)) for i in data[split_variable].unique()]
        return(splitted_data)
    
    
    def fit(self, data, target):
        
        if type(data) != Node:
            data = Node(data = data)
            self.root = data

        if self.split_variable_identification(data.data, target)[1] == 0:
            return

        split_variable = self.split_variable_identification(data.data, target)[0]
        data.child = self.split_data(data.data, split_variable)

        for i in data.child:
            if i.data['Y'].nunique() == 2:
                self.fit(i, target)
            
    
    def get_rules(self, model = None ,ruleList = []):
        
        if model == None:
            model = self.root
        
        ruleList.append(model.rule)

        if model.child == None:
            ruleList.append(model.data['Y'].mode()[0])
            return print(ruleList[1:])

        for i in model.child:
            self.get_rules(i,ruleList.copy())
    

    def predict_row(self, model, row):
        
        if model.child == None:
            return(model.data['Y'].mode()[0])

        variable = model.child[0].rule[0]
        row_value = row[variable]
        for i in model.child:
            if i.rule[1] == row_value:
                return self.predict_row(i, row)
            
    def predict(self, test):
        predicted_y = []
        for i in test.iterrows():
            x = i[1]
            y = self.predict_row(self.root, x)
            predicted_y.append(y)
        return pd.Series(predicted_y)
    
    def training_error(self):
        predict_train = self.predict(data)
        return (1 -sum(data['Y'] == predict_train)/ len(data))
            
        







class Decision_Tree_Gini_Impurity():
    """
    Decision Tree Gini Impurity is trained on data with a target variable. It is built on split variable which is indentified using the logic of Gini Impurity.
    """
    def __init__(self, root = None):
        self.root = root
        
        
    def gini_index(self, data, variable):
        """
        Calcuates the gini index for the given data and target variable
        """
        # Calculate the entropy of target variable
        gini_index = 1- sum([pow(data[variable].value_counts()[i]/ data[variable].count(),2) for i in data[variable].unique()])
        return gini_index
    
    
    def gini_gain(self, data, variable, target):
        """
        Calculates the gini impurity for the given variable and data
        """
        gini_index_varaible = sum([data[variable].value_counts()[i]/data[variable].count() * self.gini_index(data[data[variable]== i], target) for i in data[variable].unique()])
        gini_gain_value = gini_index_varaible
        return(gini_gain_value)
    
    
    def split_variable_identification(self, data, target):
        """
        Identifies the split variable based on data and target
        """
        #loop through all features and calculate gini gain for each feature
        variable_ig_required = list(data.columns)
        variable_ig_required.remove('Y')
        ig_values = [(i, self.gini_gain(data,i,'Y')) for i in variable_ig_required]
        if len(ig_values) != 0:
            split_variable = min(ig_values, key = lambda item : item[1])
        else:
            split_variable = (0,0)
        return(split_variable)
    

    def split_data(self, data, split_variable): 
        """
        Splits the data after identifying the split variable, assigns data and rule to the node.
        """
        splitted_data = [Node(data = (data[data[split_variable] == i].drop(split_variable,1)), rule = (split_variable,i)) for i in data[split_variable].unique()]
        return(splitted_data)
    
    
    def fit(self, data, target):
        """
        Fit is used to fit decision trees on the data for a given target variable
        """
        if type(data) != Node:
            data = Node(data = data)
            self.root = data

    #    if self.split_variable_identification(data.data, target)[1] == 0:
    #        return

        split_variable = self.split_variable_identification(data.data, target)[0]
        data.child = self.split_data(data.data, split_variable)

        for i in data.child:
            if i.data['Y'].nunique() != 1: 
                self.fit(i, target)
            
    
    def get_rules(self, model = None ,ruleList = []):
        """
        Returns the rules for each leaf and the major class in the leaf
        """
        if model == None:
            model = self.root
        ruleList.append(model.rule)

        if model.child == None:
            ruleList.append(model.data['Y'].mode()[0])
            return print(ruleList[1:])

        for i in model.child:
            self.get_rules(i,ruleList.copy())
    

    def predict_row(self, model, row):
        """
        This function returns the prediction for the a single sample of data using the fitted data
        """
        if model.child == None:
            return(model.data['Y'].mode()[0])

        variable = model.child[0].rule[0]
        row_value = row[variable]
        for i in model.child:
            if i.rule[1] == row_value:
                return self.predict_row(i, row)
            
    def predict(self, test):
        """
        Predict funtion will take an input data and return the prediction based on the fitted decision tree
        """
        predicted_y = []
        for i in test.iterrows():
            x = i[1]
            y = self.predict_row(self.root, x)
            predicted_y.append(y)
        return pd.Series(predicted_y)
    
    def training_error(self):
        """
        Returns the training error of the  fitted decision tree
        """
        predict_train = self.predict(data)
        return (1 -sum(data['Y'] == predict_train)/ len(data))
            


model = Decision_Tree_Gini_Impurity()

model.fit(data, 'Y')

print(model.training_error())