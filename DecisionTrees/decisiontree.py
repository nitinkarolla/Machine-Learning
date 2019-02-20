# Importing the required packages for the assignment
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mode

def data_generator(m):
    """
    Input : 
        m - Number of samples
    
    Output : 
        Dataframe with k+1 columns i.e. k features that represent X 
        and 1 the represents Y
    """
    colnames = [ 'X'+ str(i)  for i in range(21)]
    
    X = [] 
    for j in range(m):
        y = []
        a = np.random.binomial(1, p=0.5, size= 1)[0]
        y.append(a)
        for i in range(14):
            x = np.random.binomial(1, p=0.75, size= 1)[0]
            if x == 1:
                y.append(y[i])
            else:
                y.append(1-y[i])
        for i in range(6):
            x = np.random.binomial(1, p = 0.5, size = 1)[0]
            y.append(x)
        X.append(y)
     
    df = pd.DataFrame(data= X, columns= colnames)
    
    #Generating the Y
    Y = []
    for i in X:
        if i[0] == 0:
            y = mode(i[1:8])
        else :
            y = mode(i[8:15])
        Y.append(y)
    
    #Generate column names
    df['Y'] = pd.Series(data = Y)
    return(df)


class Node():
    """
    Node is a data structure which will be used for decision trees.
    
    Input :
        data = training data post split is stored
        rule = feature on which the split led to this node and the 
        value of the feature
        child = nodes of children of all this node are present 
        after the split
    """
    def __init__(self,
                 data = None,
                 rule = None,
                 child = None,
                 depth = None
                ):
        self.data = data
        self.rule = rule
        self.child = child
        self.depth = depth


class Decision_Tree_ID3():
    """
    Decision Tree ID3 is trained on data with a target variable. It 
    is built on split variable which is indentified using the logic 
    of information gain 
    """
    def __init__(self, root = None, termination_depth = None, min_leaf_size = None, sig_threshold = None, var = None):
        self.root = root
        self.termination_depth = termination_depth
        self.min_leaf_size = min_leaf_size
        self.sig_threshold = sig_threshold
        if var == None:
            self.var = []
        
    def _entropy(self, data, variable):
        """
        Calcuates the entropy for the given data and target variable
        """
        entropy_value = sum([(-data[variable].value_counts()[i]/ 
                              data[variable].count()) * np.log2((
            data[variable].value_counts()[i]/ 
            data[variable].count()) + 0.00000001) 
                             for i in data[variable].unique()])
        return entropy_value
    
    
    def _information_gain(self, data, variable, target):
        """
        Calculates the information gain for the given variable and data
        """
        infomation_content = sum([data[variable].value_counts()[i]/
                                  data[variable].count()
                                  * self._entropy(data[data[variable]== i], 
                                                 target) 
                                  for i in data[variable].unique()])
        info_gain = self._entropy(data, target) - infomation_content
        return(info_gain)
    
    
    def _split_variable_identification(self, data, target):
        """
        Identifies the split variable based on data and target
        """
        #loop through all features and calculate information gain for each feature
        variable_ig_required = list(data.columns)
        variable_ig_required.remove('Y')
        ig_values = [(i, self._information_gain(data,i,'Y')) 
                     for i in variable_ig_required]
        if len(ig_values) != 0:
            split_variable = max(ig_values, key = lambda item : item[1])
        else:
            split_variable = (0,0)
        return(split_variable)
    
    def _chi_square(self,data,var,target):
        chi_square = []
        for i in data[var].unique():
            for j in data[target].unique():
                expected_x = (data[var].value_counts()[i]/len(data[var]))
                expected_y = (data[target].value_counts()[j]/len(data[target]))
                expected = expected_x * expected_y * len(data[var])
                #print(expected)

                observed = data[(data[var] == i) & (data[target] == j )].shape[0]
                #print(observed)

                out = (expected - observed)**2 / expected
                chi_square.append(out)
        return (sum(chi_square))
    

    def _split_data(self, data, split_variable): 
        """
        Splits the data after identifying the split variable, assigns 
        data and rule to the node.
        """
        splitted_data = [Node(data = (data[data[split_variable] == i].
                                      drop(split_variable,1)),
                              rule = (split_variable,i)) 
                         for i in data[split_variable].unique()]
        return(splitted_data)
    
    
    def fit(self, data, target):
        """
        Fit is used to fit decision trees on the data for a given target variable
        """
        if type(data) != Node:
            data = Node(data = data, depth = 0)
            self.root = data
        
        #Terminating Conditions
        if self._split_variable_identification(data.data, target)[1] == 0 :
            return
        if self.termination_depth != None:
            if data.depth == self.termination_depth:
                return
        if self.min_leaf_size != None:
            if data.data.shape[0] <= self.min_leaf_size:
                return
        
        split_variable = self._split_variable_identification(data.data, target)[0]
        
        #Terminating Conditions
        if self.sig_threshold != None:
            if self._chi_square(data.data,split_variable,target) < self.sig_threshold:
                return
        
        
        data.child = self._split_data(data.data, split_variable)
        for i in data.child:
            i.depth = data.depth + 1
               
        for i in data.child:
            if i.data['Y'].nunique() != 1:
                self.fit(i, target)
            
    
    def get_rules(self, model = None ,ruleList = None):
        """
        Returns the rules for each leaf and the major class in the leaf
        """
        if ruleList == None:
            ruleList = []
        if model == None:
            model = self.root
        ruleList.append(model.rule)
        if model.child == None:
            ruleList.append(model.data['Y'].mode()[0])
            return print(ruleList[1:])
        for i in model.child:
            self.get_rules(i,ruleList.copy())
            
    def get_irrelevant_variable(self, irrelevant_variables, model = None ):
        """
        Returns the count of irrelevant variables present in the decision tree
        """
           
        if model == None:
            model = self.root
        if model.child == None:
            return
        for i in model.child:
            if i.rule[0] in irrelevant_variables:
                self.var.append(i.rule[0])
                #print(var)
            self.get_irrelevant_variable(irrelevant_variables,i)
        return list(set(self.var))
    

    def _predict_row(self, model, row):
        """
        This function returns the prediction for the a single sample of 
        data using the fitted data
        """
        if model.child == None:
            return(model.data['Y'].mode()[0])

        variable = model.child[0].rule[0]
        row_value = row[variable]
        for i in model.child:
            if i.rule[1] == row_value:
                return self._predict_row(i, row)
            
    def predict(self, test):
        """
        Predict funtion will take an input data and return the prediction
        based on the fitted decision tree
        """
        predicted_y = []
        for i in test.iterrows():
            x = i[1]
            y = self._predict_row(self.root, x)
            predicted_y.append(y)
        return pd.Series(predicted_y)
    
    def training_error(self):
        """
        Returns the training error of the  fitted decision tree
        """
        predict_train = self.predict(data)
        return (1 -sum(data['Y'] == predict_train)/ len(data))
    
    def error(self, test, target):
        """
        Returns the training error of the  fitted decision tree
        """
        predict_test = self.predict(test.drop(target, axis = 1))
        return (1 -sum(test[target] == predict_test)/ len(test))
            



data = data_generator(1000)
tree = Decision_Tree_ID3()
tree.fit(data, 'Y')
print(tree.training_error())