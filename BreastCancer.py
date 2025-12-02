# Author : Ian Ludlam
# Class  : CSE 432 Machine Learning, Miami University
# Date   : 2024-06-15
# Description : Breast Cancer Detection using Machine Learning

import numpy as np

# Obtain the dataset from UCI Machine Learning Repository 
# using code from: (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+diagnostic)
# open the dataset
from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features.to_numpy()
y = breast_cancer_wisconsin_original.data.targets .to_numpy()
  
variables = np.array(breast_cancer_wisconsin_original.variables)
metadata = breast_cancer_wisconsin_original.metadata 

# metadata 
print(metadata) 
  
# variable information 
print(variables) 

# testing to help me see the data
print(X[:5]) # features ranked from 1-10
print(y[:5]) # 2 for benign, 4 for malignant (these are all benign)

# logistic regression model from scratch (similar to example from class)

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# loss function
def loss(true, pred):
    loss = -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))
    return loss

# logistic regression class
class LogisticRegression:
    # Initialize object
    def __init__(self, dimensions):
        self.w = np.zeros((dimensions, 1))
        self.b = 0.0
    
    # Probability prediction
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)
    # Label prediction
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    # Update weights and bias
    def fit(self, X, y, lr=0.01, epochs=1000):
        N = X.shape[0]

        for epoch in range(epochs):
            # Predict probabilities
            y_pred = self.predict_proba(X)
            
            loss_value = loss(y, y_pred)

            # Compute gradients
            dw = (1 / N) * np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)

            # Update weights and bias
            self.w -= lr * dw
            self.b -= lr * db

# Preprocess the data
y = (y == 4).astype(int)  # 1 for malignant, 0 for benign
# print(y[:20]) check change
# what size
print(X.shape, y.shape)

# Initialize and train the model
model = LogisticRegression(dimensions=X.shape[1])
model.fit(X, y, lr=0.01, epochs=1000)


