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

# Preprocess the data
y = (y == 4).astype(int)  # 1 for malignant, 0 for benign
# print(y[:20]) check change
# what size
print(X.shape, y.shape)

# handle NANs by replacing them with mean of the column
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))    
X[inds] = np.take(col_means, inds[1])
print(X[:5]) # check change

# scale the features to have mean 0 and variance 1
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print(X[:5]) # check change

# Split the data into training testing and validation sets

# randomize values before splitting
np.random.seed(21)
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

trainingx = X[:500]
trainingy = y[:500]

testingx = X[500:600]
testingy = y[500:600]

validationx = X[600:]
validationy = y[600:]

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
    def fit(self, X, y, lr=0.0001, epochs=1000, lamRegulation=0.01):
        N = X.shape[0]

        for epoch in range(epochs):
            # Predict probabilities
            y_pred = self.predict_proba(X)

            # Compute gradients
            dw = (1 / N) * np.dot(X.T, (y_pred - y)) + (lamRegulation / N) * self.w
            db = np.mean(y_pred - y) + (lamRegulation / N) * self.b

            # Update weights and bias
            self.w -= lr * dw
            self.b -= lr * db


# Initialize and train the model
model = LogisticRegression(dimensions=trainingx.shape[1])
model.fit(trainingx, trainingy, lr=0.001, epochs=5000, lamRegulation = 0.01)

# Evaluate the model
probs = model.predict_proba(testingx)
preds = model.predict(testingx)
test_loss = loss(testingy, probs)
accuracy = np.mean(preds.flatten() == testingy)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")


# Validate the model



