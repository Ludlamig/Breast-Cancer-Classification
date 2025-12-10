# Author : Ian Ludlam
# Class  : CSE 432 Machine Learning, Miami University
# Date   : 2024-06-15
# Description : Breast Cancer Detection using Machine Learning

# From scratch ---------------------------------->

import numpy as np

# Obtain the dataset from UCI Machine Learning Repository 
# using code from: (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+diagnostic)
# open the dataset
from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset -------------------------------
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features.to_numpy()
y = breast_cancer_wisconsin_original.data.targets .to_numpy()

# testing to help me see the data
print(X[:5]) # features ranked from 1-10
print(y[:5]) # 2 for benign, 4 for malignant (these are all benign)

# Preprocess the data --------------------------

y = (y == 4).astype(int)  # 1 for malignant, 0 for benign

# handle NANs by replacing them with mean of the column
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))    
X[inds] = np.take(col_means, inds[1])

# scale the features to have mean 0 and variance 1
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# reshape y to be a column vector
y = y.reshape(-1, 1)

# randomize values before splitting sinc this dataset is ordered
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
    def fit(self, X, y, lr=0.01, epochs=10000):
        N = len(X)

        for epoch in range(epochs):
            # Predict probabilities
            y_pred = self.predict_proba(X)

            # Compute gradients
            dw = (1 / N) * np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)

            # Update weights and bias
            self.w -= lr * dw
            self.b -= lr * db


# Initialize and train the model
model = LogisticRegression(dimensions=trainingx.shape[1])
model.fit(trainingx, trainingy, lr=0.01, epochs=5000)

# Evaluate the model
probs = model.predict_proba(testingx)
preds = model.predict(testingx)

test_loss = loss(testingy, probs)
accuracy = np.mean(preds == testingy)

print("Test Results:")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Validate the model

# try different learning rate and epochs
# print the max validation accuracy after training with different hyperparameters

best_epochs = 5000
best_lr = 0.01

for lr in [0.001, 0.01, 0.1]:
    for epochs in [1000, 5000, 10000]:
        model = LogisticRegression(dimensions=trainingx.shape[1])
        model.fit(trainingx, trainingy, lr=lr, epochs=epochs)

        val_probs = model.predict_proba(validationx)
        val_loss = loss(validationy, val_probs)
        val_preds = model.predict(validationx)
        val_accuracy = np.mean(val_preds == validationy)

        #save if best model so far
        if accuracy < val_accuracy:
            accuracy = val_accuracy
            best_lr = lr
            best_epochs = epochs
print("Best Validation Results:")
print(f"Learning Rate: {best_lr}, Epochs: {best_epochs}, Validation Accuracy: {accuracy:.4f}")

# Begin AI Assisted metrics comparisons ----------------------- > 

def run_scratch_logistic_regression():
    import time
    from sklearn.metrics import confusion_matrix, classification_report

    model = LogisticRegression(dimensions=trainingx.shape[1])

    start_time = time.perf_counter()
    model.fit(trainingx, trainingy, lr=0.01, epochs=5000)
    train_time = time.perf_counter() - start_time

    probs = model.predict_proba(testingx)
    preds = model.predict(testingx)

    accuracy = np.mean(preds == testingy)
    bc_cm = confusion_matrix(testingy, preds)
    tn, fp, fn, tp = bc_cm.ravel()


    return {
        "accuracy": accuracy,
        "training_time": train_time,
        "confusion_matrix": bc_cm
    }
