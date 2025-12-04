# Author: Ian Ludlam
# Class : CSE 432 Machine Learning, Miami University
# Date  : 2024-06-15
# Description : SKI Models for Breast Cancer Detection
import numpy as np

# Random Forest Classification on Breast Cancer Wisconsin Diagnostic Dataset
# Using scikit-learn and chatGPT assistance

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Train Random Forest model
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# Predictions and evaluation
# -----------------------------
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Feature Importance
# -----------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 Important Features:\n", importances.head(10))



# SVM on Breast Cancer Wisconsin Diagnostic Dataset
# Using scikit-learn and chatGPT assistance

# SVM Classification on Breast Cancer Wisconsin Diagnostic Dataset
# Using scikit-learn

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# -----------------------------
# Load dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Create pipeline: StandardScaler + SVM
# -----------------------------
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# -----------------------------
# Train SVM model
# -----------------------------
svm_model.fit(X_train, y_train)

# -----------------------------
# Predictions + Evaluation
# -----------------------------
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
