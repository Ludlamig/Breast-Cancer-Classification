# Author: Ian Ludlam
# Class : CSE 432 Machine Learning, Miami University
# Date  : 2024-06-15
# Description : SKI Models for Breast Cancer Detection

# Random Forest Classification on Breast Cancer Wisconsin Diagnostic Dataset
# Using scikit-learn and chatGPT assistance

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import BreastCancer 


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


# -----------------------------
# Feature Importance
# -----------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

#---------------------------------------------------------------------------

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

# ----------------------------
# Compare models
# ----------------------------

# Time to completion
import time

# Random Forest
start_time = time.perf_counter()
rf.fit(X_train, y_train)
rf_time = time.perf_counter() - start_time

y_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred)
rf_cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = rf_cm.ravel()

#SVM 
start_time = time.perf_counter()
svm_model.fit(X_train, y_train)
svm_time = time.perf_counter() - start_time

y_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, y_pred)
svm_cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = svm_cm.ravel()

bcResults = BreastCancer.run_scratch_logistic_regression()
bc_cm = bcResults["confusion_matrix"]
bc_acc = bcResults["accuracy"]
bc_time = bcResults["training_time"]


# Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

models = [
    ("Logistic Regression (Scratch)", bc_cm),
    ("Random Forest", rf_cm),
    ("SVM (RBF)", svm_cm)
]

# Create side-by-side plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, cm) in zip(axes, models):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.suptitle("Confusion Matrices Comparison", fontsize=14)
plt.tight_layout()
plt.show()

# combine
results = [
    {
        "Model": "Logistic Regression (From Scratch)",
        "Accuracy": bc_acc,
        "Training Time (s)": bc_time
    },
    {
        "Model": "Random Forest",
        "Accuracy": rf_acc,
        "Training Time (s)": rf_time
    },
    {
        "Model": "SVM (RBF Kernel)",
        "Accuracy": svm_acc,
        "Training Time (s)": svm_time
    }
]

results_df = pd.DataFrame(results)
print(results_df)
