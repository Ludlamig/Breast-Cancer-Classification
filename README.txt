==========================
README - Breast Cancer Classification (CSE 432 Final Project)
==========================

This project implements and compares three machine learning models on the UCI Breast Cancer Wisconsin Diagnostic dataset. The goal is to evaluate performance differences between a from-scratch implementation and two scikit-learn models using accuracy, training time, and confusion matrices.

This work was completed for CSE 432: Machine Learning at Miami University.

PROJECT FILES

BreastCancer.py
- Implements Logistic Regression completely from scratch
- Handles data loading, preprocessing, training, and prediction
- Includes gradient descent optimization
- Returns accuracy, confusion matrix, and training time

SKI-Models.py
- Trains two scikit-learn models:
* Random Forest Classifier
* Support Vector Machine (RBF Kernel)
- Loads the from-scratch model results from BreastCancer.py
- Generates confusion matrices for all three models
- Prints comparison table of accuracy and training time

README.txt
- Documentation for running and understanding the project

DEPENDENCIES

This project requires Python 3.10 or higher.

Required libraries:
numpy
pandas
scikit-learn
matplotlib
ucimlrepo

pip install numpy pandas scikit-learn matplotlib ucimlrepo


HOW TO RUN THE PROJECT

Run the from-scratch Logistic Regression model:

python BreastCancer.py

This script will:
- Load and preprocess the dataset
- Train logistic regression using gradient descent
- Output accuracy, confusion matrix, and training time

Run the scikit-learn models and comparison script:

python SKI-Models.py

This script will:
- Train the Random Forest and SVM models
- Import results from the from-scratch model
- Display confusion matrices for all three models
- Print a performance comparison table

PROGRAM OUTPUT

The scripts will produce:

Console Output:
- Accuracy scores for each model
- Training times for each model
- Confusion matrices in numeric format

Visual Output:
- Plot of confusion matrix for each model
- Side-by-side comparison plot for all three confusion matrices

Comparison Table Example:
Model Accuracy Training Time (s)
Logistic Regression (Scratch) 0.97 0.196958
Random Forest 0.958042 0.564527
SVM (RBF Kernel) 0.979021 0.007587

PROJECT DESCRIPTION

This project compares:
1. A Logistic Regression model implemented entirely from scratch
2. A Random Forest classifier trained with scikit-learn
3. A Support Vector Machine (RBF Kernel) trained with scikit-learn

All models use the same dataset and are evaluated using:
- Accuracy
- Confusion Matrix
- Training Time
- Qualitative analysis of strengths and weaknesses

The project demonstrates understanding of both low-level machine learning implementation and the use of modern ML libraries.

AUTHOR

Ian Ludlam
CSE 432 - Miami University
Final Project, December 2025