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



