import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# TO DO ---- 5 POINTS --------------------- Accuracy --------------------
# calculate the accuracy



# ---------------------------------------------------------------------

# TO DO ---- 5 POINTS --------------------- Scaling --------------------
# From data remove mean and scale to unit variance



# ---------------------------------------------------------------------

# TO DO ---- 5 POINTS --------------------- Kernels --------------------
# Compare linear, rbf and sigmoid kernels



# ---------------------------------------------------------------------

# TO DO ---- 5 POINTS --------------------- Accuracy --------------------
# print all test samples on which the last classifier makes a mistake



# ---------------------------------------------------------------------

