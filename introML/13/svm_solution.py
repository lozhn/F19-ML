import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_wine

X, y = load_wine(True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#TO DO ---- 4 POINTS --------------------- Accuracy --------------------
# calculate the accuracy
# Soluion ---------------------------------------------------
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
# ---------------------------------------------------------------------

#TO DO ---- 5 POINTS --------------------- Scaling --------------------
# From data remove mean and scale to unit variance
# Solution ----------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# ---------------------------------------------------------------------

#TO DO ---- 5 POINTS --------------------- Kernels --------------------
# Compare linear, rbf and sigmoid kernels
# Solution ----------------------------------------------------------
for kernel in ['linear', 'rbf', 'sigmoid']:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(kernel, accuracy_score(y_test, y_pred))
# ---------------------------------------------------------------------

# TO DO ---- 5 POINTS --------------------- Accuracy --------------------
# print all test samples on which the last classifier makes a mistake
# Soluion ---------------------------------------------------
print(X_test[y_test != y_pred])
# ---------------------------------------------------------------------

