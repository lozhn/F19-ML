import pandas as pd
import numpy as np
from numpy.random import seed#, shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

filename = 'countries.csv'
data = pd.read_csv(filename).dropna()
feature_name = data.columns[2:-1]
data = data.values

seed(0)
name = data[:, 0]
y = data[:, 1] == 'EUROPE'
# make class labels +-1  
y = y.astype('int') * 2 - 1 
X = data[:, 2:].astype('float')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

algo = LogisticRegression()
model = AdaBoostClassifier(base_estimator=algo, n_estimators=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))

model.estimator_weights_
log_w_i = np.zeros(len(y_train), dtype='float')
for i in range(len(log_w_i)):
    item = np.reshape(X_train[i, :], (1, 18))
    for j in range(len(model.estimators_)):
        y_pred = model.estimators_[j].predict(item)
        alpha = model.estimator_weights_[j]
        log_w_i[i] -= alpha * y_train[i] * y_pred[0]

w_i = np.exp(log_w_i)
w_i = w_i / np.sum(w_i)
not_outliers = w_i < np.mean(np.sort(w_i)[:-20]) + 3 * np.std(np.sort(w_i)[:-20])

print("{} outliers".format(len(y_train) - sum(not_outliers)))

algo = LogisticRegression()
model = AdaBoostClassifier(base_estimator=algo, n_estimators=10)
model.fit(X_train[not_outliers], y_train[not_outliers])
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))

plt.scatter(range(1,21), np.sort(w_i)[-20:])
plt.plot(range(1,21), np.sort(w_i)[-20:])
plt.title("samples weights")
plt.xlabel("sorted samples")
plt.yscale('log')
plt.ylabel("weights")
plt.xticks(range(0,21,2))
plt.show()
