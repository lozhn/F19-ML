# This question is about AdaBoost algorithm.
# You should implement it using library function (DecisionTreeClassifier) as a base classifier
# Don't do any additional imports, everything is already there
#
# There are two functions you need to implement:
#      (a) fit
#      (b) predict


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class BoostingTreeClassifier:

    def __init__(self, random_state):
        self.random_state = random_state
        self.classifiers = []
        self.tree_weights = None

#TO DO ---- 10 POINTS ---------- Implement the fit function ---------------------
    def fit(self, X, y, n_trees):
        """Trains n_trees classifiers based on AdaBoost algorithm - i.e. applying same
        model on samples while changing their weights. You should only use library
        function DecisionTreeClassifier as a base classifier, the boosting algorithm
        itself should be written from scratch. Store trained tree classifiers
        in self.classifiers. Calculate tree weight for each classifier and store them
        in self.tree_weights. Initialise DecisionTreeClassifier with self.random_state

        :param X: train data
        :param y: train labels
        :param n_trees: number of trees to train
        :return: doesn't return anything
        """

#TO DO ---- 5 POINTS ---------- Implement the predict function ---------------------
    def predict(self, X):
        """Makes final predictions aggregating predictions of trained classifiers

        :param X: test data
        :return: predictions
        """



# loading and pre-processing titanic data set
titanic = pd.read_csv('titanic_modified.csv').dropna()
data = titanic[['Pclass', 'Age', 'SibSp', 'Parch']].values
labels = titanic.iloc[:, 6].values
# changing labels so that we can apply boosting
labels[np.argwhere(labels == 0)] = -1
# splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)

# setting constants
rand_state = 3
T = 10  # number of trees

# measuring accuracy using one decision tree
tree = DecisionTreeClassifier(random_state=rand_state)
tree.fit(X_train, y_train)
print('One tree accuracy:', accuracy_score(tree.predict(X_test), y_test))

# measuring accuracy using an ensemble based on boosting
ensemble = BoostingTreeClassifier(random_state=rand_state)
ensemble.fit(X_train, y_train, T)
print('Ensemble accuracy:', accuracy_score(ensemble.predict(X_test), y_test))
