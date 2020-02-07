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
        self.tree_weights = []
        self.sample_weights = []

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
        self.sample_weights = []
        N = len(X)
        
        self.sample_weights.append([])
        self.tree_weights = [1.0]
        for i in range(N):
            self.sample_weights[0].append(1 / N)
        
        for tree_ind in range(n_trees):
            self.classifiers.append(DecisionTreeClassifier(random_state=rand_state))
            self.classifiers[-1].fit(X, y)
            
            y_pred = self.classifiers[-1].predict(X)
            weighted_error = self.get_weighted_error(self.sample_weights[-1], y, y_pred)
            tree_weight = self.calculate_tree_weight(weighted_error)
            self.tree_weights.append(tree_weight)
            
            new_sample_weights = self.calculate_sample_weights(tree_weight, y, y_pred, self.sample_weights[-1])
            new_sample_weights = self.normalize_weights(new_sample_weights)
            self.sample_weights.append(new_sample_weights)
                        
            

        
        # weight update = 1/2 ln((1-weighted error(f))/weighted error(f))
        # sample update = alpha(-w) if correct, alpha(w) if incorrect
        # initial sample weights = 1/m
        # NORMALIZE


#TO DO ---- 5 POINTS ---------- Implement the predict function ---------------------
    def predict(self, X):
        """Makes final predictions aggregating predictions of trained classifiers

        :param X: test data (arrays)
        :return: predictions
        """
        predictions = []

        cl_predictions = []
        for classifier in self.classifiers:
            cl_predictions.append(classifier.predict(X))
            
        for index in range(len(cl_predictions[0])):
            pred = {-1: 0.0, 1: 0.0}
            for cl in range(len(cl_predictions)):
#                 if cl == 3:
#                     continue
                pred[cl_predictions[cl][index]] += self.tree_weights[cl]
            if pred[-1] > pred[1]:
                predictions.append(-1)
            else:
                predictions.append(1)
            
        return predictions
        
    
    def calculate_sample_weights(self, tree_weight, y, y_pred, sample_weights):
        s_ws = []
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                s_ws.append(sample_weights[i] ** (tree_weight))
            else:
                s_ws.append(sample_weights[i] ** (-tree_weight))
            
        return s_ws
    
    def get_weighted_error(self, sample_weights, y, y_pred):
        error = 0
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                error += sample_weights[i]
                
#         print("error equals = {}".format(error))
        return error  # Should I divide ny N?
    
    def calculate_tree_weight(self, error):
        print(error)
        if error == 0:
            return 1.0 / 2
        
        a = (1 - error) / error
        return 1.0/2 * np.log(a)
    
    def normalize_weights(self, weights):
        copied = []
        w_sum = sum(weights)
        for w in weights:
            copied.append(w / w_sum)
        return copied



# loading and pre-processing titanic data set
titanic = pd.read_csv('datasets/titanic_modified.csv').dropna()
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

# # measuring accuracy using an ensemble based on boosting
ensemble = BoostingTreeClassifier(random_state=rand_state)
ensemble.fit(X_train, y_train, T)
print('Ensemble accuracy:', accuracy_score(ensemble.predict(X_test), y_test))

# My ensemble trains a lot and becomes error-less on train set...


print(X_train)
print(X_train.shape)