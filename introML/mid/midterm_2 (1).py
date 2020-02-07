
# coding: utf-8

# This question is about Ridge regression and Leave-one-out-crosss validation 
# There are two functions: 
#      (a) ridge_cv - a part of its code is given, you will complete its code
#      (b) leave_one_out_split - This one is empty, you will provide its code
# Finally, you will write the code to plot the results 

# ---------------------------------------------------------------------------

# Evgeny Sorokin BS3-DS2

#TO DO --------------------- Import all libraries here ---------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# TO DO ------- 5 Points ------------ Ridge regression using cross validation ---------------------
def ridge_cv(X, y, alphas):
    '''
    performs cross-validation of Ridge regression to find optimal value of alpha
    Arguments:
    X - training data
    Y - training data labels
    aplhas - list of alphas to choose from

    Returns:
    results - list of mse (mean squarred error), for each of possible alphas
    '''
    length = len(X)
    splits = leave_one_out_split(length)
    results = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, normalize=True)
        mse = 0
        for split in splits:
            index_train, index_test = split[0], split[1]
            #--- complete code here to----------- 
            # (a) split the data into test and train as per the split indices
            # (b) fit the model
            # (c) find mse - mean squared error
            # Split
            x_train = X.iloc[index_train]
            y_train = y.iloc[index_train]
            x_test = X.iloc[index_test]
            y_test = y.iloc[index_test]
            
            # fit
            model.fit(x_train, y_train)
            
            # mse
            predicted = model.predict(x_test)
            mse += mean_squared_error(y_test, predicted)
            
            
        results.append(mse / length)
    return results


#TO DO ----- 5 Points -------------------- leave one out cross validation ---------------------
def leave_one_out_split(length):
    '''
    the method should perform splits according to leave-one-out cross-validation, i.e.:
    each time only one sample is used for testing, all others are used for training
    
    returns a list of tuples of train and test indexes for each split:
    [([train_indices_1], [test_index_1]), ([train_indices_2], [test_index_2]), ...]
    each tuple is a split
    
    pay attention - we don't split actual data, we only generate indices for splitting
  
    Arguments:
    length - #rows in dataset

    Returns:
    splits - list of tuples
    '''
    #-------------------------- Your code here -------------------------------
    splits = []
    items = [i for i in range(length)]
    
    for i in range(length):
        one = [i]
        cop = list(items)
        cop.remove(i)
        tup = (cop, one)
        splits.append(tup)

    return splits


# loading and pre-processing the dataset
hitters = pd.read_csv("Hitters.csv").dropna().drop("Player", axis=1)
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])

# Dropping the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = hitters.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Defining the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = hitters.Salary

alphas = [1e-15, 1e-10, 1e-5, 1e-3, 1e-2, 1, 3, 5]

results = ridge_cv(X, y, alphas)
print(results)



#TO DO ----- BONUS 1 Point --------------------- visualize the results ---------------------
'''
construct a figure that plots the MSE vs. alphas
xlabel: alpha
ylabel: MSE
xscale: log
title: MSE for different alpha levels for Ridge Regression
'''
    
#-------------------------- Your code here -------------------------------
plt.plot(alphas, results)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title('MSE for different alpha levels for Ridge Regression')
plt.show()

