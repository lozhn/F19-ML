# coding: utf-8

# This question is about simple Linear Regression using Gradient Descent - From scratch
# There are five functions out of which you will implement four functions: 
#      (a) split_data
#      (b) my_linear_regression
#      (c) plot_regression_line
#      (d) predict
# Finally, you are also asked to choose the best value for the learning rate using your implementation


# Evgeny Sorokin BS3-DS2
# ANSWER IS RIGHT HERE !!! 
# Best result is achieved by using lrate = 0.05, but with altering of lrate on each 100 iterations by half (RMSE ~ 6200) - lowest I met - RMSE = 3000
# If we don't alter lrate, then on lr=0.05 Overflow happens, the best in this case (ignoring lr=0.05) is lr=0.01 (RMSE ~ 10000)
# RMSE values for both examples may vary as it does not use CV

# ---------------------------------------------------------------------------


#TO DO----1 POINT ----------- Use your code to answer the following question ------------
#which learning rate gives the best results: (a)0.05, (b)0.01, (c)0.001
#YOUR ANSWER:



#TO DO --------------------- Import all libraries here --------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#TO DO ---- 2.5 POINTS --------------------- Train and test splits ---------------------
def split_data(x, y, testSize):
    '''
    Function for splitting the dataset
    Arguments:
    X - input predictors
    Y - labels
    testSize - proportion of the test data

    Returns:
    x_train - training data
    x_test  - test data
    y_train - training data labels
    y_test  - test data labels
    '''
    #-------------------------- Your code here -------------------------------
    return train_test_split(x, y, test_size=testSize)



#TO DO ------------ Simple Linear Regression using GD ---------------------
def my_linear_regression(x, y, lr, n_iterations):
    '''
    Function for linear regression based on gradient descent
    Arguments:
    x - input data
    y - labels
    lr - learning rate
    n_iterations - number of iterations
    Returns:
    beta - array of predictors coefficients
    '''
    #------------------ 4 Points ---- Your code here -------------------------------
    def cost_function(x_val, y_val, bettas):
        val = y_val - (bettas[0] + bettas[1] * x_val)
        return val ** 2
    
    def derive_slope_b0(x_val, y_val, bettas):
        return -2 * (y_val - (x_val * bettas[1] + bettas[0]))
    
    def derive_slope_b1(x_val, y_val, bettas):
        return -2 * x_val * (y_val - (x_val * bettas[1] + bettas[0]))
    
    def calc_mse(x, y, bettas):
        x_l = x.tolist()
        y_l = y.tolist()
        err = 0

        for i in range(len(x)):
            err += cost_function(x_l[i], y_l[i], bettas)
        return err / len(x)
    
    def update_bettas(x, y, bettas, rate):
        x_l = x.tolist()
        y_l = y.tolist()
        
        b0_upd_avg = 0
        b1_upd_avg = 0
        n = len(x)
        bettas_copy = list(bettas)
        
        for i in range(n):
            b0_upd_avg += derive_slope_b0(x_l[i], y_l[i], bettas)
            b1_upd_avg += derive_slope_b1(x_l[i], y_l[i], bettas)
        b0_upd_avg /= n
        b1_upd_avg /= n
        
        bettas_copy[0] -= rate * b0_upd_avg
        bettas_copy[1] -= rate * b1_upd_avg
        #print('old',bettas)
        #print('new',bettas_copy)
        
        return bettas_copy
        
    iterations = 0
    bettas = [0, 0] #b0, b1
    
    errors = []
    iter_vals = []
    while (iterations < n_iterations):
        err = calc_mse(x, y, bettas) # calculate mse on iteration
        bettas = update_bettas(x, y, bettas, lr)
        
        if iterations % 100 == 0:
            lr = 0.5 * lr
            errors.append(err)
            iter_vals.append(iterations)
        iterations += 1
        
        
    # --- BONUS 1 Point ----- construct a figure that plots the loss (squared loss) over time ---------
    '''
    xlabel: Iteration
    ylabel: Loss
    title: Training Loss
    '''
    #--------------------- Your code here -------------------------------
    plt.plot(iter_vals, errors, 'r')
    plt.title("Training loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.show()
    
    
    return bettas


#TO DO ----- BONUS 1 POINT --------- Plot prediction results against the training data -------------------
def plot_regression_line(x, y, beta):
    '''
    Function to plot regression line vs. the training data
    xlabel: Years of Experience
    ylabel: Salary
    title: Exp vs. Salary 

    Arguments:
    x - training data
    y - training labels
    beta - predictors coefficients
    '''
    #-------------------------- Your code here -------------------------------
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.title('Exp vs. Salary')
    
    # Plot train dots
    plt.scatter(x, y, c='b')
    
    # Plot line
    min_val = min(x)
    max_val = max(x)
    p = [beta[0] + min_val * beta[1], beta[0] + max_val * beta[1]]
    plt.plot([min_val, max_val], p, c='r')
    
    plt.show()


# In[27]:


#TO DO --- 2.5 POINTS --------------------- Prediction on test data ---------------------
def predict(x, beta):
    '''
    Function to predict values based on predictors coefficients
    Arguments:
    x - input test data
    beta - predictors coefficients
    
    Returns:
    y_predicted - array of predicted values
    '''
    #-------------------------- Your code here -------------------------------
    y_predicted = []
    for val in x:
        y_hat = beta[0] + val * beta[1]
        y_predicted.append(y_hat)
    return y_predicted


# In[28]:


# --------------------- Calculating RMSE ---------------------
def rmse_metric(actual, predicted):
    '''
    Function to calculate rmse using actual and predicted results
    

    Arguments:
    actual: ground truth
    predicted: prediction via model
    
    Returns:
    RMSE: root mean squared erros    
    '''
    sum_error = 0.0
    for i in range(len(actual)):
        print("Expected = %.0f, Predicted = %.0f" % (actual[i], predicted[i]))
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
    return np.sqrt(mean_error)


# --------------------- Importing the dataset ---------------------
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,0]
Y = dataset.iloc[:,1]


# --------------------- Splitting the dataset into training (70%) and test (30%) datasets ---------------------

X_train, X_test, y_train, y_test = split_data(X,Y, 0.3)


# ---------------------Estimating the models ---------------------
beta = my_linear_regression(X_train, y_train, 0.05, 1000)


# --------------------- Plotting regression line vs. the training data ---------------------
plot_regression_line(X_train.iloc[:], y_train.iloc[:], beta)


# --------------------- Prediction on test data and calculating RMSE ---------------------
X_test = np.array(X_test)
y_test = np.array(y_test)

pred = predict(X_test, beta)

rmse = rmse_metric(y_test, pred)
print('RMSE = %.3f'% (rmse))


# In[76]:


lrs = [0.01, 0.05, 0.001]
for lr in lrs:
    # --------------------- Importing the dataset ---------------------
    dataset = pd.read_csv('Salary_Data.csv')

    X = dataset.iloc[:,0]
    Y = dataset.iloc[:,1]


    # --------------------- Splitting the dataset into training (70%) and test (30%) datasets ---------------------

    X_train, X_test, y_train, y_test = split_data(X,Y, 0.3)


    # ---------------------Estimating the models ---------------------
    beta = my_linear_regression(X_train, y_train, lr, 1000)


    # --------------------- Plotting regression line vs. the training data ---------------------
    plot_regression_line(X_train.iloc[:], y_train.iloc[:], beta)


    # --------------------- Prediction on test data and calculating RMSE ---------------------
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    pred = predict(X_test, beta)

    rmse = rmse_metric(y_test, pred)
    print('RMSE = %.3f'% (rmse))



