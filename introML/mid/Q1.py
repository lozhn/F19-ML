# This question is about simple Linear Regression using Gradient Descent - From scratch
# There are five functions out of which you will implement four functions: 
#      (a) split_data
#      (b) my_linear_regression
#      (c) plot_regression_line
#      (d) predict
# Finally, you are also asked to choose the best value for the learning rate using your implementation

# ---------------------------------------------------------------------------


#TO DO----1 POINT ----------- Use your code to answer the following question ------------
#which learning rate gives the best results: (a)0.05, (b)0.01, (c)0.001
#YOUR ANSWER:



#TO DO --------------------- Import all libraries here --------------------------
import numpy as np
import pandas as pd




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
    
    return 0



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
    
        
        
    # --- BONUS 1 Point ----- construct a figure that plots the loss (squared loss) over time ---------
    '''
    xlabel: Iteration
    ylabel: Loss
    title: Training Loss
    '''
    #--------------------- Your code here -------------------------------
    
    
    
    return 0




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
    
    return 0



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
beta = my_linear_regression(X_train, y_train, 0.01, 1000)


# --------------------- Plotting regression line vs. the training data ---------------------
plot_regression_line(X_train.iloc[:], y_train.iloc[:], beta)


# --------------------- Prediction on test data and calculating RMSE ---------------------
X_test = np.array(X_test)
y_test = np.array(y_test)

pred = predict(X_test, beta)

rmse = rmse_metric(y_test, pred)
print('RMSE = %.3f'% (rmse))
