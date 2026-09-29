import math, copy
import numpy
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('./deeplearning.mplstyle')

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# Notes
    # Linear regression utilize the training data to fit parameters w, b by minimizing a measure
    # of the error between our predictions. That measure is the called the cost. That cost we cover
    # all of our training data. We then use gradient descent to repeat until convergence where parameters
    # w & b are updated simultaneously - meaning we calculate the partial derivatives for all the parameters
    # before updating any of the parameters


# Load the data set
x_train = np.array([1.0, 2.0]) #features
y_train = np.array([300.0, 500.0]) #target value

# Function to calculate the cost
def compute_cost(x, y, w, b):

    # x = array containing the input feature (size of house for example)
    # y = array containing the actual target (for example, price of house)
    # w = weight of the slope
    # b = bias of the incept parameter

    # finds the total number of the data points in dataset
    m = x.shape[0]

    # initialized to 0, acts as running accumulator for errors
    cost = 0

    # loops through each data point in m
    for i in range(m):

        # predicts the outcome using linear equation
        f_wb = w * x[i] + b

        # calculates the squared error by subtracting the actual value
        # predicted value. Squaring ensures the positive and negative
        # errors don't cancel each other out
        cost = cost + (f_wb - y [i]) ** 2

    # After adding up the errors divide the accumulated sum by 2m
    # dividing by m gives an average across the dataset
    # 2 cancels out laster when taking the derivative during gradient descent
    total_cost = 1 / (2 * m) * cost

    # if total cost is high - line is a poor fit
    # if total cost is close to zero - the line fits the data points accurately

    return total_cost


# Computing the gradient descent
def compute_gradient(x, y, w, b):

    # number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_dw = 0

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()