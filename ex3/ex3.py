#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sio
import displayData
import lrCostFunction
import oneVsAll
import predictOneVsAll

input_layer_size  = 400
num_labels = 10

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

data = sio.loadmat("ex3data1.mat")
X = data['X']
y=data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.choice(m,100)
sel = X[rand_indices]
example_width = round(np.sqrt(sel.shape[1]))
displayData.displayData(sel,example_width)

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression code from the last exercise.
#  You task here is to make sure that your regularized logistic regression implementation is vectorized.
#  After that, you will implement one-vs-all classification for the handwritten digit dataset.

# Test case for lrCostFunction
print('Testing lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[(np.ones([5,1]),np.arange(1,16).reshape(3,5).T/10)]
y_t = np.array([[1],[0],[1],[0],[1]]) >= 0.5
lambda_t = 3
[J,grad] = lrCostFunction.lrCostFunction(theta_t, X_t, y_t, lambda_t)
#[J,grad] = lrCostFunction.lrCostFunction(initial_theta, X, y, xlambda)
print('Cost: %.7f\nExpected cost: 2.534819\nGradients:'%J)
for grad_i in grad:
    print(grad_i)
print('Expected gradients:\n0.146561\n -0.548558\n 0.724722\n 1.398003')

input('Program paused. Press enter to continue.')

## ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression...')

xlambda = 0.1
all_theta = oneVsAll.oneVsAll(X, y, num_labels, xlambda)
input('Program paused. Press enter to continue.')

## ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll.predictOneVsAll(all_theta, X)
acc = np.mean(np.mean((pred == y))*100)
print('Training Set Accuracy: %.4f'%acc)
