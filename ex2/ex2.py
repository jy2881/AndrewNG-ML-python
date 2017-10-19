#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import plotDecisionBoundary

data = np.loadtxt("ex2data1.txt",delimiter = ",")
X = data[:, 0:2]
y = data[:, 2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
f1 = plotDecisionBoundary.plotData(X, y)
plt.show()

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient for logistic regression.
#  You neeed to complete the code in costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = np.shape(X)

# Add intercept term to x and X_test
X = np.c_[(np.ones(m),X)]
# Initialize fitting parameters
initial_theta = np.zeros(n+1).reshape(n+1,1)

# Compute and display initial cost and gradient
import costFunction
[cost, grad] = costFunction.CostFunction(initial_theta, X, y)
print('Cost at initial theta (zeros): %.3f\nExpected cost (approx): 0.693\nGradient at initial theta (zeros):'%cost)
for grad_print in grad:
    print('%.4f'%grad_print)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24],[0.2],[0.2]])
[cost, grad] = costFunction.CostFunction(test_theta, X, y)
print('Cost at test theta: %.3f\nExpected cost (approx): 0.218\nGradient at test theta:'%cost)
for grad_print in grad:
    print('%.4f'%grad_print)
print('Expected gradients (approx):0.043\n 2.566\n 2.647')
input('Program paused. Press enter to continue.')

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the optimal parameters theta.
# 梯度下降法http://www.cnblogs.com/LeftNotEasy/archive/2010/12/05/mathmatic_in_machine_learning_1_regression_and_gradient_descent.html
# 雅克布矩阵http://jacoxu.com/jacobian%E7%9F%A9%E9%98%B5%E5%92%8Chessian%E7%9F%A9%E9%98%B5/

result = opt.minimize(fun=costFunction.CostFunction,x0=initial_theta,args=(X,y),options={'maxiter':400},method='TNC',jac='Gradient',callback=callable)
print('Cost at theta found by fminunc: %.3f\nExpected cost (approx): 0.203\ntheta:\n'%result.fun)
for theta_print in result.x:
    print("%.3f"%theta_print)
print('Expected theta (approx):\n-25.161\n 0.206\n 0.201')

# Plot Boundary
plotDecisionBoundary.plotDecisionBoundary(result.x, X, y)

## ============== Part 4: Predict and Accuracies ==============
""" After learning the parameters, you'll like to use it to predict the outcomes
    on unseen data. In this part, you will use the logistic regression model
    to predict the probability that a student with score 45 on exam 1 and score
    85 on exam 2 will be admitted.

    Furthermore, you will compute the training and test set accuracies of our model."""
import sigmoid
import predict
prob = sigmoid.sigmoid(np.dot(result.x.reshape(1,3),np.array([[1],[45],[85]])))
print('For a student with scores 45 and 85, we predict an admission probability of %.4f\nExpected value: 0.775 +/- 0.002'%prob)
p = predict.predict(result.x, X)

accuracy = np.mean(p == y.reshape(m,1))*100
print('Train Accuracy: %.2f'%accuracy)
print('Expected accuracy (approx): 89.0')