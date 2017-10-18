#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import plotDecisionBoundary
import mapFeature
import costFunctionReg

data = np.loadtxt(r"E:\ml\exercise\w3\python\ex2data2.txt",delimiter = ",")
X = data[:, 0:2]
y = data[:, 2]

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
f1 = plotDecisionBoundary.plotData2(X, y)
plt.show()

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not linearly separable.
#  However, you would still like to use logistic regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add polynomial features
#  to our data matrix (similar to polynomial regression).

X = mapFeature.mapFeature(X[:,0], X[:,1])
initial_theta = (np.zeros(X.shape[1])).reshape(X.shape[1],1)
xlambda = 1
cost, grad = costFunctionReg.costFunctionReg(initial_theta, X, y, xlambda)
print('Cost at initial theta (zeros): %.3f\nExpected cost (approx): 0.693\nGradient at initial theta (zeros) - first five values only:'%cost)
print(grad[0:5])
print('Expected gradients (approx) - first five values only:\n0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
input('Program paused. Press enter to continue.')

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and see how regularization affects the decision coundart

#  Try the following values of lambda (0, 1, 10, 100).
initial_theta = (np.zeros(X.shape[1])).reshape(X.shape[1],1)
xlambda = 1
# Optimize
result = opt.minimize(fun=costFunctionReg.costFunctionReg,x0=initial_theta,args=(X,y,xlambda),method='TNC',jac='Gradient',callback=callable)
theta = result.x
# Plot
plotDecisionBoundary.plotDecisionBoundary(theta, X, y)
plt.title(('lambda = %g', xlambda))