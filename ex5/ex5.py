#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt
import plot
import linearRegCostFunction as lRCF
import trainLinearReg as tLR

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. The following code will load
#  the dataset into your environment and plot the data.

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1. You will have X, y, Xval, yval, Xtest, ytest in your environment
data = sco.loadmat("ex5data1.mat")
X,y,Xval,yval,Xtest = data["X"],data["y"],data["Xval"],data["yval"],data["Xtest"]
m = X.shape[0]

# Plot training data
plt = plot.plotNormal(X,y)
plt.show()
input('Program paused. Press enter to continue.\n')

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear regression.

theta = np.array([1,1]).reshape(2,1)
J,grad = lRCF.linearRegCostFunction(theta,np.c_[np.ones([m,1]),X],y,1)

print('Cost at theta = [1,1]: %f(this value should be about 303.993192)\n'%J)
input('Program paused. Press enter to continue.\n')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear regression.
print('Gradient at theta = [1 ; 1]:  [%f; %f](this value should be about [-15.303016; 598.250744])\n'%(grad[0],grad[1]))
input('Program paused. Press enter to continue.\n')

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the trainLinearReg function will use your
#  cost function to train regularized linear regression.

#  Write Up Note: The data is non-linear, so this will not give a great fit.

#  Train linear regression with lambda = 0
xlambda = 0
theta = tLR.trainLinearReg(np.c_[(np.ones([m,1]),X)],y,xlambda)

#  Plot fit over the data
plt = plot.plotNormal(X,y)
y_pred = np.dot(np.c_[(np.ones([m, 1]),X)],theta)
plt.plot(X, y_pred, color='b')
plt.show()

input('Program paused. Press enter to continue.\n')

