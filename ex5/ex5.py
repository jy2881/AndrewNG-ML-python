#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt
import plot
import linearRegCostFunction as lRCF
import trainLinearReg as tLR
import learningCurve as lC
import featureNormalize
import polyFeatures

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

# =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#  Write Up Note: Since the model is underfitting the data, we expect to see a graph with "high bias" -- Figure 3 in ex5.pdf

xlambda = 0
error_train,error_val = lC.learningCurve(np.c_[(np.ones([m,1]),X)],y,np.c_[np.ones([Xval.shape[0],1]),Xval],yval,xlambda)

# plot the learning curve
plt = plot.plotLC(np.arange(1,m+1), error_train, np.arange(1,m+1), error_val)
plt.show()

# print the contrast between training error and CV error
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(0,m):
    print(' \t%d\t\t%.4f\t%.4f'%(i,error_train[i],error_val[i]))
input('Program paused. Press enter to continue.\n')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers

p = 8
# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures.polyFeatures(X, p)
X_poly,mu,sigma = featureNormalize.featureNormalize(X_poly)  # Normalize
X_poly = np.c_[(np.ones([m,1]), X_poly)]                  # Add Ones row

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures.polyFeatures(Xtest, p)
X_poly_test = X_poly_test-mu
X_poly_test = X_poly_test/sigma
X_poly_test = np.c_[(np.ones([X_poly_test.shape[0], 1]), X_poly_test)]         # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures.polyFeatures(Xval, p)
X_poly_val = X_poly_val-mu
X_poly_val = X_poly_val-sigma
X_poly_val = np.c_[(np.ones([X_poly_val.shape[0], 1]), X_poly_val)]           # Add Ones

print('Normalized Training Example 1:')
for i in X_poly[0,:]:
    print(i,end=" ")

input('\nProgram paused. Press enter to continue.\n')

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple values of xlambda.
#  The code below runs polynomial regression with xlambda = 0. You should try running the code with different
#  values of xlambda to see how the fit and learning curve change.

xlambda = 0
theta = tLR.trainLinearReg(X_poly, y, xlambda)

# Plot training data and fit
f1 = plot.plotFit(X,y,mu,sigma,theta,p,xlambda)
f1.show()

error_train, error_val = lC.learningCurve(X_poly,y,X_poly_val,yval,xlambda)
f2 = plot.plotLC(np.arange(1,m+1),error_train,np.arange(1,m+1),error_val)
f2.title('Polynomial Regression Learning Curve (xlambda = %.2f)'%xlambda)
f2.show()
print('Polynomial Regression (lambda = %.2f)'%xlambda)
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(0,m):
    print('  \t%d\t\t%.4f\t%.4f'%(i,error_train[i],error_val[i]))

input('Program paused. Press enter to continue.\n')