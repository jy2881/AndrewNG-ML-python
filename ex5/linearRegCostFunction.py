#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def linearRegCostFunction(theta,X,y,xlambda):
    m = len(y)
    n = X.shape[1]

    # zero initial
    J = 0
    grad = np.zeros(np.shape(theta)).reshape(2,1)
    theta = theta.reshape(2,1)

    # start here
    J = 1/2/m*np.sum(np.square(np.dot(X,theta)-y)) + xlambda/2/m*np.sum(np.square(theta[1:,:]))
    grad = 1/m*(np.dot(X.T,(np.dot(X,theta)-y)))
    grad[1:,:] = grad[1:,:] + xlambda/m*theta[1:,:]
    return J,grad