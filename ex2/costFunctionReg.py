#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid

def costFunctionReg(theta,X,y,xlambda):
    # initialize some useful values
    m = len(y)
    n = theta.shape[0]
    J = 0
    y = y.reshape(m,1)
    # start
    grad = np.zeros(theta.shape[0])
    g = sigmoid.sigmoid(np.dot(X,theta))
    J = (1/m)*((np.dot(-y.T,np.log(g)))-np.dot((1-y.T),np.log(1-g))+np.sum(theta[1:]**2)*xlambda/2/m)
    matrix = np.eye(n)
    matrix[0,0] = 0
    # grad = (np.dot(X.T,(g-y))+np.dot(xlambda*matrix,theta))/m
    return J,grad