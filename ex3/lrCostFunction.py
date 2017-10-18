#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid

def lrCostFunction(theta, X, y, xlambda):
    m = len(y)
    n = theta.shape[0]
    theta = theta.reshape(n,1)
    # initialized parameters
    J = 0
    grad = np.zeros(n)

    # used for normalization
    matrixEye = np.eye(len(theta))
    matrixEye[0,0]=0
    g = sigmoid.sigmoid(np.dot(X,theta)).reshape(m,1)
    J = (1/m)*(np.dot(-1*(y.T),np.log(g))-np.dot((1.0-y.T),np.log(1-g)))+np.sum(theta[1:]**2)*xlambda/2/m
    grad = (np.dot(X.T,(g-y))+np.dot(xlambda*matrixEye,theta).reshape(n,1))/m
    return J,grad

def compute_cost_reg(theta, X, y, xlambda):
    m = len(y)
    n = theta.shape[0]
    theta = theta.reshape(n,1)
    # initialized parameters
    J = 0
    grad = np.zeros(n)

    # used for normalization
    matrixEye = np.eye(len(theta))
    matrixEye[0,0]=0
    g = sigmoid.sigmoid(np.dot(X,theta)).reshape(m,1)
    J = (1/m)*(np.dot(-1*(y.T),np.log(g))-np.dot((1.0-y.T),np.log(1-g)))+np.sum(theta[1:]**2)*xlambda/2/m
    grad = (np.dot(X.T,(g-y))+np.dot(xlambda*matrixEye,theta).reshape(n,1))/m
    return J

def compute_grad_reg(theta, X, y, xlambda):
    m = len(y)
    n = theta.shape[0]
    theta = theta.reshape(n,1)
    # initialized parameters
    J = 0
    grad = np.zeros(n)

    # used for normalization
    matrixEye = np.eye(len(theta))
    matrixEye[0,0]=0
    g = sigmoid.sigmoid(np.dot(X,theta)).reshape(m,1)
    J = (1/m)*(np.dot(-1*(y.T),np.log(g))-np.dot((1.0-y.T),np.log(1-g)))+np.sum(theta[1:]**2)*xlambda/2/m
    grad = (np.dot(X.T,(g-y))+np.dot(xlambda*matrixEye,theta).reshape(n,1))/m
    return grad