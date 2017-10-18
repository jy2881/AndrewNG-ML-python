#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
from sigmoid import *

def CostFunction(theta,X,y):
    m = len(y)
    J = 0
    grad = np.zeros(np.shape(theta))
    g = sigmoid(np.dot(X,theta))
    J = np.mean(((-y)*(np.log(g)))-((1-y)*(np.log(1-g))))
    grad = np.mean((g.reshape(m,1)-y.reshape(m,1))*X,axis=0)
    return J,grad