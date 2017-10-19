#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
# noinspection PyUnresolvedReferences
import computeCost

def GradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for iter in range(1,num_iters):
        # X是一个二维数组，y是个一维数组。
        theta = theta - (alpha/m)*(np.matrix(X)*(np.matrix(X).T*theta-np.matrix(y).T))
        J_history[iter] = computeCost.ComputeCost(X,y,theta)

    return theta