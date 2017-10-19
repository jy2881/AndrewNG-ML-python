#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    # You need to return the following variables correctly
    p = np.zeros([X.shape[0], 1])
    X = np.c_[(np.ones([m,1]),X)]

    temp = np.dot(X,all_theta.T)
    for i in range(0,m):
        maxVal = np.max(temp[i])
        p[i] = np.where(temp[i]==maxVal)
        p[i] += 1
    return p
