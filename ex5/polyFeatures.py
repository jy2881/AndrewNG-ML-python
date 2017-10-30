#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def polyFeatures(X, p):
    # zero initial
    X_poly = np.zeros([np.size(X), p])

    # start
    for i in range(0,p):
        X_poly[:,i] = X[:,0]**(i+1)

    return X_poly