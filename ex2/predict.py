#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid

def predict(theta,X):
    m = X.shape[0]
    p = np.zeros(m).reshape(m,1)

    for i in range(0,m):
        if sigmoid.sigmoid(np.dot(X[i,:],theta)) >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p
