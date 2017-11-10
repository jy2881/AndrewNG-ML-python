#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def normalizeRatings(Y,R):
    m, n = np.shape(Y)
    Ymean = np.zeros([m, 1])
    Ynorm = np.zeros(np.shape(Y))
    for i in range(0,m):
        idx = np.where(R[i,:] == 1)
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]

    return Ynorm,Ymean