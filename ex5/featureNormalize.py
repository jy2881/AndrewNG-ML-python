#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def featureNormalize(X):
    m = X.shape[1]
    mu = np.mean(X,axis=0).reshape(1,m)
    X_norm = X-mu
    sigma = np.std(X_norm,axis=0,ddof=1).reshape(1,m)
    X_norm = X_norm/sigma

    return X_norm,mu,sigma
