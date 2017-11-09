#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

# 用来计算x的均值和方差
def estimateGaussian(X):
    m,n = np.shape(X)

    # zeros initial
    mu = np.zeros([n,1])
    sigma2 = np.zeros([n,1])

    # start
    mu = np.mean(X,axis=0).reshape(n,1)
    sigma2 = np.mean((X-mu.T)**2,axis=0).reshape(n,1)
    return mu,sigma2