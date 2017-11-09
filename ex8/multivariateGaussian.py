#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

# 用来返回x每一个值的分布概率
def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)
    if (Sigma2.shape[1]-1)*(Sigma2.shape[0]-1)==0:
        Sigma2 = np.diag(Sigma2.T.tolist()[0])
    Sigma2 = np.matrix(Sigma2)
    X = X-mu.T
    p = ((2*np.pi)**(-0.5*k))*(np.linalg.det(Sigma2)**(-0.5))*np.exp(-0.5*np.sum(np.multiply(np.dot(X,np.linalg.inv(Sigma2)),X),axis=1))
    return p