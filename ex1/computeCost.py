#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def ComputeCost(X,y,theta):
    m = len(y)
    # noinspection PyUnresolvedReferences
    J = (0.5*((np.square(np.matrix(X).T*theta-np.matrix(y).T)).sum())/m)
    # 因为numpy的数组计算默认为broadcasting，所以先转为matrix再运算，也可以用np.dot()方法
    # 我觉得最好还是用矩阵进行运算，数组得随时注意broadcasting，还得转置来转置去。
    # J =(0.5*((np.square(np.dot(X.T,theta).T-y)).sum())/m)
    return J
