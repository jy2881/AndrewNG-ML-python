#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.optimize as sop
import lrCostFunction

# 这个函数的目的是为了让逻辑回归可以分多个类
def oneVsAll(X,y,num_labels,xlambda):
    # 首先先初始化一些变量
    [m,n] = np.shape(X)
    all_theta = np.zeros([num_labels, n+1])
    X = np.c_[np.ones(m).reshape(m,1),X]

    #
    initial_theta = np.zeros([n+1,1])
    for c in range(1,num_labels+1):
        judgement = (y == c)
        all_theta[c:] = sop.minimize(fun=lrCostFunction.lrCostFunction,x0=initial_theta,args=(X,judgement, xlambda),method='TNC',jac=True,tol=1e-6).x
        # sop.fmin_cg(lrCostFunction.compute_cost_reg,fprime=lrCostFunction.compute_grad_reg,x0=initial_theta.T,args=(X,judgement,xlambda),disp = 0)
    return all_theta