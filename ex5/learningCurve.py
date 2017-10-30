#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import trainLinearReg as tLR
import linearRegCostFunction as lRCF

# 用来绘制学习曲线，来看看误差随着训练样本数量增大而变化的情况
def learningCurve(X, y, Xval, yval, xlambda):
    m = X.shape[0]

    # zero initial
    error_train = np.zeros([m,1])
    error_val = np.zeros([m,1])

    # start
    for n in range(0,m):
        theta = tLR.trainLinearReg(X[:n+1,:],y[:n+1],xlambda)
        error_train[n],grad = lRCF.linearRegCostFunction(theta,X[:n+1,:],y[:n+1],0)
        error_val[n],grad = lRCF.linearRegCostFunction(theta,Xval,yval,0)

    return error_train,error_val