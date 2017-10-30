#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.optimize as sop
import linearRegCostFunction as lRCF

def trainLinearReg(X,y,xlambda):
    # Initialize Theta
    initial_theta = np.zeros([X.shape[1],1]).reshape(2,1)

    # Start
    theta = sop.minimize(fun=lRCF.linearRegCostFunction,x0=initial_theta,args=(X,y,xlambda),method='TNC',jac=True,tol=1e-6).x
    return theta
