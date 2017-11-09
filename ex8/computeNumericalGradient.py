#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def computeNumericalGradient(J,theta):
    numgrad = np.zeros(np.shape(theta))
    perturb = np.zeros(np.shape(theta))
    e = 1e-4
    for p in range(0,np.size(theta)):
        perturb[p] = e
        loss1,para = J(theta - perturb)
        loss2,para = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad