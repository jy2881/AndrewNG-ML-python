#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def selectThreshold(yval,pval):
    bestEpsilon = 0
    bestF1 = 0
    f1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval),np.max(pval),stepsize):
        predictions = (pval<epsilon)
        if np.sum(predictions==1) != 0:
            prec = np.sum((yval+predictions) == 2)/np.sum(predictions == 1)
            rec = np.sum((yval+predictions) == 2)/np.sum(yval == 1)

            f1 = 2*prec*rec/(prec+rec)

        if f1 > bestF1:
            bestF1 = f1
            bestEpsilon = epsilon

    return bestEpsilon,bestF1