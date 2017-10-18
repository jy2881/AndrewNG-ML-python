#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def mapFeature(x1, x2):
    degree = 6
    out = np.ones(x1.shape[0]).reshape(x1.shape[0],1)
    for i in range(degree):
        for j in range(i):
            newColumn = ((x1 ** (i-j))*(x2 ** j)).reshape(x1.shape[0],1)
            out = np.c_[out, newColumn]
    return out