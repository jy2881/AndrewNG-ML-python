#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def randInitializeWeights(L_in, L_out):
    W = np.random.uniform(low=-0.12,high=0.12,size=(L_out,L_in+1))
    return W