#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def debugInitializeWeights(fan_out, fan_in):
    v = np.arange(1,fan_out*(fan_in+1)+1)
    W = np.sin(v).reshape(fan_out,1+fan_in) / 10
    return W