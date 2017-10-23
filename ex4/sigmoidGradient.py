#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid

def sigmoidGradient(z):
    g = np.zeros(np.shape(z)) # zero initial
    g = sigmoid.sigmoid(z)*(1.0-sigmoid.sigmoid(z))
    return g