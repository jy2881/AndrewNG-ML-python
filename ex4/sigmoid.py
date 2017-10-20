#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def sigmoid(z):
    g = np.zeros(np.shape(z))
    g = 1/(1+np.exp(-z))
    return g