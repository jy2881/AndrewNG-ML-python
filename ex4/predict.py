#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid

def predict(Theta1, Theta2, X):
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros([m, 1])

    h1 = sigmoid.sigmoid(np.dot(np.c_[(np.ones([m,1]),X)],Theta1.T))
    h2 = sigmoid.sigmoid(np.dot(np.c_[(np.ones([m,1]),h1)],Theta2.T))

    for k in range(0,m):
        maxVal = np.max(h2[k,:])
        p[k] = np.where(h2[k,:]==maxVal)
        p[k] += 1
    return p