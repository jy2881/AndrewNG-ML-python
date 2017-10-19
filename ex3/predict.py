#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid

def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros([m,1])

    # The processing of forward-propagating
    X = np.c_[(np.ones([m,1]),X)]  # X is the input layer, so x equals a1
    a2 = sigmoid.sigmoid(np.dot(X,Theta1.T)) # FP from X to a2
    a2 = np.c_[(np.ones([a2.shape[0],1]),a2)] # Add a new column for bias
    a3 = sigmoid.sigmoid(np.dot(a2,Theta2.T)) # FP to the output layer

    for k in range(0,m):
        maxVal = np.max(a3[k,:])
        p[k] = np.where(a3[k,:]==maxVal)
    return p