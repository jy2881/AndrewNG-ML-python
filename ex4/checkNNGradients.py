#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import nnCostFunction
import computeNumericalGradient
import debugInitializeWeights

def checkNNGradients(*a):
    if len(a)==0:
        xlambda=0
    else:
        xlambda = a
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights.debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights.debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    v = np.arange(1,m+1)
    X  = debugInitializeWeights.debugInitializeWeights(m, input_layer_size - 1)
    y  = 1.0 + (np.mod(v, num_labels)).T
    m1,n1 = np.shape(Theta1)
    m2,n2 = np.shape(Theta2)

    # Unroll parameters
    nn_params = np.r_[(Theta1.ravel().reshape(m1*n1,1),Theta2.ravel().reshape(m2*n2,1))]

    # Short hand for cost function
    def cost_func(p):
        return nnCostFunction.nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)

    cost,grad = cost_func(nn_params)
    numgrad = computeNumericalGradient.computeNumericalGradient(cost_func, nn_params)

    # Visually examine the two gradient computations.  The two columns you get should be very similar.
    print(np.c_[numgrad,grad])
    print('The above two columns should be very similar.(Left-Numerical Gradient, Right-Analytical Gradient)')


