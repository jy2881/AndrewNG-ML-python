#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import computeNumericalGradient as cNG
import cofiCostFunc as cCF

def checkCostFunction(*xlambda):
    if len(xlambda)==0:
        xlambda=0

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t,Theta_t.T)
    Y[np.where(np.random.rand(Y.shape[0],Y.shape[1]) > 0.5)] = 0
    R = np.zeros(np.shape(Y))
    R[np.where(Y!=0)] = 1

    # Run Gradient Checking
    X = np.random.randn(X_t.shape[0],X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0],Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    # cost function
    def cost_func(p):
        return cCF.cofiCostFunc(p,Y,R,num_users,num_movies,num_features,xlambda)

    nn_params = np.r_[(X.ravel().reshape(num_movies*num_features,1),Theta.ravel().reshape(num_users*num_features,1))]
    numgrad = cNG.computeNumericalGradient(cost_func, nn_params)

    cost, grad = cCF.cofiCostFunc(nn_params,Y,R,num_users,num_movies,num_features,xlambda)

    # Visually examine the two gradient computations.  The two columns you get should be very similar.
    print(np.c_[numgrad,grad])
    print('The above two columns you get should be very similar.\n(Left: Numerical Gradient\tRight: Analytical Gradient)')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('''If your cost function implementation is correct, then
         the relative difference will be small (less than 1e-9).
         Relative Difference: %.16f);'''%diff)