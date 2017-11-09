#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np

def cofiCostFunc(params, Y, R, num_users, num_movies,num_features, xlambda):
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    # zero initial
    J = 0
    X_grad = np.zeros(np.shape(X))
    Theta_grad = np.zeros(np.shape(Theta))

    # start
    J = np.sum(R*((np.dot(X,Theta.T)-Y)**2))/2
    X_grad = np.dot((R*(np.dot(X,Theta.T)-Y)),Theta)
    Theta_grad = np.dot((R*(np.dot(X,Theta.T)-Y)).T,X)

    grad = np.r_[(X_grad.ravel().reshape(num_movies*num_features,1),Theta_grad.ravel().reshape(num_users*num_features,1))]
    return J,grad