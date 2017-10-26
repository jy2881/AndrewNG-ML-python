#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import sigmoid
import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, xlambda):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    m,n = np.shape(X)

    # initial output
    J = 0
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

    ## ==========================  Part 1:  ===========================
    # Feedforward the neural network and return the cost in the variable J without regularization
    ylabel = np.zeros([num_labels, m]) # Transform y from a 1-d vector into a 10-d matrix
    for i in range(0,m):
        ylabel[int(y[i]-1.0),i] = 1

    # process of FP
    X = np.c_[np.ones(m).reshape(m,1),X]
    z2 = np.dot(X,Theta1.T)
    a2 = sigmoid.sigmoid(z2)
    z3 = np.c_[np.ones(m).reshape(m,1),a2]
    z3 = np.dot(z3,Theta2.T)
    a3 = sigmoid.sigmoid(z3) # a3 is the h(x), it is the output layer.

    for i in range(0,m):
        J = J - (np.dot(np.log(a3[i,:].reshape(1,num_labels)),ylabel[:,i].reshape(num_labels,1))+\
            np.dot(np.log(1.0-a3[i,:].reshape(1,num_labels)),(1.0-ylabel[:,i].reshape(num_labels,1))))
    J = J/m

    ## ==================== Part 2: Compute the gradients ======================
    Delta1,Delta2 = np.zeros(np.shape(Theta1)),np.zeros(np.shape(Theta2))

    # BP
    for t in range(0,m):
        delta3 = (a3[t,:] - ylabel[:,t]).reshape(num_labels,1) # the error between y and a3
        delta2 = np.dot(Theta2.T,delta3) * sigmoidGradient.sigmoidGradient(np.c_[np.ones(m).reshape(m,1),a2][t,:].reshape(hidden_layer_size+1,1)) # the error generated in the hidden layer

        Delta1 = Delta1 + np.dot(delta2[1:],X[t,:].reshape(1,n+1))
        Delta2 = Delta2 + np.dot(delta3,np.c_[np.ones(m).reshape(m,1),a2][t,:].reshape(1,hidden_layer_size+1))

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    m1,n1 = np.shape(Theta1_grad)
    m2,n2 = np.shape(Theta2_grad)

    grad = np.r_[(Theta1_grad.ravel().reshape(m1*n1,1),Theta2_grad.ravel().reshape(m2*n2,1))]

    return J,grad

def regularization(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, xlambda):
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    m,n = np.shape(X)

    J,grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, xlambda)
    J = J + xlambda/2/m * (np.sum(np.sum(np.square(Theta1[:,1:]))) + np.sum(np.sum(np.square(Theta2[:,1:]))))

    ylabel = np.zeros([num_labels, m]) # Transform y from a 1-d vector into a 10-d matrix
    for i in range(0,m):
        ylabel[int(y[i]-1.0),i] = 1

    # process of FP
    X = np.c_[np.ones(m).reshape(m,1),X]
    z2 = np.dot(X,Theta1.T)
    a2 = sigmoid.sigmoid(z2)
    z3 = np.c_[np.ones(m).reshape(m,1),a2]
    z3 = np.dot(z3,Theta2.T)
    a3 = sigmoid.sigmoid(z3) # a3 is the h(x), it is the output layer.

    Delta1,Delta2 = np.zeros(np.shape(Theta1)),np.zeros(np.shape(Theta2))

    for t in range(0,m):
        delta3 = (a3[t,:] - ylabel[:,t]).reshape(num_labels,1) # the error between y and a3
        delta2 = np.dot(Theta2.T,delta3) * sigmoidGradient.sigmoidGradient(np.c_[np.ones(m).reshape(m,1),a2][t,:].reshape(hidden_layer_size+1,1)) # the error generated in the hidden layer

        Delta1 = Delta1 + np.dot(delta2[1:],X[t,:].reshape(1,n+1))
        Delta2 = Delta2 + np.dot(delta3,np.c_[np.ones(m).reshape(m,1),a2][t,:].reshape(1,hidden_layer_size+1))

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    m1,n1 = np.shape(Theta1_grad)
    m2,n2 = np.shape(Theta2_grad)

    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + xlambda/m*Theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + xlambda/m*Theta2[:, 1:]
    grad = np.r_[(Theta1_grad.ravel().reshape(m1*n1,1),Theta2_grad.ravel().reshape(m2*n2,1))]

    return J,grad