#-*-cding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import displayData
import nnCostFunction


# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10

# Load Training Data
print('Loading and Visualizing Data ...')

data = sco.loadmat('ex4data1.mat')
X,y = data['X'],data['y']
m,n = np.shape(X)

# Randomly select 100 data points to display
randomVal = np.random.choice(m,m)[:100]
sel = X[randomVal]

example_width = round(np.sqrt(sel.shape[1]))
displayData.displayData(sel,example_width)

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weight = sco.loadmat('ex4weights.mat')
Theta1, Theta2 = weight['Theta1'],weight['Theta2']
m1,n1 = np.shape(Theta1)
m2,n2 = np.shape(Theta2)

# Unroll parameters
nn_params = np.r_[(Theta1.ravel().reshape(m1*n1,1),Theta2.ravel().reshape(m2*n2,1))]

## ================ Part 3: Compute Cost (Feedforward) ================
"""To the neural network, you should first start by implementing the feedforward part of the neural network
   that returns the cost only. You should complete the code in nnCostFunction.m to return cost.
   After implementing the feedforward to compute the cost, you can verify that your implementation
   is correct by verifying that you get the same cost as us for the fixed debugging parameters.

   We suggest implementing the feedforward cost *without* regularization
   first so that it will be easier for you to debug. Later, in part 4, you
   will get to implement the regularized cost."""

print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
xlambda = 0
J = nnCostFunction.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, xlambda)

print('Cost at parameters (loaded from ex4weights): %.6f(this value should be about 0.287629)'%J)
input('Program paused. Press enter to continue.')

