#-*-cding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import displayData
import nnCostFunction
import sigmoidGradient
import randInitializeWeights
import checkNNGradients
import scipy.optimize as sop


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
J,grad = nnCostFunction.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, xlambda)

print('Cost at parameters (loaded from ex4weights): %.6f(this value should be about 0.287629)'%J)
input('Program paused. Press enter to continue.')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now continue to implement the regularization with the cost.

print('Checking Cost Function (w/ Regularization) ... ')

# Weight regularization parameter (we set this to 1 here).
xlambda = 1
J,grad = nnCostFunction.regularization(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)

print('Cost at parameters (loaded from ex4weights): %f(this value should be about 0.383770)'%J)
input('Program paused. Press enter to continue.')

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first implement the gradient for the sigmoid function.
#  You should complete the code in the sigmoidGradient.m file.

print('Evaluating sigmoid gradient...')

g = sigmoidGradient.sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
for i in g:
    print(i,end=" ")
input('\nProgram paused. Press enter to continue.')

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two layer neural network that classifies digits.
#  You will start by implementing a function to initialize the weights of the neural network(randInitializeWeights.m)

print('Initializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights.randInitializeWeights(hidden_layer_size, num_labels)
m1,n1 = np.shape(initial_Theta1)
m2,n2 = np.shape(initial_Theta2)

# Unroll parameters
initial_nn_params = np.r_[(initial_Theta1.ravel().reshape(m1*n1,1),initial_Theta2.ravel().reshape(m2*n2,1))]

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the backpropagation algorithm
#  for the neural network. You should add to the code you've written in nnCostFunction.m to
#  return the partial derivatives of the parameters.

print('Checking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients.checkNNGradients()

input('Program paused. Press enter to continue.')

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now continue to
#  implement the regularization with the cost and gradient.

print('Checking Backpropagation (w/ Regularization) ...\n ')

# Check gradients by running checkNNGradients
xlambda = 3
checkNNGradients.checkNNGradients(xlambda)

# Also output the costFunction debugging values
debug_J,degub_grad  = nnCostFunction.regularization(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)
print('Cost at (fixed) debugging parameters (w/ lambda = %.2f): %.8f '
      '\n(for lambda = 3, this value should be about 0.576051)'%(xlambda,debug_J))

input('Program paused. Press enter to continue.')

## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural network.
#  To train your neural network, we will now use "fmincg", which is a function which works similarly to "fminunc".
#  Recall that these advanced optimizers are able to train our cost functions efficiently as long as we provide them
#  with the gradient computations.

print('Training Neural Network... ')

# You should also try different values of lambda
xlambda = 1

# Now, costFunction is a function that takes in only one argument (the neural network parameters)
nn_params = sop.minimize(fun=nnCostFunction.regularization, x0=initial_nn_params,args=(input_layer_size,hidden_layer_size,num_labels, X, y, xlambda),method='TNC',jac=True,tol=1e-6).x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))

input('Program paused. Press enter to continue...\n')

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by displaying the hidden units to see
#  what features they are capturing in the data.

print('Visualizing Neural Network... \n')
displayData.displayData(Theta1[:,1:],20)
input('Program paused. Press enter to continue.\n')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict the labels.
#  You will now implement the "predict" function to use the neural network to predict the labels of the training set.
#  This lets you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
acc = np.mean(np.mean((pred==y)*100))
print('Training Set Accuracy: %.4f'%acc)



