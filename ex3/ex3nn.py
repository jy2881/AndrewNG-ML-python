#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sio
import displayData
import predict

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex3data1.mat')
X,y = data['X'],data['y']
m,n = X.shape[0],X.shape[1]

# Randomly select 100 data points to display
rand_indices = np.random.choice(m,100)
sel = X[rand_indices]
example_width = round(np.sqrt(sel.shape[1]))
displayData.displayData(sel,example_width)

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weight = sio.loadmat('ex3weights.mat')
Theta1,Theta2 = weight['Theta1'],weight['Theta2']

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict the labels.
#  You will now implement the "predict" function to use the neural network to predict the labels of the training set.
#  This lets you compute the training set accuracy.
pred = predict.predict(Theta1, Theta2, X)
acc = np.mean(pred == y) * 100
print('Training Set Accuracy: %.2f'%acc)

input('Program paused. Press enter to continue.')

# randomly permute example
rp = np.random.choice(m,m)

for i in rp:
    print('Displaying Example Image')
    displayData.displayData(X[i,:].reshape(1,n),20) # Manually type the width 20

    pred = predict.predict(Theta1,Theta2,X[i,:].reshape(1,n))
    print('Neural Network Prediction: %d'%int(pred[0]))

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:');
    if s == 'q':
        break

