#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt
import plot

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot the data.

print('Loading and Visualizing Data ...')

# Load from ex6data1,You will have X, y in your environment:
data = sco.loadmat('ex6data1.mat')
X,y = data["X"],data['y']
print(np.shape(X),np.shape(y))

# Plot training data
f1 = plot.plotData(X,y)
f1.show()
input('Program paused. Press enter to continue.')

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the decision boundary learned.

# Load from ex6data1. You will have X, y in your environment:
data1 = sco.loadmat('ex6data1.mat')

# fprintf('\nTraining Linear SVM ...\n')
#
# % You should try to change the C value below and see how the decision
# % boundary varies (e.g., try C = 1000)
# C = 1;
# model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
# visualizeBoundaryLinear(X, y, model);
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;