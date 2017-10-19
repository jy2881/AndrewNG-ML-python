#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
# noinspection PyUnresolvedReferences
import warmUpExercise
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: ')
A = warmUpExercise.WarmUpExercise()

print('Program paused. Press enter to continue.\n')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.loadtxt("ex1data1.txt",delimiter = ",") # 数据导入，按逗号分隔
X = data[:,0]
y = data[:,1]
m = len(y)# 训练集的尺寸

f1 = plt.figure()
plt.scatter(X,y,marker="x",color="r")
plt.title('plot for X and y')
plt.show()

# =================== Part 3: Cost and Gradient descent ===================
one = np.ones(m)
X = np.vstack((one,X)) # 创建一个2*m的数组
theta = np.zeros((2,1)) # 创建一个2*1的数组

# 一些梯度下降参数设置
# noinspection PyUnresolvedReferences
import computeCost
iterations = 1500
alpha = 0.01
print("Testing the cost function ...")

# compute and display initial cost
J = computeCost.ComputeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %.2f'%J)
print('Expected cost value (approx) 32.07')

# further testing of the cost function
theta = np.array([[-1],[2]])
J = computeCost.ComputeCost(X, y, theta)
print('\nWith theta = [-1 ; 2]\nCost computed = %.2f'%J)
print('Expected cost value (approx) 54.24')

input('Program paused. Press enter to continue.')
print('Running Gradient Descent ...')

# noinspection PyUnresolvedReferences
import gradientDescent
theta = gradientDescent.GradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303  1.1664')

# Plot the linear fit
plt.plot(X[1,:], (np.matrix(X).T)*theta, '-')
plt.scatter(X[1,:],y,marker="x",color="r")
plt.xlabel('Training data');plt.ylabel('Linear regression')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = (np.matrix([1, 3.5]) *theta)*10000
print('For population = 35,000, we predict a profit of %d'%predict1)
predict2 = (np.matrix([1, 7]) * theta)*10000
print('For population = 70,000, we predict a profit of %d'%predict2)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
# Grid over which we will calculate J
# noinspection PyUnresolvedReferences
theta0_vals = np.arange(-10,10,0.2)
# noinspection PyUnresolvedReferences
theta1_vals = np.arange(-1, 4, 0.05)
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(1,len(theta0_vals)):
    for j in range(1,len(theta1_vals)):
        t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i,j] =computeCost.ComputeCost(X,y,t)
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
f2 = plt.figure()
ax = Axes3D(f2)
x_axes = theta0_vals
y_axes = theta1_vals
z_axes = J_vals
ax.plot_surface(x_axes,y_axes,z_axes)
ax.legend()
plt.show()

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100