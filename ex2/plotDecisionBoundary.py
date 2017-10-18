#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt
import mapFeature

def plotData(x, y):
    f2 = plt.figure(2)

    idx_1 = np.where(y == 0)
    p1 = plt.scatter(x[idx_1, 0], x[idx_1, 1], marker='o', color='m', label='Not admitted', s=30)
    idx_2 = np.where(y == 1)
    p2 = plt.scatter(x[idx_2, 0], x[idx_2, 1], marker='+', color='c', label='Admitted', s=50)

    plt.xlabel('Exam 1 Score')
    plt.xlabel('Exam 2 Score')
    plt.legend(loc='upper right')
    # plt.show()
    return plt

def plotData2(x, y):
    f2 = plt.figure(2)

    idx_1 = np.where(y == 0)
    p1 = plt.scatter(x[idx_1, 0], x[idx_1, 1], marker='o', color='m', label='Microchip Test 1', s=30)
    idx_2 = np.where(y == 1)
    p2 = plt.scatter(x[idx_2, 0], x[idx_2, 1], marker='+', color='c', label='Microchip Test 2', s=50)

    plt.xlabel('y = 1')
    plt.xlabel('y = 2')
    plt.legend(loc='upper right')
    # plt.show()
    return plt

def plotDecisionBoundary(theta, X, y):
    f2 = plotData(X[:, 1:], y)
    # print(X[:, 1:])
    m, n = X.shape
    if n <= 3:
    # Only need 2 points to define a line, so choose two endpoints
        minVals = X[:, 1].min(0)-2
        maxVals = X[:, 1].max(0)+2
        plot_x = np.array([minVals, maxVals])
        plot_y = (-1 / theta[2]) * (plot_x.dot(theta[1]) + theta[0])
        f2.plot(plot_x, plot_y, label="Test Data", color='b')
        plt.show()

    else:
    # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature.mapFeature(u[i], v[j])* theta