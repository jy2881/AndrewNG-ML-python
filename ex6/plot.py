#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import matplotlib.pyplot as plt
import numpy as np

def plotData(X,y):
    f1 = plt.figure(2)

    idx_1 = np.where(y == 0)
    p1 = plt.scatter(X[idx_1, 0], X[idx_1, 1], marker='o', color='m', label='Not admitted', s=30)
    idx_2 = np.where(y == 1)
    p2 = plt.scatter(X[idx_2, 0], X[idx_2, 1], marker='+', color='c', label='Admitted', s=50)

    plt.xlabel('Exam 1 Score')
    plt.xlabel('Exam 2 Score')
    plt.legend(loc='upper right')
    return plt