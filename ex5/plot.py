#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt

def plotNormal(X,y):
    plt.scatter(X,y,color='r',marker='X',s=70,linewidths=0.01)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    return plt

def plotLC(X1,y1,X2,y2):
    plt.plot(X1,y1,color='y',label='Train')
    # plt.legend('Train')
    plt.plot(X2,y2,color='g',label='Cross Validation')
    # plt.legend('Cross Validation')
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0,13,0,150])
    plt.legend()
    return plt
