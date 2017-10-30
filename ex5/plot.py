#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt

def plotNormal(X,y):
    plt.scatter(X,y,color='r',marker='X',s=70,linewidths=0.01)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    return plt
