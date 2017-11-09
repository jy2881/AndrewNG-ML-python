#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import matplotlib.pyplot as plt
import multivariateGaussian as mG

def visualizeFit(X,  mu, sigma2):
    X1,X2 = np.meshgrid(np.arange(0,35,0.5),np.arange(0,35,0.5))
    m,n = np.shape(X1)
    a = X1.ravel().reshape(m*n,1)

    Z = mG.multivariateGaussian(np.c_[(X1.ravel().reshape(m*n,1),X2.ravel().reshape(m*n,1))],mu,sigma2)
    Z = Z.reshape(np.shape(X1))

    # plot points in buttom layer
    plt.scatter(X[:,0],X[:,1],marker="x",color='b',linewidths=0.01)
    plt.axis([0,30,0,30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

    # plot countours; Do not plot if there are infinities
    if (np.sum(np.isinf(Z)) == 0):
        C = plt.contour(X1, X2, Z, 10, colors = 'black', linewidth = 0.5)
        # C.contour(X1, X2, Z, 10**np.arange(-20,0,3))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
