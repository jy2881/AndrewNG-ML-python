#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt
import estimateGaussian as eG
import multivariateGaussian as mG
import plot
import selectThreshold as sT

print('Visualizing example dataset for outlier detection.')
data = sco.loadmat("ex8data1.mat")
X,Xval,yval = data["X"],data["Xval"],data["yval"]

#  Visualize the example dataset
plt.scatter(X[:,0],X[:,1],marker="x",color='b',linewidths=0.01)
plt.axis([0,30,0,30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

## ================== Part 2: Estimate the dataset statistics ===================
print('Visualizing Gaussian fit.')
#  Estimate my and sigma2
mu,sigma2 = eG.estimateGaussian(X)

# Returns the density of the multivariate normal at each data point (row) of X
p = mG.multivariateGaussian(X, mu, sigma2)
#  Visualize the fit
plot.visualizeFit(X,  mu, sigma2)

## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set probabilities
#  given the estimated Gaussian distribution

pval = mG.multivariateGaussian(Xval, mu, sigma2)

epsilon,F1 = sT.selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: %e\n'%epsilon)
print('Best F1 on Cross Validation Set:  %f\n'%F1)
print('   (you should see a value epsilon of about 8.99e-05)\n')
print('   (you should see a Best F1 value of  0.875000)\n\n')

#  Find the outliers in the training set and plot the
outliers = np.where(p < epsilon)[0]

#  Draw a red circle around those outliers
x1=x2=[]
for i in outliers:
    x1 += X[i,0]
    x2 += X[i,1]
plt.scatter(x1, x2, marker='o', color='r',LineWidth=0.02)
plt.show()
