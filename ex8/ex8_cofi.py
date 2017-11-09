#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt
import cofiCostFunc as cCF
import checkCostFunction

print('Loading movie ratings dataset.\n\n')
ex8_movies = sco.loadmat("ex8_movies.mat")
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R,Y = ex8_movies["R"],ex8_movies["Y"]
#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): %f / 5\n\n'%(np.sum(Y[0,:])/np.sum(R[0,:])))

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering. To help you debug your cost function,
#  we have included set of weights that we trained on that. Specifically,
#  you should complete the code in cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
ex8_movieParams = sco.loadmat('ex8_movieParams.mat')
X,Theta,num_users,num_movies,num_features = ex8_movieParams["X"],ex8_movieParams["Theta"],ex8_movieParams["num_users"],\
                                            ex8_movieParams["num_movies"],ex8_movieParams["num_features"]
#  Reduce the data set size so that this runs faster
num_users,num_movies,num_features = 4,5,3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

#  Evaluate cost function
nn_params = np.r_[(X.ravel().reshape(num_movies*num_features,1),Theta.ravel().reshape(num_users*num_features,1))]
J,grad = cCF.cofiCostFunc(nn_params, Y, R, num_users, num_movies,num_features, 0)
print('Cost at loaded parameters: %f\n(this value should be about 22.22)\n'%J)
input('\nProgram paused. Press enter to continue.\n')

## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement the collaborative filtering gradient function.
#  Specifically, you should complete the code in cofiCostFunc.m to return the grad argument.

print('\nChecking Gradients (without regularization) ... \n')
#  Check gradients by running checkNNGradients
checkCostFunction.checkCostFunction(1.5)

input('\nProgram paused. Press enter to continue.\n')
