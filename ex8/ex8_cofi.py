#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt

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
