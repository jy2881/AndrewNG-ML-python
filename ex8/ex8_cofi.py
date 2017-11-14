#-*-coding:utf-8 -*-
__author__ = 'Jy2881'

import numpy as np
import scipy.io as sco
import matplotlib.pyplot as plt
import cofiCostFunc as cCF
import checkCostFunction
import loadMovieList
import math
import normalizeRatings as nR
import scipy.optimize as sop

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

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for collaborative filtering.
#  You can implement it by adding the cost of regularization to the original cost computation.

#  Evaluate cost function
nn_params = np.r_[(X.ravel().reshape(num_movies*num_features,1),Theta.ravel().reshape(num_users*num_features,1))]
J,grad = cCF.cofiCostFunc(nn_params,Y,R,num_users,num_movies,num_features,1.5)

print('Cost at loaded parameters (lambda = 1.5): %f(this value should be about 31.34)\n'%J)
input('\nProgram paused. Press enter to continue.\n')

## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement regularization for the gradient.

print('\nChecking Gradients (with regularization) ... \n')
#  Check gradients by running checkNNGradients
checkCostFunction.checkCostFunction(1.5)
input('\nProgram paused. Press enter to continue.\n')

## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first add ratings that correspond
#  to a new user that we just observed. This part of the code will also allow you to put in your
#  own ratings for the movies in our dataset!

movieList = loadMovieList.loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros([1682,1])

# Check the file movie_idx.txt for id of each movie in our dataset
#For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[1-1] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[98-1] = 2

# We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings[7-1] = 3
my_ratings[12-1]= 5
my_ratings[54-1] = 4
my_ratings[64-1] = 5
my_ratings[66-1] = 3
my_ratings[69-1] = 5
my_ratings[183-1] = 4
my_ratings[226-1] = 5
my_ratings[355-1] = 5

print('\nNew user ratings:\n')
for i in range(0,len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s'%(my_ratings[i],movieList[i]),end="")

input('\nProgram paused. Press enter to continue.\n')

## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating dataset of 1682 movies and 943 users

print('\nTraining collaborative filtering...\n')

#  Load data
ex8_movies = sco.loadmat('ex8_movies.mat')
R,Y = ex8_movies["R"],ex8_movies["Y"]
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

#  Add our own ratings to the data matrix
Y = np.c_[(my_ratings,Y)]
R = np.c_[(np.ceil(my_ratings/5),R)]

#  Normalize Ratings
Ynorm, Ymean = nR.normalizeRatings(Y, R)

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters=np.r_[(X.ravel().reshape(num_movies*num_features,1),Theta.ravel().reshape(num_users*num_features,1))]
xlambda = 10

theta = sop.minimize(fun=cCF.cofiCostFunc,x0=initial_parameters,
                     args=(Ynorm,R,num_users,num_movies,num_features,xlambda),
                     method='TNC',jac=True,tol=1e-6,options={'maxiter':100, "disp":True}).x

# Unfold the returned theta back into U and W
X = theta[:num_movies*num_features].reshape(num_movies,num_features)
Theta = theta[num_movies*num_features:].reshape(num_users,num_features)

input('Recommender system learning completed.\nProgram paused. Press enter to continue.\n')

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing the predictions matrix.

p = np.dot(X,Theta.T)
my_predictions = p[:,0].reshape(num_movies,1) + Ymean.reshape(num_movies,1)
my_predictions = my_predictions[:,0]

movieList = loadMovieList.loadMovieList()

my_pred_sort = np.sort(-my_predictions)
my_ix = np.argsort(-my_predictions)

print('\nTop recommendations for you:\n')
for i in range(0,10):
    j = int(my_ix[i])
    print('Predicting rating %.1f for movie %s'%(my_predictions[j],movieList[j]),end="")
# print("my_predictions",my_predictions[:20])
# print("p",p[:20,0])

# print('\nOriginal ratings provided:\n')
# for i in range(0,len(my_ratings)):
#     if my_ratings[i] > 0:
#         print('Rated %d for %s'%(my_predictions[j],movieList[j]),end="")
