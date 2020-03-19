#!/usr/bin/env python3
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from numpy.linalg import norm

# Constant value
constantVal=1
gamma=0.5

# Data input
np.random.seed(0)
X = np.random.randn(26, 2)
Y_xor = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.array([1 if i else -1 for i in Y_xor ])
print('Actual label \n', y)

# Kernel trick
number_training = np.size(X,0) # number of training samples or number of lagrange multiplier
K = np.array([np.exp(-gamma*(norm(X[i]-X[j])**2)) \
				   for i in range(number_training) \
					  for j in range(number_training)])\
						.reshape(number_training,number_training)
P = np.outer(y,y)*K

q = -1*np.ones(number_training)

G = -1*np.eye(number_training,number_training)
N = np.eye(number_training,number_training)
h = np.zeros(number_training)
k = constantVal*np.ones(number_training)

A = np.ones(number_training)*y
b = 0.0

# Define and solve the CVXPY problem.
x = cp.Variable(number_training)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T@x),
                 [G@x <= h,
				  N@x <= k,
                  A@x == b])
prob.solve()

print("The optimal value is", prob.value)
print("A solution x is\n", x.value)
alpha_lagrange = x.value
index_support_vector = alpha_lagrange>1.0e-8
alpha_lagrange_positive = alpha_lagrange[index_support_vector]
support_vector = X[index_support_vector]
support_vector_Y = y[index_support_vector]

# compute intercept (b)
intercept=0.0
for I_X_DO in range(len(support_vector)):
	dummy = np.sum(( alpha_lagrange_positive[j]*support_vector_Y[j]\
					*np.exp(-gamma*(norm(support_vector[j]-support_vector[I_X_DO])**2)) \
						for j in range(len(support_vector)) ))
	dummy = support_vector_Y[I_X_DO] - dummy
	intercept += dummy
b = intercept/len(support_vector)
print(b)	

# Compute the accuracy of all training samples
def prediction(X):
	y_pred=np.zeros(X.shape[0])
	for I_X_DO in range(X.shape[0]):
		y_pred[I_X_DO]=np.sign(np.sum(( alpha_lagrange_positive[j]\
					      *support_vector_Y[j]\
						     *np.exp(-gamma*(norm(support_vector[j]-X[I_X_DO])**2)) \
							    for j in range(len(support_vector)) )) +b )
	return y_pred

y_prediction =  prediction(X)
print('prediction', y_prediction)

# prepare to plot contour
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
X_pred = np.c_[xx.ravel(), yy.ravel()]
Z = prediction(X_pred)
Z = Z.reshape(xx.shape)
contours = plt.contour(xx, yy, Z, levels=[0], colors = ['green'] ,linewidths=2, linestyles='dashed' )

# plot data in 2D
col = np.where(y==1, 'r', 'b')
plt.scatter(X[:,0], X[:,1], c = col)
plt.show()

