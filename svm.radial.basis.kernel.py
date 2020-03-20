#!/usr/bin/env python3

##------------------------------------------------------------------------##
########################## Load libraries ##################################
##------------------------------------------------------------------------##
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from numpy.linalg import norm

##-------------------------------------------------------------------------##
############################ Functions ###################################### 
##-------------------------------------------------------------------------##

# Kernel definition 
'''
In: x: (n, 1) - 1D vector
	gamma: scalar value 
Out: scalar value 
'''
def kernel(x, gamma): 
	return np.exp(-gamma*(norm(x)**2))

# Gamma Matrix computation
'''
In: data: (n, None) - 2D vector - None: means that any value
	gamma: scalar value - metric of kernel function
	kernel : a function
Out: gam_matrix: (n,n) 
'''
def gam_matrix(data, gamma , kernel):
	nr_data = data.shape[0]
	gam_matrix = np.array([kernel(data[i]-data[j],gamma) \
				               	 for i in range(nr_data) \
					             for j in range(nr_data)]) \
					             .reshape(nr_data,nr_data)
	return gam_matrix

# optimization SVM problems
'''
In: data: (n,None) - 2D vector - None: means that any value - this value for gam_matrix
	gamma: scalar number
	kernel: function
	gam_matrix: a function to return a matrix
Out: alpha: 1D vector - positive multipliers
	 support_vector: data points nearest the hyperplane 
	 support_vector_y: label of support vectors 
Notes:
	Refer to cvxpy for formulation of QUADRATIC PROGRAMMING SOLVER
'''

def optmization_cv(data, gamma, kernel, gam_matrix, constantVal):
	# Quadratic of objective function part
	nr_data = data.shape[0] 
	P = np.outer(y,y)*gam_matrix(data, gamma , kernel)

	q = -1*np.ones(nr_data)

    # Constraints part
	G = -1*np.eye(nr_data, nr_data)
	N = np.eye(nr_data, nr_data)
	h = np.zeros(nr_data)
	k = constantVal*np.ones(nr_data)
   
	A = np.ones(nr_data)*y
	b = 0.0

	# Define and solve the CVXPY problem.
	x = cp.Variable(nr_data)
	problem = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T@x),
                 [G@x <= h,
				  N@x <= k,
                  A@x == b])
	problem.solve()

	print("The optimal value is: ", problem.value)
	print("A solution multipliers are: \n", x.value)

	# multipliers of dua-form SVM optimization 
	alpha = x.value
	alpha_positive = alpha[alpha>1.0e-8]
	support_vector = X[alpha>1.0e-8]
	support_vector_Y = y[alpha>1.0e-8]
	return alpha_positive, support_vector, support_vector_Y

# compute intercept
'''
In: alpha: 1D vector - positive multipliers from optimization_cv()
	sup_x: support vector machine - data points nearest the hyperplane
	sup_y: label of support vector machine - label of data points near the separator
 
Out: intercept: scalar
'''
def compute_b(alfa, sup_x, sup_y, gamma):
	# retrieve multipliers (alpha)
	# support vector x
	# support vector y	
	intercept=0.0
	for I_X_DO in range(len(sup_vector)):
		dummy = np.sum( (alfa[j]*sup_y[j]*kernel(sup_x[j]-sup_x[I_X_DO], gamma)) \
									for j in range(len(sup_x)) )
		dummy = sup_y[I_X_DO] - dummy
		intercept += dummy
	intercept = intercept/len(sup_x)
	return intercept


# Compute the accuracy of all training samples
'''
In: data: (m,None) - 2D vector - None means any value
Out: y_pred : (m,1) - 1D vector
'''
def prediction(data):
	y_pred=np.zeros(data.shape[0])
	for I_X_DO in range(data.shape[0]):
		y_pred[I_X_DO]=np.sign(np.sum(( alpha[j]*sup_vector_y[j]\
						                *kernel(sup_vector[j]-data[I_X_DO], gamma) \
							            for j in range(len(sup_vector)) )) + b)
	return y_pred

## ----------------------------------------------------------------------------------------------##
####################################### MAIN PROGRAM ##############################################   
## ----------------------------------------------------------------------------------------------##
# Constant value
constant_value=1
gamma=0.5

# Data input
'''
Data X must be in dimension (m,n) where:
	m: number of training samples
	n: number of featues, attributes or dimensions.
Label y must be a vector 1D with value -1 or 1.
'''
np.random.seed(0)
X = np.random.randn(26, 2)
Y_xor = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.array([1 if i else -1 for i in Y_xor ])
print('Actual Label: \n', y)

# support vectors x, y and multipliers
alpha, sup_vector, sup_vector_y = optmization_cv(X, gamma, kernel, gam_matrix, constant_value)


# compute intercept
b = compute_b(alpha, sup_vector, sup_vector_y, gamma)

# prediction 
y_prediction =  prediction(X)
print('Predicted Label: \n', y_prediction)

# prepare to plot contour
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
X_pred = np.c_[xx.ravel(), yy.ravel()]
Z = prediction(X_pred)
Z = Z.reshape(xx.shape)
contours = plt.contour(xx, yy, Z, levels=[0], colors = ['green'] ,linewidths=2, linestyles='dashed' )

# plot data in 2D
colors = np.where(y==1, 'r', 'b')
plt.scatter(X[:,0], X[:,1], c = colors)
plt.show()

