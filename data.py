#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


'''
In: num_points : numper of data points
Out: X_new: (num_points,2)
	 y_new: (num_points,1) : label of X_new
	 X: the separator of two classes in X_new labeled by y_new
'''
def dataset(num_points):
	np.random.seed(0)
	num_training  = int(num_points/2) 

	# create a function order four 
	x1 = np.linspace(-2,2,num_training)
	x2 = np.power(x1,4) 
	x2 = 2*x2/np.max(x2)
	X = np.c_[x1,x2]
 
	# add some random to seperate two classes
	X_random = np.random.rand(num_training,2) + 0.05
	X_up = X + X_random*np.array([[0, 0.5] for _ in range(num_training)])
	y_up = np.array([1 for _ in range(num_training)])
	X_down = X + X_random*np.array([[0, -0.7] for _ in range(num_training)])
	y_down = np.array([-1 for _ in range(num_training)])
	X_new = np.r_[X_up, X_down]
	y_new = np.r_[y_up, y_down]

	return X, X_new, y_new

'''
# Verify the data points created
X, X_new, y_new = dataset(30)
plt.plot(X[:,0], X[:,1], c ='r', marker = '.')
plt.scatter(X_new[:15,0], X_new[:15,1], c ='b', marker = 'x')
plt.scatter(X_new[15:,0], X_new[15:,1], c ='g', marker = 'o')
plt.show()
'''
