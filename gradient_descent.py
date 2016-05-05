import numpy as np


def compute_cost(x,y,theta):
    m = np.shape(x)[0]
    pred = np.dot(x,theta)
    cost = sum((pred-y)**2)/(2*m)
    return cost


def feature_normalize(x):
	mu = np.mean(x,axis=0)
	sigma = np.std(x,axis=0)
	normalized_features = (x-mu)/sigma if sigma !=0 else x
	return normalized_features

def gradient_descent(x,y,initial_theta,learning_rate,num_iterations):
    m = np.shape(x)[0]
    theta = initial_theta
    cost = compute_cost(x,y,theta)
    for i in xrange(num_iterations):
		predictions = np.dot(x,theta)
		delta = np.dot(x.transpose(),(predictions-y))/m
		theta = theta - learning_rate*delta
    return theta