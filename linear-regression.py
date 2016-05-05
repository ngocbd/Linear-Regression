import numpy as np


def linear_regression(x, y):
    x_trans = np.transpose(x)
    xT_x = np.dot(x_trans, x)
    A = np.dot(np.linalg.inv(xT_x), x_trans)
    theta = np.dot(A, y)
    return theta


