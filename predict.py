import numpy as np


def predict(x,theta):
    y = np.dot(x.transpose(),theta)
    return y