import numpy as np



def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x1 = np.exp(x)
    s = np.sum(x1)
    return x1 / s

def tanh(x):
    return np.tanh(x)