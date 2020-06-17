import numpy as np
import math


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x1 = np.exp(x)
    s = np.sum(x1)
    return x1 / s

def tanh(x):
    return np.tanh(x)

def OneHotVector(ind, sz=10):
    res = np.zeros(sz)
    res[ind] = 1.0
    return res

def OneHotVectorBatch(l, sz=10):
    res = np.zeros((l.shape[0], sz))
    for i in range(l.shape[0]):
        res[i] = OneHotVector(l[i], sz)
    return res

def CrossEntropyLoss(r, a):
    return -math.log(r[a])

def CrossEntropyLossBatch(rl, al):
    res = np.zeros(rl.shape[0])
    for i in range(rl.shape[0]):
        res[i] = CrossEntropyLoss(rl[i], al[i])
    return np.sum(res)/rl.shape[0], res
