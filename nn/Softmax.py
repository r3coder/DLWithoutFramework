import numpy as np
import cupy as cp

class Softmax:
    def __init__(self):
        self.v = np.zeros(1)
    
    def Forward(self, x):
        if self.v.shape != x.shape:
            self.v = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_ = x[i]
            x_ = np.exp(x_ - np.amax(x_))
            self.v[i] = x_ / np.sum(x_)
        return self.v
    
    def Backward(self, x):
        return x
    
