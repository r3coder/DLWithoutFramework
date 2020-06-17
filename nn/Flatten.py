import numpy as np
import cupy as cp

class Flatten:
    def __init__(self):
        self.v = np.zeros(1)
        self.dim = (1)

    def Forward(self, x):
        if self.v.shape != (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]):
            self.v = np.zeros((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
            self.dim = (x.shape[1], x.shape[2], x.shape[3])
        for i in range(x.shape[0]):
            self.v[i] = np.reshape(x[i],(-1))
        return self.v

    def Backward(self, x):
        return np.zeros((x.shape[0], self.dim[0], self.dim[1], self.dim[2]))
        
