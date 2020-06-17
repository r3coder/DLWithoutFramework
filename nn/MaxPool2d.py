import numpy as np
import cupy as cp

class MaxPool2d:
    def __init__(self, size=2):
        self.v = np.zeros(1)
        self.sz = size

    def Forward(self, x):
        if self.v.shape != (x.shape[0], x.shape[1], x.shape[2]//self.sz, x.shape[3]//self.sz):
            self.v = np.zeros((x.shape[0], x.shape[1], x.shape[2]//self.sz, x.shape[3]//self.sz))
        for ib in range(x.shape[0]):
            for ic in range(x.shape[1]):
                for i1 in range(x.shape[2]//self.sz):
                    for i2 in range(x.shape[3]//self.sz):
                        self.v[ib,ic,i1,i2] = np.amax(x[ib,ic,i1*self.sz:(i1+1)*self.sz, i2*self.sz:(i2+1)*self.sz])
        return self.v

    def Backward(self, x):
        return np.zeros((x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2))
        
