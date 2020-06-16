import numpy as np
import cupy as cp

class MaxPool2d:
    def __init__(self, size=2):
        # self.v = np.zeros(1)
        self.v = 1
        self.sz = size

    def Forward(self, x):
        self.v = np.zeros((x.shape[0], x.shape[1]//self.sz, x.shape[2]//self.sz))
        # self.v = cp.zeros((x.shape[0], x.shape[1]//self.sz, x.shape[2]//self.sz))
        for i in range(x.shape[0]):
            for i1 in range(x.shape[1]//self.sz):
                for i2 in range(x.shape[2]//self.sz):
                    a = x[i][i1*self.sz:i1*self.sz+self.sz,i2*self.sz:i2*self.sz+self.sz]
                    self.v[i][i1][i2] = np.amax(a)
        return self.v

    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
        
