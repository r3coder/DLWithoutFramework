import numpy as np
import cupy as cp

class Concat:
    def __init__(self):
        self.v = np.zeros(1)
        self.channels = (0)
    
    def Forward(self, x):
        if self.v.shape != (x[0].shape[0], x[0].shape[1]+x[1].shape[1], x[0].shape[2], x[0].shape[3]):
            self.v = np.zeros((x[0].shape[0], x[0].shape[1]+x[1].shape[1], x[0].shape[2], x[0].shape[3]))
            self.channels = (x[0].shape[1], x[1].shape[1])
        for i in range(x[0].shape[0]):
            self.v[i] = np.concatenate((x[0][i], x[1][i]))
        return self.v

    def Backward(self, x):
        return x[:,:self.channels[0],:,:], x[:,self.channels[0]:,:,:]
