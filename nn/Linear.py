import numpy as np
import cupy as cp

class Linear:

    def __init__(self, in_channels, out_channels, bias=True):
        # weight
        # self.w = np.sqrt(np.random.randn(out_channels, in_channels, dtype=cp.float32))
        self.w = np.random.randn(out_channels, in_channels) * np.sqrt(2/(out_channels+in_channels))
        # Bias
        self.b = np.zeros(out_channels)
        # Gradient
        self.g = np.zeros((out_channels, in_channels))
        # Output values
        self.v = np.zeros(out_channels)

    def Forward(self, x):
        if self.v.shape != (x.shape[0], self.w.shape[0]):
            self.v = np.zeros((x.shape[0], self.w.shape[0]))
        for i in range(x.shape[0]):
            self.v[i] = np.dot(self.w, x[i])
        return self.v

    def Backward(self, x):
        return np.zeros((x.shape[0],self.v.shape[1]))
