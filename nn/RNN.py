import numpy as np
import cupy as cp

class RNN:
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Weight and bias
        self.w_u = np.random.randn(hidden_channels, in_channels) * np.sqrt(2/(hidden_channels+in_channels))
        self.w_w = np.random.randn(hidden_channels, hidden_channels) * np.sqrt(2/(hidden_channels+hidden_channels))
        self.w_v = np.random.randn(out_channels, hidden_channels) * np.sqrt(2/(hidden_channels+out_channels))
        self.b_h = np.zeros(hidden_channels)
        self.b_y = np.zeros(out_channels)

        # Hidden Unit
        self.h = np.zeros(hidden_channels)
        # Cache Values
        self.v = np.zeros(1)
        # Property
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.isTrainable = True


    def Forward(self, x):
        if self.v.shape != (x.shape[0], self.out_channels):
            self.v = np.zeros((x.shape[0], self.out_channels))
        self.h = np.zeros(hidden_channels)
        for ib in range(x.shape[0]):
            self.h = np.dot(self.w_u, x[ib][0])
            for ie in range(1,x.shape[1]):
                self.h = np.tanh(np.dot(self.w_w, self.h) + np.dot(self.w_u, x[ib][ie]) + self.b_h
            self.v[ib] = np.dot(self.w_v, self.h) + self.b_y
        return self.v

    def Backward(self, x):
        pass
    

    def Info(self):
        return "RNN (%d, [%d], %d)"%(self.in_channels, self.hidden_channels, self.out_channels)
