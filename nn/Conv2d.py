import numpy as np
import cupy as cp

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'):

        # kernel
        # self.k = cp.random.uniform(-1, 1, (out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.k = np.random.uniform(-1, 1, (out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # Bias
        self.b = np.zeros((out_channels, 1, 1))
        # Gradient
        self.g = np.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # values
        self.v = np.zeros(1)
        # values
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def Forward(self, x):
        if self.v.shape != (x.shape[0], self.k.shape[0], x.shape[2], x.shape[3]):
            self.v = np.zeros((x.shape[0], self.k.shape[0], x.shape[2], x.shape[3]))
        x_ = np.pad(x,((0,0),(0,0),(1,1),(1,1)), mode='constant', constant_values=(0))
        for ib in range(x.shape[0]):
            for io in range(self.k.shape[0]): # Output size
                for i1 in range(x.shape[2]):
                    for i2 in range(x.shape[3]):
                        self.v[ib,io,i1,i2] = np.sum(np.multiply(self.k[io,:], x_[ib,:,i1:i1+self.kernel_size[0],i2:i2+self.kernel_size[0]]))
        return self.v

    def Backward(self, x):
        return np.zeros((x.shape[0], self.in_channels, x.shape[2], x.shape[3]))
