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
        self.v = np.zeros((out_channels, 1, 1))
        # values
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def Forward(self, x):
        self.v = np.zeros((self.k.shape[0], x.shape[1], x.shape[2]))
        x_ = np.zeros((x.shape[0], x.shape[1]+self.padding*2, x.shape[2]+self.padding*2))
        for i in range(x.shape[0]):
            x_ = np.pad(x[i], (self.padding, self.padding), mode='constant', constant_values=(0))
        # x_ = cp.array(x_)
        for io in range(self.k.shape[0]): # Output size
            for ii in range(self.k.shape[1]): # Input size
                for i1 in range(x.shape[1]):
                    for i2 in range(x.shape[2]):
                        # self.v[io][i1][i2] += cp.sum(cp.multiply(self.k[io][ii], x_[i1:i1+self.kernel_size[0],i2:i2+self.kernel_size[0]]))
                        self.v[io][i1][i2] += np.sum(np.multiply(self.k[io][ii], x_[i1:i1+self.kernel_size[0],i2:i2+self.kernel_size[0]]))
        # return cp.asnumpy(self.v)
        return self.v

    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
