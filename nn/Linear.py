import numpy as np


class Linear:

    def __init__(self, in_channels, out_channels, bias=True):
        # weight
        self.w = np.random.uniform(-0.2, 0.2, (out_channels, in_channels))
        # Bias
        self.b = np.zeros(out_channels)
        # Gradient
        self.g = np.zeros((out_channels, in_channels))
        # Output values
        self.v = np.zeros(out_channels)

    def Forward(self, x):
        self.v = np.dot(self.w, x)
        return self.v

    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass