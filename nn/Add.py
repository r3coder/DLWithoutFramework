import numpy as np


class Add:
    def __init__(self, shape):
        # Gradient
        self.g = np.zeros(shape)
        # Output values
        self.v = np.zeros(shape)
    
    
    def Forward(self, x):
        self.v = x[0] + x[1]
        return self.v
    
    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass