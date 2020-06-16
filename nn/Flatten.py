import numpy as np
import cupy as cp

class Flatten:
    def __init__(self):
        self.v = cp.zeros(1)

    def Forward(self, x):
        self.v = cp.reshape(x, (-1))
        return self.v

    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
        
