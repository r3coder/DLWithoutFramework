import numpy as np
import cupy as cp

class ReLU:
    def __init__(self):
        self.v = np.zeros(1)
    
    def Forward(self, x):
        self.v = np.maximum(x, 0)
        return self.v
    
    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
    
