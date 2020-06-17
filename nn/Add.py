import numpy as np
import cupy as cp

class Add:
    def __init__(self):
        # Gradient
        self.g = np.zeros(1)
        # Output values
        self.v = np.zeros(1)
    
    
    def Forward(self, x):
        self.v = x[0] + x[1]
        return self.v
    
    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
