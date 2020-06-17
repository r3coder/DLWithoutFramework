import numpy as np
import cupy as cp

class Tanh:
    def __init__(self):
        self.v = np.zeros(1)
    
    def Forward(self, x):
        self.v = np.tanh(x)
        return self.v
    
    def Backward(self, x):
        return x
    
