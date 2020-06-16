import numpy as np


class Softmax:
    def __init__(self):
        self.v = np.zeros(1)
    
    def Forward(self, x):
        x1 = np.exp(x)
        s = np.sum(x1)
        self.v = x1 / s
        return self.v
    
    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
    
