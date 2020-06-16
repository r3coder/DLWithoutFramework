import numpy as np


class Concat:
    def __init__(self):
        self.v = np.zeros(1)
        pass
    
    def Forward(self, x):
        self.v = np.concatenate(x)
        return self.v
    
    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
