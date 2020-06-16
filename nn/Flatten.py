import numpy as np


class Flatten:
    def __init__(self):
        self.v = np.zeros(1)

    def Forward(self, x):
        self.v = np.reshape(x, (-1))
        return self.v

    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
        
