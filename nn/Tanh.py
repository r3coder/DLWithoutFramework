import numpy as np

class Tanh:
    def __init__(self):
        # Cache
        self.v = np.zeros(1)
        # Property
        self.isTrainable = False

    def Forward(self, x):
        # if self.v.shape != x.shape:
            # self.v = np.zeros(x.shape)
        self.v = np.tanh(x)
        return self.v
    
    def Backward(self, x):
        return x
        # return 1.0 - np.square(np.tanh(x))
    
    def Info(self):
        return "[ ]Tanh %s"%(str(self.v.shape))
