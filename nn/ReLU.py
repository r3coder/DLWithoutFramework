import numpy as np

class ReLU:
    def __init__(self):
        # Cache
        self.v = np.zeros(1)
        # Property
        self.isTrainable = False
    
    def Forward(self, x):
        # if self.v.shape != x.shape:
        #     self.v = np.zeros(x.shape)
        self.v = np.maximum(x, 0)
        return self.v
    
    def Backward(self, x):
        # x = np.maximum(x, 0)
        # x[x > 0] = 1
        return x 
    
    def Info(self):
        return "[ ]ReLU %s"%(str(self.v.shape))
