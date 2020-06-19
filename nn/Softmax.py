import numpy as np

class Softmax:
    def __init__(self):
        # Cache
        self.v = np.zeros(1)
        # Property
        self.isTrainable = False

    def Forward(self, x):
        if self.v.shape != x.shape:
            self.v = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_ = x[i]
            x_ = np.exp(x_ - np.amax(x_))
            self.v[i] = x_ / np.sum(x_)
        return self.v
    
    def Backward(self, x):
        return x
    
    def Info(self):
        return "[ ]Softmax %s"%(str(self.v.shape))
