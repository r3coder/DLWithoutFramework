import numpy as np

class Flatten:
    def __init__(self):
        # Cache values
        self.v = np.zeros(1)
        self.g = np.zeros(1)
        # Property
        self.dim = (1)
        self.isTrainable = False
        
    def Forward(self, x):
        if self.v.shape != (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]):
            self.v = np.zeros((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        self.dim = (x.shape[1], x.shape[2], x.shape[3])
        for i in range(x.shape[0]):
            self.v[i] = np.reshape(x[i],(-1))
        return self.v

    def Backward(self, x):
        if self.g.shape != (x.shape[0], self.dim[0], self.dim[1], self.dim[2]):
            self.g = np.zeros((x.shape[0], self.dim[0], self.dim[1], self.dim[2]))
        for i in range(x.shape[0]):
            self.g[i] = x[i].reshape((self.dim[0], self.dim[1], self.dim[2]))
        return self.g
    
    def Info(self):
        return "[ ]Flatten %s -> %d"%(str(self.dim), self.dim[0]*self.dim[1]*self.dim[2])
