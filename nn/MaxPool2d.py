import numpy as np

class MaxPool2d:
    def __init__(self, size=2):
        # Cache
        self.v = np.zeros(1)
        self.g = np.zeros(1)
        self.ind = np.zeros(1)
        # Property
        self.sz = size  # Pooling size
        self.isTrainable = False # True if it is trainable

    def Forward(self, x):
        # if self.v.shape != (x.shape[0], x.shape[1], x.shape[2]//self.sz, x.shape[3]//self.sz):
        self.v = np.zeros((x.shape[0], x.shape[1], x.shape[2]//self.sz, x.shape[3]//self.sz))
        
        # if self.ind.shape != (x.shape[0], x.shape[1], x.shape[2]//self.sz, x.shape[3]//self.sz, 2):
        self.ind = np.zeros((x.shape[0], x.shape[1], x.shape[2]//self.sz, x.shape[3]//self.sz, 2), dtype=int)
        
        for ib in range(x.shape[0]):
            for ic in range(x.shape[1]):
                for i1 in range(x.shape[2]//self.sz):
                    for i2 in range(x.shape[3]//self.sz):
                        w = x[ib,ic,i1*self.sz:(i1+1)*self.sz, i2*self.sz:(i2+1)*self.sz]
                        indx = np.where(w == w.max())
                        self.ind[ib,ic,i1,i2,0] = indx[0][0]
                        self.ind[ib,ic,i1,i2,1] = indx[1][0]
                        self.v[ib,ic,i1,i2] = w[self.ind[ib,ic,i1,i2,0], self.ind[ib,ic,i1,i2,1]]
                        # self.v[ib,ic,i1,i2] = np.amax(x[ib,ic,i1*self.sz:(i1+1)*self.sz, i2*self.sz:(i2+1)*self.sz])
        return self.v

    def Backward(self, x):
        self.g = np.zeros((x.shape[0], x.shape[1], x.shape[2]*self.sz, x.shape[3]*self.sz))
        for ib in range(x.shape[0]):
            for ic in range(x.shape[1]):
                for i1 in range(x.shape[2]):
                    for i2 in range(x.shape[3]):
                        self.g[ib,ic,i1*2+self.ind[ib,ic,i1,i2,0],i2*2+self.ind[ib,ic,i1,i2,1]] = x[ib,ic,i1,i2]
        return self.g
        
    def Info(self):
        return "[ ]MaxPool2d Size=%d"%(self.sz)
