import numpy as np

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, padding_mode='zeros', lr = 0.001, dropout=True, dropout_rate = 0.5):

        # kernel and bias
        self.k = np.random.randn(out_channels, in_channels,kernel_size[0], kernel_size[1]) * np.sqrt(2/(out_channels*kernel_size[0]*kernel_size[1]))
        # self.k = np.random.uniform(-1, 1, (out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.b = np.zeros((out_channels, 1, 1))
        # Cache
        self.g = np.zeros(1)
        self.gk = np.zeros(1)
        self.v = np.zeros(1)
        self.c = np.zeros(1)
        # Property
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = lr
        self.isTrainable = True
        self.isDropout = dropout
        self.dropout_rate = dropout_rate


    def Forward(self, x):
        if self.v.shape != (x.shape[0], self.k.shape[0], x.shape[2], x.shape[3]):
            self.v = np.zeros((x.shape[0], self.k.shape[0], x.shape[2], x.shape[3]))
        if self.c.shape != x.shape:
            self.c = np.zeros(x.shape)
        self.c = x
        x_ = np.pad(x,((0,0),(0,0),(1,1),(1,1)), mode='constant', constant_values=(0))
        for ib in range(x.shape[0]):
            for io in range(self.k.shape[0]): # Output size
                for i1 in range(x.shape[2]):
                    for i2 in range(x.shape[3]):
                        self.v[ib,io,i1,i2] = np.sum(np.multiply(self.k[io,:], x_[ib,:,i1:i1+self.kernel_size[0],i2:i2+self.kernel_size[0]]))
        return self.v

    def Backward(self, x, x2 = ""):
        if self.g.shape != (x.shape[0], self.in_channels, x.shape[2], x.shape[3]):
            self.g = np.zeros((x.shape[0], self.in_channels, x.shape[2], x.shape[3]))
        self.gk = np.zeros(self.k.shape)
        if x2 != "":
            x = x + x2
        x_ = np.pad(x,((0,0),(0,0),(1,1),(1,1)), mode='constant', constant_values=(0))

        
        # Update gradients
        for ib in range(x.shape[0]):
            for ic in range(x.shape[1]):
                for ik in range(self.c.shape[1]):
                    for i1 in range(self.gk.shape[2]):
                        for i2 in range(self.gk.shape[3]):
                            self.gk[ic,ik,i1,i2] += np.sum(np.multiply(x_[ib,ic,i1:i1+x.shape[2],i2:i2+x.shape[3]], self.c[ib,ik,:,:]))
        # Dropout
        if self.isDropout:
            dx = np.zeros(self.out_channels*self.in_channels)
            dx[:int(self.out_channels*self.in_channels*self.dropout_rate)] = 1
            np.random.shuffle(dx)
            dx = np.reshape(dx,(self.out_channels,self.in_channels))
            d = np.zeros(self.k.shape)
            for io in range(self.out_channels):
                for ii in range(self.in_channels):
                    if dx[io, ii] == 1:
                        d[io, ii] = np.ones(self.kernel_size)
                    else:
                        d[io, ii] = np.zeros(self.kernel_size)
            self.k -= np.multiply(self.gk * self.lr / x.shape[0], d)
        else:
            self.k -= self.gk * self.lr / x.shape[0]


        for ib in range(x.shape[0]):
            for io in range(self.k.shape[1]): # Input size
                for i1 in range(x.shape[2]):
                    for i2 in range(x.shape[3]):
                        self.g[ib,io,i1,i2] = np.sum(np.multiply(self.k[:,io], x_[ib,:,i1:i1+self.kernel_size[0],i2:i2+self.kernel_size[0]]))
                        # self.v[ib,io,i1,i2] = np.sum(np.multiply(self.k[:,io,::-1,::-1], x_[ib,:,i1:i1+self.kernel_size[0],i2:i2+self.kernel_size[0]]))
        return self.g
    
    def Info(self):
        return "[T]Conv2d %dx%d %s"%(self.in_channels, self.out_channels, str(self.kernel_size))
