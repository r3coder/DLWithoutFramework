import numpy as np

class Linear:
    def __init__(self, in_channels, out_channels, lr=0.0001, dropout=True, dropout_ratio = 0.5):
        # weight and bias
        self.w = np.random.randn(out_channels, in_channels) * np.sqrt(2/(out_channels+in_channels))
        self.b = np.zeros(out_channels)
        # Cache Values
        self.v = np.zeros(1)
        self.c = np.zeros(1)
        # Property
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.isTrainable = True
        self.isDropout = True
        self.dropout_ratio = 0.5

    def Forward(self, x):
        self.c = x
        if self.v.shape != (x.shape[0], self.out_channels):
            self.v = np.zeros((x.shape[0], self.out_channels))
        for i in range(x.shape[0]):
            self.v[i] = np.dot(self.w, x[i]) + self.b
        return self.v

    def Backward(self, x):
        # SGD
        self.g = np.zeros((x.shape[0], self.in_channels))
        for i in range(x.shape[0]):
            self.g[i] = np.dot(self.w.T, x[i])
        
        # Gradient for back
        grad = np.zeros(self.w.shape)
        for ib in range(x.shape[0]):
            grad += np.dot(x[ib].reshape((x.shape[1],1)), self.c[ib].reshape((1,self.c.shape[1]))) * self.lr
        # Dropout
        if self.isDropout:
            d = np.zeros(self.in_channels*self.out_channels)
            d[:int(self.in_channels*self.out_channels*self.dropout_ratio)] = 1
            np.random.shuffle(d)
            d = np.reshape(d,(self.out_channels, self.in_channels))
            self.w -= np.multiply(grad / x.shape[0], d)
        else:
            self.w -= grad / x.shape[0]

        return self.g

    def Info(self):
        return "[T]Linear (%d, %d)"%(self.w.shape[1], self.w.shape[0])
