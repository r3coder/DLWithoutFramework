import numpy as np
import cupy as cp
import nn.Functions as F

class RNN:
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=True, dropout_rate = 0.5):
        # Weight and bias
        self.w_u = np.random.randn(hidden_channels, in_channels) * np.sqrt(2/(hidden_channels+in_channels))
        self.w_w = np.random.randn(hidden_channels, hidden_channels) * np.sqrt(2/(hidden_channels+hidden_channels))
        self.w_v = np.random.randn(out_channels, hidden_channels) * np.sqrt(2/(hidden_channels+out_channels))
        self.b_h = np.zeros(hidden_channels)
        self.b_y = np.zeros(out_channels)
        
        # Gradients
        self.gw_u = np.zeros((hidden_channels, in_channels))
        self.gw_w = np.zeros((hidden_channels, hidden_channels))
        self.gw_v = np.zeros((out_channels, hidden_channels))
        # Hidden Unit
        self.h = np.zeros(hidden_channels)
        # Cache Values
        self.v = np.zeros(1)
        # Property
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.isTrainable = True


    def Forward(self, x):
        if self.v.shape != (x.shape[0], self.out_channels):
            self.v = np.zeros((x.shape[0], self.out_channels))
        self.h = np.zeros(self.hidden_channels)
        for ib in range(x.shape[0]):
            self.h = np.dot(self.w_u, x[ib][0])
            for ie in range(1,x.shape[1]):
                self.h = F.Tanh(np.dot(self.w_w, self.h) + np.dot(self.w_u, x[ib][ie])) + self.b_h
            self.v[ib] = np.dot(self.w_v, self.h) + self.b_y
        # return F.Softmax(self.v)
        return self.v

    def Backward(self, x):
        self.gw_u = np.zeros((self.hidden_channels, self.in_channels))
        self.gw_w = np.zeros((self.hidden_channels, self.hidden_channels))
        self.gw_v = np.zeros((self.out_channels, self.hidden_channels))
        
        self.g_h = np.zeros((x.shape[0], self.hidden_channels))
        # for i in range(x.shape[0]):
            # self.g_h[i] = np.dot(self.gw_v.T, x[i])
        """
        # Gradient for back
        grad = np.zeros(self.w.shape)
        for ib in range(x.shape[0]):
            grad += np.dot(x[ib].reshape((x.shape[1],1)), self.c[ib].reshape((1,self.c.shape[1]))) * self.lr
        # Dropout
        if self.isDropout:
            d = np.zeros(self.in_channels*self.out_channels)
            d[:int(self.in_channels*self.out_channels*self.dropout_rate)] = 1
            np.random.shuffle(d)
            d = np.reshape(d,(self.out_channels, self.in_channels))
            self.w -= np.multiply(grad / x.shape[0], d)
        else:
            self.w -= grad / x.shape[0]
        """
        return self.g_h
    

    def Info(self):
        return "RNN (%d, [%d], %d)"%(self.in_channels, self.hidden_channels, self.out_channels)

#n = RNN(300, 256, 300)
#d = np.random.randn(32,39,300)
#f = np.random.randn(300)

#print(n.Forward(d))
#print(n.Backward(f))
