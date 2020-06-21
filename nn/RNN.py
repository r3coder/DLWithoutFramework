import numpy as np
import cupy as cp
import nn.Functions as F

class RNN:
    def __init__(self, in_channels, hidden_channels, out_channels, lr=0.001, dropout=True, dropout_rate = 0.5):
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
        self.v_h = np.zeros(1)
        self.v_i = np.zeros(1)
        # Property
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.isTrainable = True
        self.sequenceSize = -1
        self.batchSize = -1
        self.memory = 20
        self.lr = lr
    
    def Forward(self, x):
        self.batchSize = x.shape[0]
        self.sequenceSize = x.shape[1]
        self.memory = self.sequenceSize # Adjust this later, maybe
        if self.v.shape != (x.shape[0], self.out_channels):
            self.v = np.zeros((x.shape[0], self.out_channels))
        self.v_i = x
        if self.v_h.shape != (x.shape[0],self.sequenceSize, self.hidden_channels):
            self.v_h = np.zeros((x.shape[0],self.sequenceSize, self.hidden_channels))
        
        self.h = np.zeros((self.hidden_channels))
        for ib in range(x.shape[0]):
            for ie in range(x.shape[1]):
                k = np.dot(self.w_u, x[ib,ie])
                self.h = F.Tanh(np.dot(self.w_w, self.h) + np.dot(self.w_u, x[ib,ie])) + self.b_h
                self.v_h[ib, ie] = np.copy(self.h)
                self.v[ib] = np.dot(self.w_v, self.h) + self.b_y

        return self.v

    def Backward(self, x=np.zeros(1)):
        self.gw_u = np.zeros((self.hidden_channels, self.in_channels))
        self.gw_w = np.zeros((self.hidden_channels, self.hidden_channels))
        self.gw_v = np.zeros((self.out_channels, self.hidden_channels))
        
        self.g_h = np.zeros((self.batchSize, self.hidden_channels))
        # Get error 
        for ib in range(self.batchSize):
            for ie in range(1,self.sequenceSize-1):
                err = np.zeros(self.in_channels)
                # grad update of V
                self.gw_v += np.dot(err.reshape((err.shape[0],1)), \
                        self.v_h[ib, ie].reshape((1,self.v_h.shape[2])))
                # First Hidden's error
                # grad update of W (another for loop)
                b = F.TanhD(np.dot(self.gw_v.T, err))
                erind = ie - 1
                while erind >= 0:
                    self.gw_w += np.dot(b.reshape((b.shape[0],1)), \
                            self.v_h[ib,erind].reshape((1,self.v_h.shape[2])))
                    b = F.TanhD(np.dot(self.gw_v.T, b))
                    erind -= 1
                # grad update of U (last one)
        
        self.w_u -= self.gw_u * self.lr / self.batchSize
        self.w_w -= self.gw_w * self.lr / self.batchSize
        self.w_v -= self.gw_v * self.lr / self.batchSize

        
        return self.g_h
    

    def Info(self):
        return "RNN (%d, [%d], %d)"%(self.in_channels, self.hidden_channels, self.out_channels)

#n = RNN(300, 256, 300)
#d = np.random.randn(32,39,300)
#f = np.random.randn(300)

#print(n.Forward(d))
#print(n.Backward(f))
