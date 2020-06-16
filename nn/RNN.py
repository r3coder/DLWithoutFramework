import numpy as np
import cupy as cp

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        mean = 0
        std = 0.01
        
        #lstm cell weights
        self.w_forget = np.random.normal(mean,std,(input_size+hidden_size,hidden_size))
        self.w_input = np.random.normal(mean,std,(input_size+hidden_size,hidden_size))
        self.w_output = np.random.normal(mean,std,(input_size+hidden_size,hidden_size))
        self.w_gate = np.random.normal(mean,std,(input_size+hidden_size,hidden_size))

        self.w_hidden_output = np.random.normal(mean,std,(hidden_size, output_size))

        self.v = np.zeros(output_size)

    def Forward(self, x):
        return self.v
    
    def GradAdd(self, x):
        pass

    def Backward(self, x):
        pass
    

