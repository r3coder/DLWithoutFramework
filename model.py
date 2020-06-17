
from log import logger

from nn.Linear import Linear
from nn.Conv2d import Conv2d
from nn.RNN import RNN

from nn.ReLU import ReLU
from nn.Tanh import Tanh
from nn.Softmax import Softmax

from nn.Concat import Concat
from nn.Add import Add
from nn.MaxPool2d import MaxPool2d
from nn.Flatten import Flatten



class Model:

    def __init__(self, output_size):
        self.cnn = []
        self.cnn.append(Conv2d(3,16,(3,3)))     # 0
        self.cnn.append(ReLU())                 # 1
        self.cnn.append(Conv2d(16,16,(3,3)))    # 2
        self.cnn.append(ReLU())                 # 3
        self.cnn.append(MaxPool2d())            # 4
        self.cnn.append(Conv2d(16,32,(3,3)))    # 5
        self.cnn.append(ReLU())                 # 6
        self.cnn.append(MaxPool2d())            # 7
        self.cnn.append(Conv2d(32,32,(3,3)))    # 8
        self.cnn.append(ReLU())                 # 9
        self.cnn.append(MaxPool2d())            # 10
        self.cnn.append(Conv2d(32,64,(3,3)))    # 11
        self.cnn.append(ReLU())                 # 12

        self.concat = []
        self.concat.append(Concat())
        self.concat.append(MaxPool2d())
        self.concat.append(Flatten())
        self.concat.append(Linear(1536, 256))
        self.concat.append(Tanh())

        self.emb = []
        self.emb.append(RNN(output_size, 256, 256))

        self.linear = [] # concat and linear
        self.linear.append(Add())
        self.linear.append(Linear(256,256))
        self.linear.append(ReLU())
        self.linear.append(Linear(256,output_size))
        self.linear.append(Softmax())

    def Forward(self, input_image, input_word):
        for i in range(len(self.cnn)):
            input_image = self.cnn[i].Forward(input_image)
            if i == 10: 
                input_image_c4 = input_image
            logger.PrintDebug("CNN " + str(i) + " " + str(input_image.shape))
        input_image = (input_image_c4, input_image)
        for i in range(len(self.concat)):
            input_image = self.concat[i].Forward(input_image)
            logger.PrintDebug("Concat " + str(i) + " " + str(input_image.shape))
        for i in range(len(self.emb)):
            input_word = self.emb[i].Forward(input_word)
            logger.PrintDebug("Embedding " + str(i) + " " + str(input_word.shape))
        input_linear = (input_image, input_word)
        for i in range(len(self.linear)):
            input_linear = self.linear[i].Forward(input_linear)
            logger.PrintDebug("Linear " + str(i) + " " + str(input_linear.shape))
        return input_linear



    
import numpy as np
import cupy as cp
l1 = np.random.uniform(0, 1, (32, 3, 64, 64))
l2 = np.random.uniform(-1, 1, (32, 39, 300))
m = Model(300)
# logger.PrintDebug(str(m.Forward(l1, l2)))
m.Forward(l1, l2)

