
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
        self.cnn.append(Conv2d(1,4,(3,3)))      # 0
        self.cnn.append(ReLU())                 # 1
        self.cnn.append(MaxPool2d())            # 2
        self.cnn.append(Conv2d(4,8,(3,3)))      # 3
        self.cnn.append(ReLU())                 # 4
        self.cnn.append(MaxPool2d())            # 5
        self.cnn.append(Conv2d(8,16,(3,3)))     # 6
        self.cnn.append(ReLU())                 # 7
        self.cnn.append(Conv2d(16,16,(3,3)))    # 8
        self.cnn.append(ReLU())                 # 9
        self.cnn.append(Conv2d(16,24,(3,3)))    # 10
        self.cnn.append(ReLU())                 # 11
        

        self.cf = []
        self.cf.append(Concat())
        self.cf.append(Flatten())
        self.cf.append(Linear(1176,1000))
        self.cf.append(ReLU())
        self.cf.append(Linear(1000,500))
        self.cf.append(ReLU())
        self.cf.append(Linear(500,10))
        self.cf.append(Softmax())

    def Forward(self, input_image):
        for i in range(len(self.cnn)):
            input_image = self.cnn[i].Forward(input_image)
            # logger.PrintDebug("CNN " + str(i) + " " + str(input_image.shape))
        return input_image

import gzip
import os
from urllib import request
import numpy as np


def LoadLabel(loc):
    logger.PrintDebug("Loading label data from "+loc)
    with gzip.open(loc,'rb') as f:
        raw = f.read()
        if int.from_bytes(raw[:4], byteorder='big') != 2049: # Simple data verification
            return -1
        nImages = int.from_bytes(raw[4:8],byteorder='big')
        res = np.zeros((nImages),dtype=np.int32)
        for imageIdx in range(nImages):
            res[imageIdx] = int(raw[8+imageIdx])
    logger.PrintDebug("Total "+str(nImages)+" Labels loaded")
    return res
 

# Load image with simple normalization (0 to 1)
def LoadImage(loc):
    logger.PrintDebug("Loading image data from "+loc)
    with gzip.open(loc,'rb') as f:
        raw = f.read()
        if int.from_bytes(raw[:4], byteorder='big') != 2051: # Simple data verification
            return -1
        nImages = int.from_bytes(raw[4:8],byteorder='big')
        rows = int.from_bytes(raw[8:12],byteorder='big')
        cols = int.from_bytes(raw[12:16],byteorder='big')
        res = np.zeros((nImages,1 , rows, cols))
        
        for imageIdx in range(nImages):
            for rowIdx in range(rows):
                for colIdx in range(cols):
                    res[imageIdx,0,rowIdx,colIdx] = int(raw[16+imageIdx*rows*cols+rowIdx*cols+colIdx]) / 255.0
            if imageIdx % 10000 == 9999 and imageIdx + 1 != nImages:
                logger.PrintDebug("  "+str(imageIdx + 1)+" Images loaded")
    logger.PrintDebug("Total "+str(nImages)+" Images loaded")
    return res 


if __name__ == "__main__":
    logger.PrintDebug("Simple CNN Network Training",col='b')

    np.set_printoptions(precision=3)
    # Download data if not exists
    locData = "./data_mnist/"
    url = "http://yann.lecun.com/exdb/mnist/"
    locTrainLabel = "train-labels-idx1-ubyte.gz"
    locTrainData = "train-images-idx3-ubyte.gz"
    locTestLabel = "t10k-labels-idx1-ubyte.gz"
    locTestData = "t10k-images-idx3-ubyte.gz"
    if not os.path.exists(locData):
        os.mkdir(locData)
    if not os.path.exists(locData + locTrainLabel):
        request.urlretrieve(url + locTrainLabel, locData + locTrainLabel )
    if not os.path.exists(locData + locTrainData):
        request.urlretrieve(url + locTrainData, locData + locTrainData )
    if not os.path.exists(locData + locTestLabel):
        request.urlretrieve(url + locTestLabel, locData + locTestLabel )
    if not os.path.exists(locData + locTestData):
        request.urlretrieve(url + locTestData, locData + locTestData )
    logger.PrintDebug("Data Download / Verify Complete", col='g')

    # Load test/train data/labels
    trainData = LoadImage(locData + locTrainData)
    testData = LoadImage(locData + locTestData)
    trainLabel = LoadLabel(locData + locTrainLabel)
    testLabel = LoadLabel(locData + locTestLabel)
    if trainData.shape[0] != trainLabel.shape[0]:
        logger.PrintDebug("ERROR: Train data/label count mismatch",col='r')
        sys.exit(0)
    if testData.shape[0] != testLabel.shape[0]:
        logger.PrintDebug("ERROR: Test data/label count mismatch",col='r')
        sys.exit(0)
    logger.PrintDebug("Data Load Complete", col='g')
    
    # Configurate model
    m = Model(10)
    
    batchSize = 32
    learningRate = 0.001
    numEpochs = 10
    trainDataCount = trainLabel.shape[0]
    testDataCount = testLabel.shape[0]

    # Training starting!
    logger.PrintDebug("   Training Start!   ", bg='r') 
    
    trainIdx = np.arange(trainLabel.shape[0])
    
    for epoch in range(numEpochs):
        # Shuffle image index
        np.random.shuffle(trainIdx)

        for batchIdx in range(
        # Load 1 batch
        batchData = trainData[0:32,:,:]
        batchLabel
        # Forward the batch
        m.Forward(batch)

        # Get Error and add gradients

        # backward pass

