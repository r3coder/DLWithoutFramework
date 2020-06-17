
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

import nn.Functions as F

import numpy as np

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
        self.cf.append(Linear(1960,1000))
        self.cf.append(ReLU())
        self.cf.append(Linear(1000,500))
        self.cf.append(ReLU())
        self.cf.append(Linear(500,10))
        self.cf.append(Softmax())

    def Forward(self, input_image):
        for i in range(len(self.cnn)):
            input_image = self.cnn[i].Forward(input_image)
            if i == 8:
                input_2 = input_image
            # logger.PrintDebug("CNN " + str(i) + " " + str(input_image.shape))
        input_image = (input_image, input_2)
        for i in range(len(self.cf)):
            input_image = self.cf[i].Forward(input_image)
            # logger.PrintDebug("CF " + str(i) + " " + str(input_image.shape))
        return input_image
    
    def Backward(self, error):
        for i in range(len(self.cf)-1, 0, -1):
            error = self.cf[i].Backward(error)
            logger.PrintDebug("CF " + str(i) + " " + str(error.shape))
        error, error2 = self.cf[0].Backward(error)
        logger.PrintDebug("CF 0 " + str(error.shape) + str(error2.shape))
        for i in range(len(self.cnn)-1, -1, -1):
            error = self.cnn[i].Backward(error)
            logger.PrintDebug("CNN " + str(i) + " " + str(error.shape))
import gzip
import os
import sys
from urllib import request


# Load data
def LoadRawData(loc):
    logger.PrintDebug("Loading data from "+loc)
    with gzip.open(loc,'rb') as f:
        raw = f.read()
        if int.from_bytes(raw[:4], byteorder='big') == 2051:
            # This is image file
            nImages = int.from_bytes(raw[4:8],byteorder='big')
            rows = int.from_bytes(raw[8:12],byteorder='big')
            cols = int.from_bytes(raw[12:16],byteorder='big')
            res = np.zeros((nImages,1 , rows, cols))
        
            for imageIdx in range(nImages):
                for rowIdx in range(rows):
                    for colIdx in range(cols):
                        res[imageIdx,0,rowIdx,colIdx] = int(raw[16+imageIdx*rows*cols+rowIdx*cols+colIdx]) / 255.0
                if imageIdx % 10000 == 9999 and imageIdx + 1 != nImages:
                    logger.PrintDebug("  "+str(imageIdx + 1)+" Images loaded",end='\r')
            logger.PrintDebug("Total "+str(nImages)+" Images loaded")
        elif int.from_bytes(raw[:4], byteorder='big') == 2049:
            # This is label file
            nImages = int.from_bytes(raw[4:8],byteorder='big')
            res = np.zeros((nImages),dtype=np.int32)
            for imageIdx in range(nImages):
                res[imageIdx] = int(raw[8+imageIdx])
            logger.PrintDebug("Total "+str(nImages)+" Labels loaded")
        else:
            logger.PrintDebug("File is not a MNIST image or label file.",col="r")
            return -1
    return res 

def LoadData(loc, rawloc):
    if not os.path.exists(loc + ".npy"):
        res = LoadRawData(rawloc)
        np.save(loc, res)
    else:
        res = np.load(loc + ".npy")
        logger.PrintDebug("Loaded " + loc + ".npy")
    return res

if __name__ == "__main__":
    logger.PrintDebug("Simple CNN Network Training",col='b')

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    # np.set_printoptions(threshold=sys.maxsize)
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
    logger.PrintDebug("Raw Data file checked", col='g')

    # Load test/train data/labels
    trainData = LoadData(locData + "train_images", locData + locTrainData)
    testData = LoadData(locData + "test_images", locData + locTestData)
    trainLabel = LoadData(locData + "train_labels", locData + locTrainLabel)
    testLabel = LoadData(locData + "test_labels", locData + locTestLabel)
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
    numEpochs = 5
    trainDataCount = trainData.shape[0]
    testDataCount = testData.shape[0]
    batchCount = trainDataCount // batchSize
    
    # Training starting!
    logger.PrintDebug("   Training Start!   ", bg='r') 
    trainIdx = np.arange(trainData.shape[0])
    batchData = np.zeros((batchSize, trainData.shape[1], trainData.shape[2], trainData.shape[3]))
    batchLabel = np.zeros(batchSize, dtype=int)
    
    for epoch in range(numEpochs):
        logger.PrintDebug("   Epoch " + str(epoch+1) + "   ", col='k',bg='w') 
        # Shuffle image index
        np.random.shuffle(trainIdx)

        for batchIdx in range(batchCount):
            for i in range(batchSize):
                batchData[i] = trainData[trainIdx[i+batchIdx*batchSize]]
                batchLabel[i] = trainLabel[trainIdx[i+batchIdx*batchSize]]
            
            result = m.Forward(batchData)
            loss, _ = F.CrossEntropyLossBatch(result, batchLabel)
            
            answer = F.OneHotVectorBatch(batchLabel) # Get One-hot vector
            m.Backward(result - answer) # Backward propagation of the error
            

            # Print stuff
            logger.PrintDebug(str("(E%2d/%2d|B%4d/%4d)(%5.1f|%5.1f) Loss : %.3f" \
                    %(epoch+1,numEpochs,batchIdx+1,batchCount, \
                    float(100*(epoch/numEpochs+(batchIdx+1)/batchCount/numEpochs)), \
                    float(100*(batchIdx+1)/batchCount), loss)),end = '\r') 
        # Evaluate data
        result = m.Forward(testData)
        
        logger.PrintDebug("   Epoch " + str(epoch+1) + " Finish   ", col='k',bg='g') 
    logger.PrintDebug("   Training Finish!   ", bg='r') 
