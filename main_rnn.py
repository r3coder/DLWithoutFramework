
from log import logger
import args

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

#RNNNet : RNN
class RNNNet:
    def __init__(self, output_size, lr=0.01):
        self.cf = []
        self.cf.append(Flatten())
        self.cf.append(Linear(784,1000,lr=lr))
        self.cf.append(ReLU())
        self.cf.append(Linear(1000,1000,lr=lr))
        self.cf.append(ReLU())
        self.cf.append(Linear(1000,500,lr=lr))
        self.cf.append(ReLU())
        self.cf.append(Linear(500,256,lr=lr))
        self.cf.append(ReLU())
        self.cf.append(Linear(256,10,lr=lr))
        self.cf.append(Softmax())

    def Forward(self, input_image):
        for i in range(len(self.cf)):
            input_image = self.cf[i].Forward(input_image)
        return input_image
    
    def Backward(self, error):
        for i in range(len(self.cf)-1, -1, -1):
            error = self.cf[i].Backward(error)
    
    def StrModelName(self):
        return "RNNNet"

    def StrModelStructure(self):
        a = ""
        for i in range(len(self.cf)):
            a += "%s"%(self.cf[i].Info()) + "\n"
        return a

import gzip
import os
import sys
from urllib import request

if __name__ == "__main__":
    logger.PrintDebug("Simple RNN Network Training",col='b')
    # Parse args
    args = args.parse()
    
    # Numpy options
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    # np.set_printoptions(threshold=sys.maxsize)
    
    # Download data if not exists
    locData = "./data_mnist/"
    url = "http://yann.lecun.com/exdb/mnist/"
    locTrainLabel = "train-labels-idx1-ubyte.gz"
    # if not os.path.exists(locData):
        # os.mkdir(locData)
    # if not os.path.exists(locData + locTrainLabel):
        # request.urlretrieve(url + locTrainLabel, locData + locTrainLabel )
    logger.PrintDebug("Raw Data file checked", col='g')

    # Load test/train data/labels
    trainData = LoadData(locData + "train_images", locData + locTrainData)
    testData = LoadData(locData + "test_images", locData + locTestData)
    trainLabel = LoadData(locData + "train_labels", locData + locTrainLabel)
    testLabel = LoadData(locData + "test_labels", locData + locTestLabel)
    logger.PrintDebug("Data Load Complete", col='g')
    
    
    batchSize = args.batch_size
    learningRate = args.learning_rate
    numEpochs = args.epoch
    
    numClasses = 10
    trainDataCount = trainData.shape[0]
    testDataCount = testData.shape[0]
    batchCount = trainDataCount // batchSize
    
    # Configurate model
    if args.model == "" or args.model == "LeeNetv2":
        m = LeeNetv2(numClasses, lr=learningRate, dropout=args.dropout, dropout_rate=args.dropout_rate)
    elif args.model == "LeeNetL":
        m = LeeNetL(numClasses, lr=learningRate, dropout=args.dropout, dropout_rate=args.dropout_rate)
    elif args.model == "PANet":
        m = PANet(numClasses, lr=learningRate, dropout=args.dropout, dropout_rate=args.dropout_rate)
    else:
        logger.PrintDebug("ERROR: Unknown Model specification",col='r')
        sys.exit(0)

    # Forward Once for deciding model structure
    m.Forward(trainData[0:1])
    # Print Basic Information
    logger.PrintDebug("   Model Specification   ",col="k", bg='y')
    print(m.StrModelStructure())
    logger.PrintDebug("   Training Specification   ",col="k", bg='y')
    print("Batch Size : %d"%batchSize)
    print("Learning Rate : %f"%learningRate)
    print("numEpochs : %d"%args.epoch)
    print("Training Data : MNIST")
    print()

    # Log file
    if args.log:
        modelName = m.StrModelName()
        locLogFile = "./logs/log_" + modelName + ".txt"
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        logFile = open(locLogFile,"w")
        logger.PrintDebug("Recording log file at " + locLogFile)
    
        # Write basic informations
        logFile.write(m.StrModelName() + "\n")
        logFile.write(m.StrModelStructure() + "\n")
        logFile.write("Start time : " + (logger.GetCurrentTime()) + "\n\n")
    
    # Training starting!
    logger.PrintDebug("   Training Start!   ", bg='r') 
    trainIdx = np.arange(trainData.shape[0])
    batchData = np.zeros((batchSize, trainData.shape[1], trainData.shape[2], trainData.shape[3]))
    batchLabel = np.zeros(batchSize, dtype=int)
    
    for epoch in range(numEpochs):
        logger.PrintDebug("   Epoch " + str(epoch+1) + "   ", col='k',bg='w') 
        # Shuffle image index
        np.random.shuffle(trainIdx)

        logger.PrintDebug("   Training   ", col='k',bg='b') 
        # Batch Training for SGD
        for batchIdx in range(batchCount):
            for i in range(batchSize):
                batchData[i] = trainData[trainIdx[i+batchIdx*batchSize]]
                batchLabel[i] = trainLabel[trainIdx[i+batchIdx*batchSize]]
            
            result = m.Forward(batchData)
            loss, _ = F.CrossEntropyLossBatch(result, batchLabel)
            
            answer = F.OneHotVectorBatch(batchLabel) # Get One-hot vector
            m.Backward(result - answer) # Backward propagation of the error
            
            # sys.exit(0)
            # Print stuff
            if args.log:
                logFile.write("%.4f\t"%loss)
            logger.PrintDebug(str("(E%2d/%2d|B%4d/%4d)(%5.1f|%5.1f) Loss : %.3f" \
                    %(epoch+1,numEpochs,batchIdx+1,batchCount, \
                    float(100*(epoch/numEpochs+(batchIdx+1)/batchCount/numEpochs)), \
                    float(100*(batchIdx+1)/batchCount), loss)),end = '\r') 
            # print()    
        print()
        logger.PrintDebug("   Evaluating   ", col='k',bg='b') 
        # Evaluate data
        result = m.Forward(testData)
        confusionMatrix = np.zeros((numClasses, numClasses))
        for ind, res in enumerate(result):
            indx = np.where(res == res.max())
            confusionMatrix[testLabel[ind], indx[0]] += 1
        logger.PrintDebug("Confusion Matrix")
        print(confusionMatrix)
        correct = np.sum(np.diagonal(confusionMatrix))
        logger.PrintDebug("Test Accuracy %d/%d"%(correct,testLabel.shape[0]))
    if args.log:
        logFile.write("\nConfusion Matrix\n"+str(confusionMatrix) +"\n\n")
        logFile.write("Test Accuracy %d/%d\n"%(correct,testLabel.shape[0]))


        logger.PrintDebug("   Epoch " + str(epoch+1) + " Finish   ", col='k',bg='g') 
    logger.PrintDebug("   Training Finish!   ", bg='r')
    if args.log:
        logFile.write("Elapsed Time : " + logger.GetElapsedTime())
        logFile.close()
        logger.PrintDebug("Log file recorded to "+locLogFile) 
    
