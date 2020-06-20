
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


import numpy as np

# LeeNetR : RNN Net
class LeeNetR:
    def __init__(self, in_sz, hid_sz, out_sz, lr=0.01, dropout=True, dropout_rate=0.5):
        self.rnn = RNN(in_sz, hid_sz, out_sz)

    def Forward(self, input_stream):
        input_stream = self.rnn.Forward(input_stream)
        return input_stream
    
    def Backward(self, error):
        error = self.rnn.Backward(error)
    
    def StrModelName(self):
        return "LeeNetR"

    def StrModelStructure(self):
        a = ""
        a += "%s"%(self.rnn.Info()) + "\n"
        return a

import gzip
import os
import sys
from urllib import request
import load
import string

def Load(loc, wordEmb, emb_size = 50):
    res = []
    table = str.maketrans(dict.fromkeys(string.punctuation))
    f = open(loc, 'r')
    exceptions = 0
    while True:
        line = f.readline()
        if not line:
            break
        if len(line) < 1:
            continue
        # Remove Punctuations
        # Should I handle with 'd -> ed???
        line = line.translate(table).lower().rstrip()
        line = line.split(" ")
        for elem in line:
            if elem == "":
                continue
            try:
                res.append(wordEmb[elem])
            except:
                # logger.PrintDebug("Word not found! %s"%elem,col="r")
                res.append(np.zeros(emb_size))
                exceptions += 1
    f.close()
    r = np.array(res)
    logger.PrintDebug("Excepted word count : %d"%exceptions, col="r")
    logger.PrintDebug("Loaded word size : %s"%(str(r.shape)), col="g")
    return r

if __name__ == "__main__":
    logger.PrintDebug("Simple RNN Network Training",col='b')
    # Parse args
    args = args.parse()
    
    # Numpy options
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    # np.set_printoptions(threshold=sys.maxsize)
    
    # Download data if not exists
    locData = "./data_tinyshakespeare/"
    url = "http://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    locFile = "input.txt"
    if not os.path.exists(locData):
        os.mkdir(locData)
    if not os.path.exists(locData + locFile):
        request.urlretrieve(url, locData + locFile)
    logger.PrintDebug("Raw Data file checked", col='g')

    # Load and Process Data
    embSize = args.word_embedding_size
    streamSize = args.sequence_size
    wordEmb = load.glove6B(sz=embSize)
    data = Load(locData + locFile, wordEmb, emb_size = embSize)
    
    batchSize = args.batch_size
    learningRate = args.learning_rate
    numEpochs = args.epoch
    
    dataCount = data.shape[0]
    batchCount = dataCount // embSize
    
    # Configurate model
    m = LeeNetR(embSize, 50, embSize, lr=learningRate, dropout=args.dropout, dropout_rate=args.dropout_rate)

    # Forward Once for deciding model structure
    tmpData = np.zeros((1,streamSize,embSize))
    tmpData[0] = data[0:streamSize]
    m.Forward(tmpData)
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
    trainIdx = np.arange(batchCount) * batchSize
    batchData = np.zeros((batchSize, streamSize, embSize))
    
    for epoch in range(numEpochs):
        logger.PrintDebug("   Epoch " + str(epoch+1) + "   ", col='k',bg='w') 
        # Shuffle image index
        np.random.shuffle(trainIdx)

        logger.PrintDebug("   Training   ", col='k',bg='b') 
        # Batch Training for SGD
        for batchIdx in range(batchCount):
            for i in range(batchSize):
                batchData[i] = data[trainIdx[batchIdx]+i]
            
            result = m.Forward(batchData)
            
            m.Backward(result) # Backward propagation of the error
            loss = 99
            # sys.exit(0)
            # Print stuff
            # if args.log:
                # logFile.write("%.4f\t"%loss)
            logger.PrintDebug(str("(E%2d/%2d|B%4d/%4d)(%5.1f|%5.1f) Loss : %.3f" \
                    %(epoch+1,numEpochs,batchIdx+1,batchCount, \
                    float(100*(epoch/numEpochs+(batchIdx+1)/batchCount/numEpochs)), \
                    float(100*(batchIdx+1)/batchCount), loss)),end = '\r') 
            # print()    
        print()
        logger.PrintDebug("   Evaluating   ", col='k',bg='b') 
        # Evaluate data
        # result = m.Forward(testData)
    if args.log:
        pass

        logger.PrintDebug("   Epoch " + str(epoch+1) + " Finish   ", col='k',bg='g') 
    logger.PrintDebug("   Training Finish!   ", bg='r')
    if args.log:
        logFile.write("Elapsed Time : " + logger.GetElapsedTime())
        logFile.close()
        logger.PrintDebug("Log file recorded to "+locLogFile) 
    
