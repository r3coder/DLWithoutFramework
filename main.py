
import args
import model
import load
import image
from log import logger
import numpy as np
from model import Model

if __name__ == "__main__":
    logger.PrintDebug("Initiated")
    # parse args
    args = args.parse()
    # Numpy set precision
    np.set_printoptions(precision=3)

    # Download Data if argument is checked or falied to verify
    if args.download or not load.Verify():
        load.Download()
    
    dataTrain = []; dataTrainLabel = []
    # load data to memory
    # dataTrain, dataTrainLabel = load.Flickr8k(mode = "train")
    dataTest, dataTestLabel = load.Flickr8k(mode = "test")

    # Normalize images
    for i in range(len(dataTest)):
        dataTest[i] = image.Normalize(dataTest[i])
    for i in range(len(dataTrain)):
        dataTrain[i] = image.Normalize(dataTrain[i])
    logger.PrintDebug("Image Normalized")

    # Add modified images


    
    logger.PrintDebug("Added Modified Images")
    # Load embedded word data
    # dataEmbWord = load.glove6B(args.word_embedding_size)

    # Prepare model
    m = Model(args.word_embedding_size)
    # Training
    
    for ep in range(args.epoch):
        pass

    logger.PrintDebug("===== COMPLETE! =====")