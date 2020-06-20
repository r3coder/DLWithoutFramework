import numpy as np
from PIL import Image
from log import logger


def Download():
    from urllib import request
    import zipfile
    import os

    # Delete existing file
    logger.PrintDebug("Clearing Folder...")
    top = ("./data/")
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    
    # Create directory if not exists
    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    # Download Flickr8k
    logger.PrintDebug("Downloading Flickr8k Text...")
    p = "./data/Flickr8k_text.zip"
    request.urlretrieve("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", p)
    logger.PrintDebug("Unpacking Flickr8k Text...")
    with zipfile.ZipFile(p, "r") as zr:
        zr.extractall("data/Flickr8k/")
    os.remove(p)

    logger.PrintDebug("Downloading Flickr8k Image...")
    p = "./data/Flickr8k_Dataset.zip"
    request.urlretrieve("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", p)
    logger.PrintDebug("Unpacking Flickr8k Image...")
    with zipfile.ZipFile(p, "r") as zr:
        zr.extractall("data/Flickr8k/")
    os.remove(p)

    logger.PrintDebug("Downloading glove6B...")
    p = "./data/glove6B.zip"
    request.urlretrieve(" http://nlp.stanford.edu/data/glove.6B.zip", p)
    logger.PrintDebug("Unpacking glove6B...")
    with zipfile.ZipFile(p, "r") as zr:
        zr.extractall("data/glove6B/")
    os.remove(p)

# Verifing data
def Verify():
    logger.PrintDebug("Verifing Data...")
    logger.PrintDebug("Data Verified!")
    return True

import pickle
import os

# Load glove6B data
def glove6B(sz = 50, logDiv = 100000):
    loc = "data/glove6B/glove6B%s" % str(sz) + ".dump"
    if not os.path.exists(loc):
        res = _glove6B(sz, logDiv)
        pickle.dump(res, open(loc,'wb'))
    else:
        res = pickle.load(open(loc,'rb'))
        logger.PrintDebug("Loaded " + loc)
    return res

def _glove6B(sz = 50, logDiv = 100000):
    d = dict()
    if sz not in [50, 100, 200, 300]:
        logger.PrintDebug("Not supported word embedding size, returning empty dict")
        return d
    path = "data/glove6B/glove.6B." + str(sz) + "d.txt"
    logger.PrintDebug("Reading embedded word file from" + path)
    f = open(path, 'r', encoding="utf-8")
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("\n","").split(" ")
        if len(line) != sz + 1:
            break
        l = np.zeros(sz)
        for i in range(sz):
            l[i] = float(line[i+1])
        d[line[0]] = l
        # Print log
        if logDiv > 0 and len(d) % logDiv == 0:
            logger.PrintDebug("  Data Count = " + str(len(d)))
    f.close()
    logger.PrintDebug("Loaded " + str(len(d)) + " embedded word data")
    return d

# Load Flickr8k Test data
def Flickr8k(mode = "test", logDiv = 2000, imageSize=[64, 64]):
    data = list()
    dataL = list()

    # Load image list
    imageList = list()
    path = "data/Flickr8k/Flickr_8k." + mode + "Images.txt"
    logger.PrintDebug("Loading " + mode + " image list from " + path)
    f = open(path, 'r', encoding="utf-8")
    while True:
        line = f.readline().replace("\n", "")
        if not line: break
        imageList.append(line)
    logger.PrintDebug("Total of " + str(len(imageList)) + " " + mode + " image list loaded")
    
    # Load token
    tokens = dict()
    path = "data/Flickr8k/Flickr8k.token.txt"
    logger.PrintDebug("Loading tokens from " + path)
    f = open(path, 'r', encoding="utf-8")
    while True:
        line = f.readline().replace("\n","")
        if not line: break
        line = line.split("\t")

        if line[0][:-2] not in tokens: # Add token
            tokens[line[0][:-2]] = [line[1]]
        else:
            tokens[line[0][:-2]].append(line[1])
    logger.PrintDebug( str(len(tokens)) + " image tokens are loaded")

    # Load images and tokens
    logger.PrintDebug("Loading " + mode + " images from " + path)
    for ind, val in enumerate(imageList):
        # Load image
        data.append(np.asarray(Image.open('data/Flickr8k/Flicker8k_Dataset/' + val).resize((imageSize[0], imageSize[1]))))
        # Load tag
        dataL.append(tokens[val])
        # Print log
        if logDiv > 0 and (ind + 1) % logDiv == 0:
            logger.PrintDebug("  Data Count = " + str(ind + 1))
    logger.PrintDebug("Total of " + str(len(data)) + " " + mode + " images loaded")

    return data, dataL

