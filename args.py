import argparse
def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise argparse.ArgumentTypeError("Not Boolean value")


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dl","--download", type=str2bool, default=False,
            help = "Download Data if true. Data will download if verifing fails")
    parser.add_argument("-e","--epoch", type=int, default=10,
            help = "Training epochs")
    parser.add_argument("-bs","--batch-size", type=int, default=32,
            help = "Batch Size")
    parser.add_argument("-lr","--learning-rate", type=float, default=0.001,
            help = "Learning rate")
    parser.add_argument("-ws", "--word-embedding-size", type=int, choices=[50,100,200,300],default=300,
            help = "Size of the word embedding vector")
    parser.add_argument("-ss", "--sequence-size", type=int, default=50,
            help = "Size of the sequence when training tinyshakespeare")
    parser.add_argument("-d","--dropout", type=str2bool, default=True,
            help = "True if dropout is used at training")
    parser.add_argument("-dr","--dropout-rate", type=float, default=0.5,
            help = "Dropout rate")
    parser.add_argument("-l","--log", type=str2bool, default=False,
            help = "Left log file if true. It is saved on ./logs/") 
    parser.add_argument("-m","--model", type=str, default = "",
            help = "Model to train, leave blank for Basic")
    """
    parser.add_argument("-i","--int", type=int, choices=[0,1,2], default=1,
            help = "Integer with choices")
    parser.add_argument("-sb","--str2bool", type=str2bool, default=False,
            help = "String to Bool")
    parser.add_argument("-s","--string", type=str, default = "",
            help = "String")
    """
    return parser.parse_args()
