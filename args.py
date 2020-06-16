import argparse
def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise argparse.ArgumentTypeError("Not Boolean value")


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--download", type=str2bool, default=False,
            help = "Download Data if true. Data will download if verifing fails")
    parser.add_argument("-e","--epoch", type=int, default=10,
            help = "Training epochs")
    parser.add_argument("-lr","--learning-rate", type=float, default=0.001,
            help = "Learning rate")
    parser.add_argument("-ws", "--word-embedding-size", type=int, choices=[50,100,200,300],default=300,
            help = "Size of the word embedding vector")
    """
    parser.add_argument("-i","--int", type=int, choices=[0,1,2], default=1,
            help = "Integer with choices")
    parser.add_argument("-sb","--str2bool", type=str2bool, default=False,
            help = "String to Bool")
    parser.add_argument("-s","--string", type=str, default = "",
            help = "String")
    """
    return parser.parse_args()