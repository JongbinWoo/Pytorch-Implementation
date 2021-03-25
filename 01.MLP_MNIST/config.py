import argparse
import torch

class AttributeDict(dict):
    def __init__(self):
        self.__dict__ = self

class ConfigTree:
    def __init__(self):
        self.DATASET = AttributeDict()
        self.SAVEDIR = AttributeDict()
        self.SYSTEM = AttributeDict()
        self.TRAIN = AttributeDict()
        self.MODEL = AttributeDict()
        self.KD = AttributeDict()

def get_config():
    # parser = argparse.ArgumentParser(description='MLP')
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--lr', default=0.001, type=float)

    # args = parser.parse_args()

    config = ConfigTree()
    config.SYSTEM.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.SAVEDIR = './data/outputs'
    config.DATASET.ROOT = './data'
    config.DATASET.NUM_CLASSES = 10
    config.DATASET.RATIO = 0.3

    config.TRAIN.AUGMENTATION = {'size': 224,
                                 'use_flip': True,
                                 'use_color_jitter': False,
                                 'use_normalize:': True}
    config.TRAIN.EPOCH = 10 #args.epochs
    config.TRAIN.BATCH_SIZE = 256 # args.batch_size
    config.TRAIN.BASE_LR = 0.001 #args.lr 
    config.TRAIN.PERIOD = 3

    config.MODEL.OPTIM = 'Adam'
    config.MODEL.HIDDEN = [32, 64, 128]

    return config



