import torch.nn as nn

def decode(bottleneck, channels = 3):
    return nn.Sequential(nn.Linear(bottleneck, 512),
                         nn.ReLU(),
                         nn.BatchNorm1d(512),
                         nn.Linear(512, 1024),
                         nn.ReLU(),
                         nn.BatchNorm1d(1024),
                         nn.Linear(1024, 32*32*channels),
                         nn.Tanh())