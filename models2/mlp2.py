import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, channels = 3, bottleneck = 256):
        super(SimpleMLP, self).__init__()
        self.channels = channels
        self.bottleneck = bottleneck
        self.bigpass = nn.Sequential(nn.Linear(32*32 * self.channels, 1024),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(1024),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(512),
                                     nn.Linear(512, self.bottleneck),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(self.bottleneck),
                                     nn.Linear(self.bottleneck, 10))
    def forward(self, x):
        x = x.view(-1, 32**2 * self.channels)
        return self.bigpass(x)

class SimpleMLPAutoencoder(nn.Module):
    def __init__(self, channels = 3, bottleneck = 256):
        super(SimpleMLPAutoencoder, self).__init__()
        self.channels = channels
        self.bottleneck = bottleneck
        self.encoder = nn.Sequential(nn.Linear(32*32*self.channels, 1024),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(1024),
                                      nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      nn.Linear(512, self.bottleneck),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(self.bottleneck))

        self.decoder = nn.Sequential(nn.Linear(self.bottleneck, 512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    nn.Linear(512, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 32*32*self.channels),
                                    nn.Tanh())

    def forward(self, x):
        out = x.view(-1, 32*32*self.channels)
        out = self.encoder(out)
        out = self.decoder(out)
        return out.view(-1, self.channels, 32, 32)
        

class HydraMLPAutoencoder(nn.Module):
    def __init__(self, channels = 3, bottleneck = 256):
        super(HydraMLPAutoencoder, self).__init__()
        self.channels = channels
        self.bottleneck = bottleneck
        self.encoder = nn.Sequential(nn.Linear(32*32*self.channels, 1024),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(1024),
                                      nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      nn.Linear(512, self.bottleneck),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(self.bottleneck))

        self.decoder = nn.Sequential(nn.Linear(self.bottleneck, 512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    nn.Linear(512, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 32*32*self.channels),
                                    nn.Tanh())

        self.prediction = nn.Sequential(nn.Linear(self.bottleneck, 10))

    def forward(self, x):
        out = x.view(-1, 32*32*self.channels)
        encoded = self.encoder(out)
        predicted = self.prediction(encoded)
        decoded = self.decoder(encoded)
        return decoded.view(-1, self.channels, 32, 32), predicted