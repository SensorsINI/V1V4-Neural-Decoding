import torch
import torch.nn as nn
from torch.nn import (AdaptiveAvgPool1d, BatchNorm1d, Conv1d, Dropout,
                      Dropout2d, Linear, ReLU, Sequential)


class CNNSE(nn.Module):

    def __init__(self, num_classes, input_channels):
        super().__init__()
        self.num_classes = num_classes
        self.se = SELayer(channel=input_channels, reduction=16)
        self.cnn_layers = Sequential(
            Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding='same'),
            ReLU(inplace=True),
            BatchNorm1d(32),
            Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            ReLU(inplace=True),
            BatchNorm1d(32),
            Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            ReLU(inplace=True),
            BatchNorm1d(32),
            AdaptiveAvgPool1d(1),
            Dropout(0.5),
        )

        self.linear_layers = Sequential(Linear(in_features=32, out_features=self.num_classes))

    # Defining the forward pass
    def forward(self, x):
        x, _ = self.se(x)
        x = self.cnn_layers(x)
        x = torch.squeeze(x, 2)
        x = self.linear_layers(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.weight_reduc = None
        self.y = None
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), 
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        self.y = self.avg_pool(x).view(b, c)
        self.y = self.fc(self.y).view(b, c, 1)
        return x * self.y.expand_as(x), self.y.expand_as(x)
