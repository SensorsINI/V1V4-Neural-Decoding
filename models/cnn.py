import torch
import torch.nn as nn
from torch.nn import (AdaptiveAvgPool1d, BatchNorm1d, Conv1d, Dropout, Linear,
                      ReLU, Sequential)


class CNN(nn.Module):

    def __init__(self, num_classes, input_channels):
        super().__init__()
        self.num_classes = num_classes
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
        x = self.cnn_layers(x)
        x = torch.squeeze(x, 2)
        x = self.linear_layers(x)
        return x
