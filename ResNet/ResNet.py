import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_1x1conv=False):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv1x1 = None
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.bnorm2(self.conv2(x))
        if self.conv1x1:
            residual = self.conv1x1(residual)
        return self.relu(x + residual)
