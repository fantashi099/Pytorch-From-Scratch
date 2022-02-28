import enum
import torch
import torch.nn as nn
from DenseNet import DenseNet

# Reduce (channels) model complexity
def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class Model(nn.Module):
    def __init__(self, dense_block = DenseNet, num_classes=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bnrom1 = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense_layer, self.in_channels = self.make_layer(dense_block, 64, 32)
        self.bnorm2 = nn.BatchNorm2d(self.in_channels)
        self.global_MaxPool = nn.AdaptiveMaxPool2d((1,1))
        self.fc1 = nn.Linear(self.in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def make_layer(self, dense_block, in_channels, out_channels):
        layers = []
        num_conv_in_dense_blocks = [2,2,2,2]
        # dense_block(num_conv, in, out)
        for index, idx_block in enumerate(num_conv_in_dense_blocks):
            layers.append(dense_block(idx_block, in_channels, out_channels))
            # Number of output channels after dense block (next in_channels)
            # Cuz they are concatenated (stack and stack and...)
            in_channels += out_channels * idx_block
            
            # Transition layer that halves the number of channels
            # after creating dense_block
            if index != len(num_conv_in_dense_blocks) - 1:
                layers.append(transition_block(in_channels, in_channels // 2))
                in_channels = in_channels // 2
        return nn.Sequential(*layers), in_channels
    
    def forward(self, x):
        x = self.maxPool(self.relu(self.bnrom1(self.conv1(x))))
        x = self.dense_layer(x)
        x = self.relu(self.bnorm2(x))
        # use adaptive_avg_pool2d 1.1k to achieve global average pooling, 
        # just set the output size to (1, 1)
        x = self.global_MaxPool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x