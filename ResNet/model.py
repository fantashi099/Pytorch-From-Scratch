from turtle import forward
import torch
import torch.nn as nn

from ResNet import ResNet

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bnrom = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block1a = ResNet(64, 64, use_1x1conv=True)
        self.res_block1b = ResNet(64, 64)

        self.res_block2a = ResNet(64, 128, use_1x1conv=True)
        self.res_block2b = ResNet(128, 128)

        self.res_block3a = ResNet(128, 256, use_1x1conv=True)
        self.res_block3b = ResNet(256, 256)

        self.res_block4a = ResNet(256, 512, use_1x1conv=True)
        self.res_block4b = ResNet(512, 512)

        self.global_AvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.maxPool(self.bnrom(self.conv1(x)))
        x = self.res_block1b(self.res_block1a(x))
        x = self.res_block2b(self.res_block2a(x))
        x = self.res_block3b(self.res_block3a(x))
        x = self.res_block4b(self.res_block4a(x))

        # use adaptive_avg_pool2d 1.1k to achieve global average pooling, 
        # just set the output size to (1, 1)
        x = self.global_AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.softmax(x)