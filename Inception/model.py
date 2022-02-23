import torch
import torch.nn as nn

from Inception import InceptionA

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incept1 = InceptionA(10)
        self.incept2 = InceptionA(20)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = nn.Linear(2200,10)
    
    def forward(self, x):
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.incept1(x)
        x = self.relu(self.maxpool(self.conv2(x)))
        x = self.incept2(x)
        x = torch.flatten(x, 1)
        x = self.logsoftmax(self.fc(x))
        return x