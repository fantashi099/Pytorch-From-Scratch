import torch
import torch.nn as nn

class InceptionA(nn.Module):
    def __init__(self, in_channels=1):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.branch_pool_2 = nn.Conv2d(in_channels, 24, kernel_size=1)


    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        outs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.concat(outs, 1) # dim = 1