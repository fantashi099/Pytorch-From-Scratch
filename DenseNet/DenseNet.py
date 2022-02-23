import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )

class DenseNet(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseNet, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                out_channels * i + in_channels, out_channels
            ))
        self.net = nn.Sequential(*layer)
    
    def forward(self, x):
        for block in self.net:
            out = block(x)
            # Concatenate the input and output of each block
            # on the channel dimension
            x = torch.cat((x, out), dim=1)
        return x