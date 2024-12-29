# Import packages
import torch.nn as nn


# Classes
class ConvBranch(nn.Module):

    def __init__(self, channels, kernel_size, p_drop=0, use_norm=False):
        super(ConvBranch, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        conv_list = []
        if use_norm:
            conv_list.append(nn.BatchNorm2d(self.channels[0]))
        for i in range(len(self.channels) - 1):
            conv_list.append(nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=self.kernel_size,
                                       padding="same"))
            conv_list.append(nn.ReLU())
        conv_list.append(nn.Dropout(p=p_drop))
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        return self.conv(x)
