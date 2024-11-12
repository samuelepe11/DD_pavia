# Import packages
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt101_64X4D_Weights

from Networks.BaseResNeXt import BaseResNeXt
from Enumerators.ProjectionType import ProjectionType


# Class
class BaseResNeXt101(BaseResNeXt):

    def __init__(self, params=None, device="cpu"):
        super(BaseResNeXt101, self).__init__(params=params, device=device)

        # Define pre-trained network
        self.res_next = models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT, progress=False)
        self.freeze_layers()

        # Define extra convolutional layers
        self.conv_sizes[0] = self.res_next.fc.in_features
        for i in range(len(self.conv_sizes) - 1):
            self.__dict__["ll_layer_" + str(i)] = nn.Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1],
                                                            kernel_size=self.kernel_size)
            self.__dict__["ll_relu_" + str(i)] = nn.ReLU()
            self.__dict__["ap_layer_" + str(i)] = nn.Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1],
                                                            kernel_size=self.kernel_size)
            self.__dict__["ap_relu_" + str(i)] = nn.ReLU()
