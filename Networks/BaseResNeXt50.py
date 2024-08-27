# Import packages
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from Enumerators.ProjectionType import ProjectionType


# Class
class BaseResNeXt50(nn.Module):
    # Define attributes
    input_dim = 224
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.RandomOrder([
                transforms.RandomAffine(degrees=15, translate=(0.3, 0.3), scale=(0.8, 1.2), fill=0),
                transforms.RandomResizedCrop(size=input_dim, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.GaussianBlur(kernel_size=53, sigma=(5, 10)),
            ]),
        ], p=0.5),
        transforms.ToTensor(),
    ])

    def __init__(self, params=None):
        super(BaseResNeXt50, self).__init__()

        # Define pre-trained network
        self.res_next = models.resnext50_32x4d()
        for param in self.res_next.layer4.parameters():
            param.requires_grad = True

        # Define extra convolutional layers
        self.conv_sizes = [self.res_next.fc.in_features] + [params["n_conv_neurons"]] * params["n_conv_layers"]
        self.kernel_size = params["kernel_size"]
        for i in range(len(self.conv_sizes) - 1):
            self.__dict__["ll_layer_" + str(i)] = nn.Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1],
                                                            kernel_size=self.kernel_size)
            self.__dict__["ll_relu_" + str(i)] = nn.ReLU()
            self.__dict__["ap_layer_" + str(i)] = nn.Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1],
                                                            kernel_size=self.kernel_size)
            self.__dict__["ap_relu_" + str(i)] = nn.ReLU()

        # Define extra fully connected layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_sizes = [self.conv_sizes[-1]] * params["n_fc_layers"]
        for i in range(len(self.fc_sizes)):
            if i != len(self.fc_sizes) - 1:
                self.__dict__["fc_" + str(i)] = nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1])
                self.__dict__["fc_relu_" + str(i)] = nn.ReLU()
            else:
                self.__dict__["fc_" + str(i)] = nn.Linear(self.fc_sizes[i], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, projections, projection_types):
        # Apply network
        output_list = []
        for projection_type, projection in zip(projection_types, projections):
            # Turn 2D image into 3D image
            output = torch.stack([projection] * 3, dim=-1)
            output = output.permute(0, 3, 1, 2)
            ##if self.res_next.training:
            ##    output = self.data_transforms(output)

            # Extract features
            output = self.res_next.conv1(output)
            output = self.res_next.layer1(output)
            output = self.res_next.layer2(output)
            output = self.res_next.layer3(output)
            output = self.res_next.layer4(output)

            if projection_type == ProjectionType.AP:
                layer_type = "ap"
            else:
                layer_type = "ll"
            for i in range(len(self.conv_sizes) - 1):
                output = self.__dict__[layer_type + "_layer_" + str(i)](output)
                output = self.__dict__[layer_type + "_relu_" + str(i)](output)

            # GAP layer
            output = self.gap(output)
            output_list.append(output)

        # Join latent vectors
        output = torch.mean(torch.cat(output_list, dim=0), dim=0, keepdim=True)
        output = output.view(output.size(0), -1)
        for i in range(len(self.fc_sizes)):
            output = self.__dict__["fc_" + str(i)](output)
            if i != len(self.fc_sizes) - 1:
                output = self.__dict__["fc_relu_" + str(i)](output)
        output = self.sigmoid(output)
        return output

    def set_training(self, training=True):
        if training:
            self.train()
            self.res_next.train()
        else:
            self.eval()
            self.res_next.eval()

        # Set specific layers
        for layer in self.__dict__.keys():
            if isinstance(self.__dict__[layer], nn.Module):
                self.__dict__[layer].training = training

    def set_cuda(self, cuda=True):
        if cuda:
            self.cuda()
            self.res_next.cuda()
        else:
            self.cpu()
            self.res_next.cpu()

        # Set specific layers
        for layer in self.__dict__.keys():
            if isinstance(self.__dict__[layer], nn.Module):
                if cuda:
                    self.__dict__[layer].cuda()
                else:
                    self.__dict__[layer].cpu()
