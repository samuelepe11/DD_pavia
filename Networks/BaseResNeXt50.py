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
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
                transforms.RandomResizedCrop(size=input_dim, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.GaussianBlur(kernel_size=5, sigma=(1, 1.5)),
            ]),
        ], p=0.3),
        transforms.ToTensor(),
    ])

    def __init__(self, params=None, device="cpu"):
        super(BaseResNeXt50, self).__init__()

        # Define pre-trained network
        self.res_next = models.resnext50_32x4d()
        for param in self.res_next.layer4.parameters():
            param.requires_grad = True
        self.device = device

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
        for i in range(projections.shape[1]):
            # Turn 1-channel image into 3-channel image
            output = torch.stack([projections[:, i, :, :]] * 3, dim=1)
            if self.res_next.training:
                output = torch.stack([self.data_transforms(output[i]) for i in range(output.shape[0])], dim=0)
            output = output.to(self.device)

            # Extract features
            output = self.res_next.conv1(output)
            output = self.res_next.layer1(output)
            output = self.res_next.layer2(output)
            output = self.res_next.layer3(output)
            output = self.res_next.layer4(output)

            ap_types_idx = [x == ProjectionType.AP for x in projection_types[:, i]]
            ap_output = output[ap_types_idx]
            ll_types_idx = [not x for x in ap_types_idx]
            ll_output = output[ll_types_idx]
            for j in range(len(self.conv_sizes) - 1):
                ap_output = self.__dict__["ap_layer_" + str(j)](ap_output)
                ap_output = self.__dict__["ap_relu_" + str(j)](ap_output)
                ll_output = self.__dict__["ll_layer_" + str(j)](ll_output)
                ll_output = self.__dict__["ll_relu_" + str(j)](ll_output)
            output = torch.cat([ap_output, ll_output], dim=0)

            # GAP layer
            output = self.gap(output)
            output = output.view(output.size(0), -1)
            output_list.append(output)

        # Join latent vectors
        output = torch.mean(torch.stack(output_list, dim=0), dim=0)
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
