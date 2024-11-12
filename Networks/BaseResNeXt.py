# Import packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights

from Enumerators.ProjectionType import ProjectionType


# Class
class BaseResNeXt(nn.Module):
    # Define attributes
    input_dim = 224
    freezable_layers = ["conv1", "layer1", "layer2", "layer3"]
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
        super(BaseResNeXt, self).__init__()

        # Define pre-trained network
        self.res_next = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT, progress=False)
        self.freeze_layers()
        self.device = device

        # Define extra convolutional layers
        self.conv_sizes = [self.res_next.fc.in_features] + [params["n_conv_neurons"]] * params["n_conv_layers"]
        self.kernel_size = params["kernel_size"]
        for i in range(len(self.conv_sizes) - 1):
            setattr(self, f"ll_layer_{i}", nn.Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1],
                                                     kernel_size=self.kernel_size))
            setattr(self, f"ll_relu_{i}", nn.ReLU())
            setattr(self, f"ap_layer_{i}", nn.Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1],
                                                     kernel_size=self.kernel_size))
            setattr(self, f"ap_relu_{i}", nn.ReLU())

        # Antero-posterior views are a less used in clinical practice
        self.ap_weight = nn.Parameter(torch.randn(1))

        # Define extra fully connected layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_sizes = [self.conv_sizes[-1]] * params["n_fc_layers"]
        for i in range(len(self.fc_sizes)):
            if i != len(self.fc_sizes) - 1:
                setattr(self, f"fc_{i}", nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]))
                setattr(self, f"fc_relu_{i}", nn.ReLU())
            else:
                setattr(self, f"fc_{i}", nn.Linear(self.fc_sizes[i], 1))
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
                ap_output = getattr(self, "ap_layer_" + str(j))(ap_output)
                ap_output = getattr(self, "ap_relu_" + str(j))(ap_output)
                ll_output = getattr(self, "ll_layer_" + str(j))(ll_output)
                ll_output = getattr(self, "ll_relu_" + str(j))(ll_output)

            # Recreate the original batch-wise order
            output_ordered_list = []
            count_ap = 0
            count_ll = 0
            for j in range(len(ap_types_idx)):
                if ap_types_idx[j]:
                    output_ordered_list.append(ap_output[count_ap] * self.ap_weight)
                    count_ap += 1
                else:
                    output_ordered_list.append(ll_output[count_ll])
                    count_ll += 1
            output = torch.cat([ap_output, ll_output], dim=0)

            # GAP layer
            output = self.gap(output)
            output = output.view(output.size(0), -1)
            output_list.append(output)

        # Join latent vectors
        output = torch.mean(torch.stack(output_list, dim=0), dim=0)
        for i in range(len(self.fc_sizes)):
            output = getattr(self, "fc_" + str(i))(output)
            if i != len(self.fc_sizes) - 1:
                output = getattr(self, "fc_relu_" + str(i))(output)
        output = self.sigmoid(output)
        return output

    def set_training(self, training=True):
        if training:
            self.train()
        else:
            self.eval()

        # Set specific layers
        for layer_name, _ in self.named_children():
            getattr(self, layer_name).training = training

    def set_cuda(self, cuda=True):
        if cuda:
            self.cuda()
        else:
            self.cpu()

        # Set specific layers
        for layer_name, _ in self.named_children():
            if cuda:
                getattr(self, layer_name).cuda()
            else:
                getattr(self, layer_name).cpu()

    def freeze_layers(self):
        for layer in self.freezable_layers:
            for name, layer_obj in self.res_next.named_children():
                if name == layer:
                    for param in layer_obj.parameters():
                        param.requires_grad = False
