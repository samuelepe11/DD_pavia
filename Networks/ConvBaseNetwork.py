# Import packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from DataUtils.XrayDataset import XrayDataset
from Enumerators.ProjectionType import ProjectionType


# Classes
class PretrainedFeatureExtractor(nn.Module):

    def __init__(self, feature_extractor_model, freezable_layers):
        super(PretrainedFeatureExtractor, self).__init__()

        # Freeze network layers
        ConvBaseNetwork.freeze_layers(feature_extractor_model, freezable_layers)

        # Extract up to penultimate layer
        self.features = nn.Sequential(*list(feature_extractor_model.children())[:-2])
        self.output_channels = list(self.features.parameters())[-1].shape[0]

    def forward(self, x):
        return self.features(x)


class ConvBranch(nn.Module):

    def __init__(self, channels, kernel_size):
        super(ConvBranch, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        conv_list = []
        for i in range(len(self.channels) - 1):
            conv_list.append(nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=self.kernel_size, padding="same"))
            conv_list.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        return self.conv(x)


class ConvBaseNetwork(nn.Module):

    # Define attributes
    input_dim = None
    freezable_layers = None

    def __init__(self, feature_extractor_model, params, device="cpu"):
        super(ConvBaseNetwork, self).__init__()
        self.device = device
        self.params = params

        # Define attributes
        self.input_dim = 224
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.RandomOrder([
                    transforms.RandomAffine(degrees=10, translate=(0.01, 0.01), scale=(0.8, 1.2), fill=0),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.GaussianBlur(kernel_size=11, sigma=(1, 1.5)),
                ]),
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define pre-trained network
        self.feature_extractor = PretrainedFeatureExtractor(feature_extractor_model=feature_extractor_model,
                                                            freezable_layers=self.freezable_layers)

        # Define new convolutional branches
        self.conv_segment_sizes = ([self.feature_extractor.output_channels] + [params["n_conv_segment_neurons"]]
                                   * params["n_conv_segment_layers"])
        self.kernel_size = params["kernel_size"]
        self.segment_branches = nn.ModuleList([ConvBranch(channels=self.conv_segment_sizes,
                                                          kernel_size=self.kernel_size)
                                               for _ in XrayDataset.segment_dict.keys()])

        self.conv_view_sizes = ([params["n_conv_segment_neurons"]] + [params["n_conv_view_neurons"]]
                                * params["n_conv_view_layers"])
        self.view_branches = nn.ModuleList([ConvBranch(channels=self.conv_view_sizes, kernel_size=self.kernel_size)
                                            for _ in ProjectionType])

        # Define final fully connected layers
        self.fc_sizes = [self.conv_view_sizes[-1]] * params["n_fc_layers"]
        fc_list = []
        for i in range(len(self.fc_sizes) - 1):
            fc_list.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]))
            fc_list.append(nn.ReLU())
        fc_list.append(nn.Linear(self.fc_sizes[-1], 1))
        fc_list.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x, segment_ids, view_ids):
        batch_size, n_channels, height, width = x.shape

        # Process all channels through the pre-trained network (ResNet)
        x = x.view(batch_size * n_channels, 1, height, width)
        x = x.repeat(1, 3, 1, 1)
        if self.set_training:
            x = torch.stack([self.data_transforms(img) for img in x])
            x = x.to(self.device)
        features = self.feature_extractor(x)
        if not features.is_contiguous():
            features = features.contiguous()

        # Segment-specific processing
        segment_processed = torch.zeros(batch_size * n_channels, self.params["n_conv_segment_neurons"],
                                        features.shape[-2], features.shape[-1], device=x.device)
        for i in range(len(XrayDataset.segment_dict)):
            segment_type = list(XrayDataset.segment_dict.keys())[i]
            mask = torch.tensor([1 if segm_id == segment_type else 0 for segm_id in segment_ids])
            segment_processed[mask] = self.segment_branches[i](features[mask])
        if not segment_processed.is_contiguous():
            segment_processed = segment_processed.contiguous()

        # View-specific processing
        view_processed = torch.zeros(batch_size * n_channels, self.params["n_conv_view_neurons"], 
                                     segment_processed.shape[-2], segment_processed.shape[-1], device=x.device)
        for i in range(len(ProjectionType)):
            view_type = list(ProjectionType)[i]
            mask = torch.tensor(view_ids == view_type).view(batch_size * n_channels)
            view_processed[mask] = self.view_branches[i](segment_processed[mask])
        if not view_processed.is_contiguous():
            view_processed = view_processed.contiguous()

        # Reconstruct batch dimensions
        view_processed = view_processed.view(batch_size, n_channels, *view_processed.shape[1:])

        # Apply global average pooling for each channel
        gap_output = torch.mean(view_processed, dim=(-2, -1))

        # Average across all channels to aggregate information
        channel_avg = torch.mean(gap_output, dim=1)

        # Pass through final fully connected layer
        out = self.fc(channel_avg)
        return out

    def set_training(self, training=True):
        if training:
            self.train()
        else:
            self.eval()

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

    @staticmethod
    def freeze_layers(network, freezable_layers):
        for name, layer_obj in network.named_children():
            if name in freezable_layers:
                for param in layer_obj.parameters():
                    param.requires_grad = False
            else:
                for param in layer_obj.parameters():
                    param.requires_grad = True
