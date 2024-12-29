# Import packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from Networks.PretrainedFeatureExtractor import PretrainedFeatureExtractor
from Networks.ConvBranch import ConvBranch
from DataUtils.XrayDataset import XrayDataset
from Enumerators.ProjectionType import ProjectionType


# Classes
class ConvBaseNetwork(nn.Module):
    # Define attributes
    input_dim = 224
    training_data_transforms = transforms.Compose([
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
    ])
    inference_data_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_removable_layers = 2
    freezable_layers = ["bn1", "conv1", "layer1", "layer2", "layer3"]

    def __init__(self, feature_extractor_model, params, device="cpu"):
        super(ConvBaseNetwork, self).__init__()
        self.device = device
        self.params = params

        # Define modules
        self.kernel_size = params["kernel_size"]
        self.feature_extractor = None
        self.attention_module = None
        self.conv_segment_sizes = None
        self.segment_branches = None
        self.conv_view_sizes = None
        self.view_branches = None
        self.fc_sizes = None
        self.fc = None
        self.initialize_modules(feature_extractor_model, params)

    def initialize_modules(self, feature_extractor_model, params):
        # Define pre-trained network
        self.feature_extractor = PretrainedFeatureExtractor(feature_extractor_model=feature_extractor_model,
                                                            freezable_layers=self.freezable_layers,
                                                            n_removable_layers=self.n_removable_layers)

        # Define new convolutional branches
        self.conv_segment_sizes = ([self.feature_extractor.output_channels] + [params["n_conv_segment_neurons"]]
                                   * params["n_conv_segment_layers"])
        self.segment_branches = nn.ModuleList([ConvBranch(channels=self.conv_segment_sizes,
                                                          kernel_size=self.kernel_size, p_drop=params["p_dropout"],
                                                          use_norm=params["use_batch_norm"])
                                               for _ in XrayDataset.segment_dict.keys()])

        self.conv_view_sizes = ([params["n_conv_segment_neurons"]] + [params["n_conv_view_neurons"]]
                                * params["n_conv_view_layers"])
        self.view_branches = nn.ModuleList([ConvBranch(channels=self.conv_view_sizes, kernel_size=self.kernel_size,
                                                       p_drop=params["p_dropout"], use_norm=params["use_batch_norm"])
                                            for _ in ProjectionType])

        # Define final fully connected layers
        self.fc_sizes = [self.conv_view_sizes[-1]] * params["n_fc_layers"]
        self.initialize_fc_module()

    def initialize_fc_module(self):
        fc_list = []
        for i in range(len(self.fc_sizes) - 1):
            fc_list.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]))
            fc_list.append(nn.ReLU())
        fc_list.append(nn.Linear(self.fc_sizes[-1], 1))
        fc_list.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x, segment_ids, view_ids):
        batch_size, n_channels, height, width = x.shape

        # Process all channels through the pre-trained convolutional network
        x = x.view(batch_size * n_channels, 1, height, width)
        x = x.repeat(1, 3, 1, 1)

        if self.training:
            x = [self.training_data_transforms(img) for img in x]
        x = torch.stack([self.inference_data_transforms(img) for img in x])
        x = x.to(self.device)
        x = self.feature_extractor(x)
        if not x.is_contiguous():
            x = x.contiguous()

        # Segment- and view-specific processing
        x_processed = self.apply_convolutional_branches(x, batch_size, n_channels, segment_ids, view_ids)

        # Reconstruct batch dimensions
        x_processed = x_processed.view(batch_size, n_channels, *x_processed.shape[1:])

        # Apply global average pooling for each channel
        gap_output = torch.mean(x_processed, dim=(-2, -1))

        # Average across all channels to aggregate information
        channel_avg = torch.mean(gap_output, dim=1)

        # Pass through final fully connected layer
        out = self.fc(channel_avg)
        return out

    def apply_convolutional_branches(self, x, batch_size, n_channels, segment_ids, view_ids):
        # Segment-specific processing
        segment_processed = torch.zeros(batch_size * n_channels, self.conv_segment_sizes[-1],
                                        x.shape[-2], x.shape[-1], device=x.device)
        for i in range(len(XrayDataset.segment_dict)):
            segment_type = list(XrayDataset.segment_dict.keys())[i]
            mask = torch.tensor([True if segm_id == segment_type else False for segm_id in segment_ids])
            mask = mask.unsqueeze(1).repeat(1, n_channels)
            mask = mask.view(batch_size * n_channels)
            segment_processed[mask] = self.segment_branches[i](x[mask])
        if not segment_processed.is_contiguous():
            segment_processed = segment_processed.contiguous()

        # View-specific processing
        view_processed = torch.zeros(batch_size * n_channels, self.conv_view_sizes[-1],
                                     segment_processed.shape[-2], segment_processed.shape[-1], device=x.device)
        for i in range(len(ProjectionType)):
            view_type = list(ProjectionType)[i]
            mask = torch.tensor(view_ids == view_type).view(batch_size * n_channels)
            view_processed[mask] = self.view_branches[i](segment_processed[mask])
        if not view_processed.is_contiguous():
            view_processed = view_processed.contiguous()

        return view_processed

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
