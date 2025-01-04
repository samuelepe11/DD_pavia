# Import packages
import torch
import torch.nn as nn

from Networks.ConvBaseNetwork import ConvBaseNetwork
from Networks.ConvBranch import ConvBranch
from Networks.PretrainedAttentionModule import PretrainedAttentionModule
from DataUtils.XrayDataset import XrayDataset


# Class
class ConvAttentionNetwork(ConvBaseNetwork):

    # Define attributes
    freezable_layers = []
    n_removable_layers = None

    def __init__(self, feature_extractor_model, params=None, device="cpu"):
        super(ConvAttentionNetwork, self).__init__(feature_extractor_model=feature_extractor_model, params=params,
                                                   device=device)

    def initialize_modules(self, feature_extractor_model, params):
        # Define pre-trained network
        projection_module, encoder_module = feature_extractor_model
        self.attention_module = PretrainedAttentionModule(projection_module=projection_module,
                                                          encoder_module=encoder_module, device=self.device)

        # Define new convolutional branches
        self.conv_segment_sizes = ([3] + [params["n_conv_segment_neurons"]] * params["n_conv_segment_layers"])
        self.segment_branches = nn.ModuleList([ConvBranch(channels=self.conv_segment_sizes,
                                                          kernel_size=self.kernel_size, p_drop=params["p_dropout"],
                                                          use_norm=params["use_batch_norm"])
                                               for _ in XrayDataset.segment_dict.keys()])

        self.conv_view_sizes = ([params["n_conv_segment_neurons"]] + [params["n_conv_view_neurons"]]
                                * params["n_conv_view_layers"] + [3])
        self.view_branches = nn.ModuleList([ConvBranch(channels=self.conv_view_sizes, kernel_size=self.kernel_size,
                                                       p_drop=params["p_dropout"], use_norm=params["use_batch_norm"])
                                            for _ in XrayDataset.segment_dict.keys()])

        # Define final fully connected layers
        self.fc_sizes = [self.attention_module.output_channels] * params["n_fc_layers"]
        self.initialize_fc_module()

    def forward(self, x, segment_ids, view_ids):
        batch_size, n_channels, height, width = x.shape

        # Segment- and view-specific processing
        x = x.view(batch_size * n_channels, 1, height, width)
        x = x.repeat(1, 3, 1, 1)
        if self.training:
            x = [self.training_data_transforms(img) for img in x]
        x = torch.stack([self.inference_data_transforms(img) for img in x])
        x = x.to(self.device)
        x_processed = self.apply_convolutional_branches(x, batch_size, n_channels, segment_ids, view_ids)

        # Post-process all the features with the pre-trained attention-based network
        x_processed = self.attention_module(x_processed)
        if not x_processed.is_contiguous():
            x_processed = x_processed.contiguous()

        # Reconstruct batch dimensions
        x_processed = x_processed.view(batch_size, n_channels, *x_processed.shape[1:])

        # Average across all channels to aggregate information
        channel_avg = torch.mean(x_processed, dim=1)

        # Pass through final fully connected layer
        out = self.fc(channel_avg)
        return out
