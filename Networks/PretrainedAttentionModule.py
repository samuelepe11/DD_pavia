# Import packages
import torch
import torch.nn as nn


# Classes
class PretrainedAttentionModule(nn.Module):

    def __init__(self, projection_module, encoder_module, device="cpu"):
        super(PretrainedAttentionModule, self).__init__()
        self.device = device

        # Freeze all network layers
        for param in projection_module.parameters():
            param.requires_grad = False
        for param in encoder_module.parameters():
            param.requires_grad = False

        self.projection_module = projection_module
        self.encoder_module = encoder_module

        # Extract up to penultimate layer
        self.output_channels = list(self.encoder_module.parameters())[-1].shape[0]

    def forward(self, x):
        # Project patches
        batch_size = x.size(0)
            
        patches = self.projection_module(x)
        patches = patches.flatten(2).transpose(1, 2)

        # Introduce class token
        cls_token = nn.Parameter(torch.randn(1, 1, self.output_channels))
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens.to(self.device)
        patches_with_cls = torch.cat((cls_tokens, patches), dim=1)

        # Encode
        encoding = self.encoder_module(patches_with_cls)
        cls_encoding = encoding[:, 0, :]
        return cls_encoding
