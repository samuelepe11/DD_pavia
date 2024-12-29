# Import packages
import torch.nn as nn


# Classes
class PretrainedFeatureExtractor(nn.Module):

    def __init__(self, feature_extractor_model, freezable_layers, n_removable_layers):
        super(PretrainedFeatureExtractor, self).__init__()

        # Freeze network layers
        PretrainedFeatureExtractor.freeze_layers(feature_extractor_model, freezable_layers)

        # Extract up to penultimate layer
        self.features = nn.Sequential(*list(feature_extractor_model.children())[:-n_removable_layers])
        self.output_channels = list(self.features.parameters())[-1].shape[0]

    def forward(self, x):
        return self.features(x)

    @staticmethod
    def freeze_layers(network, freezable_layers):
        for name, layer_obj in network.named_children():
            if name in freezable_layers:
                for param in layer_obj.parameters():
                    param.requires_grad = False
            else:
                for param in layer_obj.parameters():
                    param.requires_grad = True
