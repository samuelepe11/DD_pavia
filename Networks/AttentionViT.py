# Import packages
from torchvision import models
from torchvision.models import ViT_B_32_Weights

from Networks.ConvAttentionNetwork import ConvAttentionNetwork


# Class
class AttentionViT(ConvAttentionNetwork):

    # Define attributes
    conv_proj_ind = 0
    encoder_ind = 1
    feature_extractor_model = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT, progress=False)
    conv_proj = feature_extractor_model.conv_proj
    encoder = feature_extractor_model.encoder

    def __init__(self, params=None, device="cpu", weight_loss=False):
        super(AttentionViT, self).__init__(feature_extractor_model=(self.conv_proj, self.encoder), params=params,
                                           device=device, weight_loss=weight_loss)
