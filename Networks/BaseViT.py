# Import packages
from torchvision import models
from torchvision.models import ViT_B_32_Weights

from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class BaseViT(ConvBaseNetwork):

    # Define attributes
    freezable_layers = []
    feature_extractor_model = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT, progress=False)

    def __init__(self, params=None, device="cpu"):
        super(BaseViT, self).__init__(feature_extractor_model=self.feature_extractor_model, params=params,
                                      device=device)
