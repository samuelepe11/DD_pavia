# Import packages
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights

from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class BaseResNeXt50(ConvBaseNetwork):

    # Define attributes
    freezable_layers = ["bn1", "conv1", "layer1", "layer2", "layer3"]
    feature_extractor_model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT, progress=False)

    def __init__(self, params=None, device="cpu"):
        super(BaseResNeXt50, self).__init__(feature_extractor_model=self.feature_extractor_model, params=params,
                                            device=device)
