# Import packages
from torchvision import models
from torchvision.models import ResNeXt101_64X4D_Weights

from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class BaseResNeXt101(ConvBaseNetwork):

    # Define attributes
    freezable_layers = ["bn1", "conv1", "layer1", "layer2", "layer3"]
    feature_extractor_model = models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT, progress=False)

    def __init__(self, params=None, device="cpu", weight_loss=False):
        super(BaseResNeXt101, self).__init__(feature_extractor_model=self.feature_extractor_model, params=params,
                                             device=device, weight_loss=weight_loss)
