# Import packages
from torchvision import models

from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class BaseResNet18(ConvBaseNetwork):

    # Define attributes
    freezable_layers = ["bn1", "conv1", "layer1", "layer2", "layer3"]
    feature_extractor_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    def __init__(self, params=None, device="cpu", weight_loss=False):
        super(BaseResNet18, self).__init__(feature_extractor_model=self.feature_extractor_model, params=params,
                                           device=device, weight_loss=weight_loss)
