# Import packages
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights

from Networks.BaseResNeXt50 import BaseResNeXt50


# Class
class BaseResNeXt50AllFreeze(BaseResNeXt50):
    freezable_layers = ["bn1", "conv1", "layer1", "layer2", "layer3", "layer4"]

    def __init__(self, params=None, device="cpu", weight_loss=False, transpose=False):
        super(BaseResNeXt50AllFreeze, self).__init__(params=params, device=device, weight_loss=weight_loss, transpose=transpose)
