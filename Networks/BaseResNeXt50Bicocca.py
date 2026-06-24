# Import packages
import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights

from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class BaseResNeXt50Bicocca(ConvBaseNetwork):

    # Define attributes
    feature_extractor_model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT, progress=False)
    feature_extractor_model.fc = nn.Sequential(nn.Linear(feature_extractor_model.fc.in_features, 2))
    feature_extractor_model.load_state_dict(torch.load("bicocca_weights.pt"))

    inference_data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, params=None, device="cpu", weight_loss=False, transpose=False):
        super(BaseResNeXt50Bicocca, self).__init__(feature_extractor_model=self.feature_extractor_model, params=params,
                                                   device=device, weight_loss=weight_loss, transpose=transpose)
