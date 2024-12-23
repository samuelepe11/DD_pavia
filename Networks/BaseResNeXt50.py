# Import packages
from Networks.ConvBaseNetwork import ConvBaseNetwork
from Enumerators.NetType import NetType


# Class
class BaseResNeXt50(ConvBaseNetwork):

    # Define attributes
    input_dim = 224
    freezable_layers = ["bn1", "conv1", "layer1", "layer2", "layer3"]

    def __init__(self, params=None, device="cpu"):
        super(BaseResNeXt50, self).__init__(net_type=NetType.RES_NEXT50, params=params, device=device)
