# Import packages
from enum import Enum


# Class
class NetType(Enum):
    BASE_RES_NEXT50 = "ResNeXt-50 model for preprocessing"
    BASE_RES_NEXT101 = "ResNeXt-101 model for preprocessing"
    BASE_VIT = "Visual Transformer model for preprocessing"
    ATTENTION_VIT = "Visual Transformer model for post-processing"
