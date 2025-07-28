# Import packages
from enum import Enum


# Class
class ExtraDatasetType(Enum):
    BUU = "Burapha LSPINE dataset"
    AASCE = "Accurate Automated Spinal Curvature Estimation MICCAI 2019 dataset"
    DD = "Original cropped dataset from the Donald Duck project"

    def get_dataset_name(self):
        if self == ExtraDatasetType.BUU:
            return "BUU_LSPINE"
        elif self == ExtraDatasetType.AASCE:
            return "AASCE_MICCAI"
        elif self == ExtraDatasetType.DD:
            return "DD_bicocca"
        else:
            return None

    def get_ref_id_start(self):
        if self == ExtraDatasetType.BUU:
            return 10000
        elif self == ExtraDatasetType.AASCE:
            return 20000
        elif self == ExtraDatasetType.DD:
            return 30000
        else:
            return None
