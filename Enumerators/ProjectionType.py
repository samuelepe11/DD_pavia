# Import packages
from enum import Enum


# Class
class ProjectionType(Enum):
    AP = "antero-posterior"
    LAT = "latero-lateral"

    def translated_value(self):
        if self == ProjectionType.AP:
            return "antero-posteriore"
        else:
            return "latero-laterale"
