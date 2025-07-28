# Import packages
import cv2

from DataUtils.PatientInstance import PatientInstance
from Enumerators.ProjectionType import ProjectionType


# Class
class ExtraPatientInstance(PatientInstance):
    def __init__(self, pt_info, pt_instances, img_folder_path, proj_ids, segment_ids):
        super().__init__(pt_info, pt_instances, img_folder_path, proj_ids, segment_ids)

    def store_patient_generalities(self, pt_info):
        for k, v in pt_info.items():
            if hasattr(self, k):
                if k == "label" and not isinstance(v, list):
                    v = bool(v)
                setattr(self, k, v)
            else:
                print("Incorrect key:", k, "detected!")

    def store_image_data(self, pt_instances, proj_ids=None, segment_ids=None):
        for segment_id in self.segments:
            segment_data = []
            segment_instances = [pt_instances[i] for i in range(len(pt_instances)) if segment_ids[i] == segment_id]
            for i, instance in enumerate(segment_instances):
                projection_identifier = ProjectionType.AP if proj_ids[i] == "AP" else ProjectionType.LAT
                projection_path = self.dicom_folder_path + proj_ids[i] + "/" + instance
                img = cv2.imread(projection_path, cv2.IMREAD_GRAYSCALE)
                if not isinstance(self.label, list):
                    y = str(segment_id) if self.label is not None and self.label else ""
                else:
                    y = self.fracture_position[i] if self.label[i] else ""
                segment_data.append((projection_identifier, img, y))

            self.pt_data.append(segment_data)
        if isinstance(self.label, list):
            self.fracture_position = [self.fracture_position[i] for i in range(len(self.label)) if self.label[i]]
            self.label = any(self.label)
