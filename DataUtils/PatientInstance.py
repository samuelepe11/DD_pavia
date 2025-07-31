# Import packages
import os
from pydicom import dcmread

from Enumerators.ProjectionType import ProjectionType


# Class
class PatientInstance:
    def __init__(self, pt_info, pt_instances, dicom_folder_path, proj_ids=None, segment_ids=None):
        self.dicom_folder_path = dicom_folder_path

        self.id = None
        self.sex = None
        self.birth = None
        self.acquisition_date = None
        self.age = None
        self.spondylarthrosis_present = None
        self.segments = None
        self.projections_expected = None
        self.label = None
        self.fracture_position = None
        self.clinical_report = None
        self.notes = None
        self.pt_data = []

        # Store patient generalities
        self.store_patient_generalities(pt_info)

        # Read DICOM data
        self.store_image_data(pt_instances, proj_ids, segment_ids)

    def store_patient_generalities(self, pt_info):
        self.id = pt_info["id"]
        self.sex = pt_info["sex"]
        self.birth = pt_info["birth"]
        self.spondylarthrosis_present = pt_info["spondylarthrosis_present"]

        segments = pt_info["segments"].split("-")
        self.segments = [segment.strip() for segment in segments]
        projections_expected = pt_info["projections"].replace("feb", "2")
        projections_expected = projections_expected.split("-")
        self.projections_expected = [int(projection) for projection in projections_expected]

        self.label = bool(pt_info["label"])
        if self.label:
            self.fracture_position = pt_info["fracture_position"].split("-")
        else:
            self.fracture_position = None
        self.clinical_report = pt_info["clinical_report"]

        for mode in ["ct", "mri"]:
            present = mode + "_present"
            date = mode + "_date"
            report = mode + "_report"
            self.__dict__[present] = bool(pt_info[present])
            if self.__dict__[present]:
                self.__dict__[date] = pt_info[date]
                self.__dict__[report] = pt_info[report]
            else:
                self.__dict__[date] = None
                self.__dict__[report] = None

        self.notes = pt_info["notes"]
        if self.notes == "nan":
            self.notes = None

    def store_image_data(self, pt_instances, proj_ids=None, segment_ids=None):
        for instance in pt_instances:
            _, segment_id = PatientInstance.get_patient_and_segment(instance)
            if "bis" in instance:
                bis = True
            else:
                bis = False
            segment_folder = self.dicom_folder_path + instance + "/DICOM/"
            projections = os.listdir(segment_folder)
            if len(projections) == 0:
                print("Segment", segment_id, "is missing for patient", self.id, "...")
                continue
            segment_data = []
            for projection in projections:
                projection_file = segment_folder + projection
                dicom_data = dcmread(projection_file)

                # Get image information from DICOM file
                if "a.p." in dicom_data.SeriesDescription:
                    projection_identifier = ProjectionType.AP
                elif "lat" in dicom_data.SeriesDescription:
                    projection_identifier = ProjectionType.LAT
                else:
                    continue
                img = dicom_data.pixel_array

                if self.acquisition_date is None:
                    self.acquisition_date = PatientInstance.adjust_date(dicom_data.AcquisitionDate)
                    self.compute_age()

                # Get segment label
                y = ""
                if self.label:
                    temp_y = [pos for pos in self.fracture_position if segment_id in pos]
                    y = "-".join(temp_y)

                segment_data.append((projection_identifier, img, y))

            if bis:
                self.pt_data[-1] += segment_data
            else:
                self.pt_data.append(segment_data)

    def get_segment_images(self, segment_id):
        if segment_id not in self.segments:
            print("Patient", self.id, "does not have segment", segment_id)
            return None

        segment_index = self.segments.index(segment_id)
        segment_data = self.pt_data[segment_index]
        return segment_data

    def compute_age(self):
        b_day, b_month, b_year = map(int, self.birth.split("/"))
        a_day, a_month, a_year = map(int, self.acquisition_date.split("/"))
        self.age = a_year - b_year - ((a_month, a_day) < (b_month, b_day))

    @staticmethod
    def adjust_date(date):
        year = "".join(list(date)[:4])
        month = "".join(list(date)[4:6])
        day = "".join(list(date)[6:])
        return day + "/" + month + "/" + year

    @staticmethod
    def get_patient_and_segment(instance_name):
        instance_name = list(instance_name)
        segment = instance_name[-1].upper()
        patient = int("".join(instance_name[:-1]))
        return patient, segment
