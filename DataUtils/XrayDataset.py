# Import packages
import pandas as pd
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset

from DataUtils.PatientInstance import PatientInstance
from Enumerators.ProjectionType import ProjectionType
from Enumerators.SetType import SetType


# Class
class XrayDataset(Dataset):
    # Define class attributes
    data_fold = "dati_s_matteo/"
    results_fold = "results/"
    preliminary_fold = "preliminary_analysis/"
    info_names = ["id", "sex", "birth", "segments", "projections", "spondylarthrosis_present", "label",
                  "fracture_position", "clinical_report", "ct_present", "ct_date", "ct_report", "mri_present",
                  "mri_date", "mri_report", "notes"]
    segment_dict = {"C": "cervical", "D": "thoracic", "L": "lumbar", "S": "sacral-coccygeal"}
    segment_dict_ita = {"C": "cervicale", "D": "dorsale", "L": "lombare", "S": "sacro-coccigeo"}
    classes = ["fracture absent", "fracture present"]

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.data_dir = working_dir + self.data_fold
        self.results_dir = working_dir + self.results_fold
        self.preliminary_dir = self.results_dir + self.preliminary_fold

        self.info_file_path = None
        self.dicom_folder_path = None
        self.info_file = None
        self.info_file_descr = None
        self.dicom_instances = None
        self.len = None
        self.patient_data = None

        self.set_type = None
        self.training_pts = []
        self.training_names = []
        self.validation_pts = []
        self.validation_names = []
        self.test_pts = []
        self.test_names = []

    def __getitem__(self, ind):
        if self.patient_data is None:
            print("DICOM data missing!")
            return None

        instance_name = self.dicom_instances[ind]
        pt_id, segment_id = PatientInstance.get_patient_and_segment(instance_name)
        pt_instance = self.get_patient(pt_id)
        segment_data = pt_instance.get_segment_images(segment_id=segment_id)
        return segment_data, (pt_id, segment_id)

    def __len__(self):
        return self.len

    def get_patient(self, pt_id):
        for pt in self.patient_data:
            if pt.id == pt_id:
                return pt
        else:
            print("Patient not found!")
            return None

    def read_info_file(self, info_file_name):
        self.info_file_path = self.data_dir + info_file_name
        self.info_file = pd.read_csv(self.info_file_path, delimiter=";", encoding="latin1")
        self.info_file_descr = self.info_file.columns
        self.info_file.columns = self.info_names

    def read_dicom_folder(self, dicom_folder_name):
        if self.info_file is None:
            print("Information file missing!")
            return

        self.dicom_folder_path = self.data_dir + dicom_folder_name
        self.dicom_instances = os.listdir(self.dicom_folder_path)
        self.len = len(self.dicom_instances)
        self.patient_data = []
        for pt in self.info_file["id"]:
            pt_info = self.info_file.loc[pt - 1]
            pt_label = f"{pt:03}"
            pt_instances = [instance for instance in self.dicom_instances if pt_label in instance]
            if len(pt_instances) != 0:
                self.patient_data.append(PatientInstance(pt_info, pt_instances, self.dicom_folder_path))
        bis_instances = [instance for instance in self.dicom_instances if "bis" in instance]
        self.dicom_instances = list(set(self.dicom_instances) - set(bis_instances))
        print("All the DICOM data have been read!")

    def store_dataset(self, dataset_name):
        file_path = self.data_dir + dataset_name + ".pt"
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        print("The dataset", dataset_name, "have been stored!")

    def show_item(self, ind):
        instance_name = self.dicom_instances[ind]
        pt, segment = PatientInstance.get_patient_and_segment(instance_name)
        title = "Patient " + str(pt) + " - " + XrayDataset.get_segment_name(segment) + " segment"
        item = self.__getitem__(ind)
        n_projections = len(item)

        plt.figure(figsize=(5 * n_projections, 5))
        plt.suptitle(title)

        for i in range(n_projections):
            projection_id, img, y = item[i]
            if y == "":
                label = "no fracture"
            else:
                label = y
            plt.subplot(1, n_projections, i + 1)
            plt.title(projection_id.value + " (" + label + ")")
            plt.imshow(img, cmap="gray")
        plt.show()

    def show_patient(self, pt_id):
        flag = False
        for ind in range(len(self.dicom_instances)):
            instance = self.dicom_instances[ind]
            pt_id_i, _ = PatientInstance.get_patient_and_segment(instance)
            if pt_id_i == pt_id:
                self.show_item(ind)
                flag = True
        if not flag:
            print("Patient", pt_id, "not found...")

    def count_data(self):
        # Define directory for preliminary evaluation
        if self.set_type is None:
            self.preliminary_dir += "pooled/"
        else:
            self.preliminary_dir += self.set_type.value + "/"

        # Count patients
        n_pt = len(self.patient_data)
        print("Number of patients:", n_pt)

        # Count fracture patients
        plt.figure(figsize=(15, 5))
        plt.suptitle("Fracture distributions")
        plt.subplot(1, 3, 1)
        frac_pt = len(XrayDataset.get_fracture_patients(self.patient_data))
        healthy_pt = n_pt - frac_pt
        XrayDataset.draw_pie_plot([healthy_pt, frac_pt], self.classes, "Fractures per patient", None)

        # Count segments
        n_segments = np.sum([len(pt.pt_data) for pt in self.patient_data])
        print("Number of segments:", n_segments)

        # Count fracture segments and projections
        n_frac_segments = 0
        n_projections = 0
        n_frac_projections = 0
        for pt in self.patient_data:
            for segment in pt.pt_data:
                # Count fracture segments
                if segment[0][2] != "":
                    n_frac_segments += 1

                # Count projections
                n_projections += len(segment)
                # Count fracture projections
                for projection in segment:
                    if projection[2] != "":
                        n_frac_projections += 1

        plt.subplot(1, 3, 2)
        n_healthy_segments = n_segments - n_frac_segments
        XrayDataset.draw_pie_plot([n_healthy_segments, n_frac_segments], self.classes,
                                  "Fractures per segment", None)

        print("Number of projections:", n_projections)
        plt.subplot(1, 3, 3)
        n_healthy_projections = n_projections - n_frac_projections
        XrayDataset.draw_pie_plot([n_healthy_projections, n_frac_projections], self.classes,
                                  "Fractures per projection", None)

        plt.savefig(self.preliminary_dir + "frac_distributions.jpg")
        plt.close()

        # Count segment types
        print("\nNumber of segments per segment type:")
        for s in self.segment_dict.keys():
            s_name = self.segment_dict[s]
            plt.figure(figsize=(10, 10))
            plt.suptitle("Distributions for " + s_name + " segment")

            s_pt = [pt for pt in self.patient_data if s in pt.segments]
            n_s_pt = len(s_pt)
            print(" -", s_name, n_s_pt)

            plt.subplot(2, 2, 1)
            n_s_pt_frac = len([pt for pt in XrayDataset.get_fracture_patients(s_pt) if s in
                               ".".join(pt.fracture_position)])
            n_s_pt_no_frac = n_s_pt - n_s_pt_frac
            XrayDataset.draw_pie_plot([n_s_pt_no_frac, n_s_pt_frac], self.classes,
                                      "Fracture distribution", None)

            # Count projection types per segment type
            n_ap = 0
            n_lat = 0
            n_frac_ap = 0
            n_frac_lat = 0
            for pt in s_pt:
                s_ind = pt.segments.index(s)
                try:
                    for projection in pt.pt_data[s_ind]:
                        if projection[0] == ProjectionType.AP:
                            n_ap += 1
                            if projection[2] != "":
                                n_frac_ap += 1
                        else:
                            n_lat += 1
                            if projection[2] != "":
                                n_frac_lat += 1
                except IndexError:
                    print("    > Issues with patient", pt.id)

            plt.subplot(2, 2, 2)
            projection_names = [ProjectionType.AP.value, ProjectionType.LAT.value]
            XrayDataset.draw_pie_plot([n_ap, n_lat], projection_names,
                                      "Projection type distribution", None)

            plt.subplot(2, 2, 3)
            XrayDataset.draw_pie_plot([n_ap - n_frac_ap, n_frac_ap], self.classes,
                                      "Fractures per projection " + projection_names[0],
                                      None)
            plt.subplot(2, 2, 4)
            XrayDataset.draw_pie_plot([n_lat - n_frac_lat, n_frac_lat], self.classes,
                                      "Fractures per projection " + projection_names[1],
                                      None)

            plt.savefig(self.preliminary_dir + "distribution_segment_" + s + ".jpg")
            plt.close()

        # Count projection types
        plt.figure(figsize=(10, 5))
        plt.suptitle("Fracture distribution per projection")
        print("\nNumber of projections per projection type:")
        i = 1
        for p in ProjectionType:
            n_p_pt = 0
            n_p_pt_frac = 0
            for pt in self.patient_data:
                for segment in pt.pt_data:
                    for projection in segment:
                        if projection[0] == p:
                            n_p_pt += 1

                            if projection[2] != "":
                                n_p_pt_frac += 1

            print(" -", p.value, n_p_pt)

            plt.subplot(1, 2, i)
            n_s_pt_no_frac = n_p_pt - n_p_pt_frac
            XrayDataset.draw_pie_plot([n_s_pt_no_frac, n_p_pt_frac], self.classes,
                                      "Fractures per projection " + p.value, None)
            i += 1
        plt.savefig(self.preliminary_dir + "frac_distributions_per_segment.jpg")

    def divide_dataset(self, train_perc):
        # Get training data
        n_pt = len(self.patient_data)
        n_train = int(np.round(n_pt * train_perc))

        indices = random.sample(range(n_pt), n_train)
        remaining_indices = list(set(range(n_pt)) - set(indices))
        self.find_names(SetType.TRAIN, indices)

        # Get validation data
        n_val = int(np.round((n_pt - n_train) / 2))
        indices = random.sample(remaining_indices, n_val)
        self.find_names(SetType.VAL, indices)

        # Get test data
        indices = list(set(remaining_indices) - set(indices))
        self.find_names(SetType.TEST, indices)

    def find_names(self, set_type, indices):
        self.__dict__[set_type.value + "_pts"] = []
        self.__dict__[set_type.value + "_names"] = []
        for ind in indices:
            pt_id = self.patient_data[ind].id
            self.__dict__[set_type.value + "_pts"].append(pt_id)
            self.__dict__[set_type.value + "_names"] += [instance for instance in self.dicom_instances
                                                         if f"{pt_id:03}" in instance]

    def define_set_type(self, set_type):
        self.set_type = set_type
        if set_type == SetType.VAL:
            self.dicom_instances = self.validation_names
            ref_pts = self.validation_pts
        elif set_type == SetType.TEST:
            self.dicom_instances = self.test_names
            ref_pts = self.test_pts
        else:
            # set_type == SetType.TRAIN
            self.dicom_instances = self.training_names
            ref_pts = self.training_pts
        self.len = len(self.dicom_instances)
        self.patient_data = [pt for pt in self.patient_data if pt.id in ref_pts]

    def max_projection_number(self):
        projection_numbers = []
        for pt in self.patient_data:
            for segment in pt.pt_data:
                projection_numbers.append(len(segment))

        return np.max(projection_numbers)

    @staticmethod
    def load_dataset(working_dir, dataset_name, set_type=None):
        file_path = working_dir + XrayDataset.data_fold + dataset_name + ".pt"
        with open(file_path, "rb") as file:
            dataset = pickle.load(file)
        print("The dataset", dataset_name, "have been loaded!")

        # Correct mistakes in previously stored classes
        if dataset.working_dir == "./../":
            addon = "./."
            for attr in dataset.__dict__.keys():
                val = dataset.__dict__[attr]
                if isinstance(val, str) and val.startswith("./../"):
                    dataset.__dict__[attr] = addon + val
        dataset.len = len(dataset.dicom_instances)

        # Correct mistakes in previously stored files
        for i in range(dataset.len):
            instance = dataset.dicom_instances[i]
            instance_list = list(instance)
            if "Copia" in instance:
                dataset.dicom_instances[i] = "".join(instance_list[:4])
            elif len(instance) > 4:
                dataset.dicom_instances[i] = "".join(instance_list[:3] + [instance_list[4]])

        # Correct mistakes in the original data files
        for datum in dataset.patient_data:
            if datum.id == 112:
                datum.pt_data = [datum.pt_data[0], datum.pt_data[2]]
            if datum.id == 335:
                datum.segments = ["C", datum.segments[0]]
            if datum.id == 117:
                datum.segments = ["C"] + datum.segments[:2]
            if datum.id == 81:
                datum.segments = [datum.segments[0], "L"]

        if set_type is not None:
            dataset.define_set_type(set_type)
            dataset.store_dataset(dataset_name=dataset_name + "_" + set_type.value)

        return dataset

    @staticmethod
    def get_segment_name(segment):
        return XrayDataset.segment_dict[segment]

    @staticmethod
    def get_segment_name_ita(segment):
        return XrayDataset.segment_dict_ita[segment]

    @staticmethod
    def get_fracture_patients(data_list):
        return [pt for pt in data_list if pt.label]

    @staticmethod
    def draw_pie_plot(data, labels, title, file_name):
        if file_name is not None:
            plt.figure()

        plt.pie(data, labels=labels, autopct="%1.1f%%")
        plt.title(title)

        if file_name is not None:
            plt.savefig(file_name, dpi=300)
            plt.close()


# Main
if __name__ == "__main__":
    # Set matplotlib
    plt.close("all")
    matplotlib.use("TkAgg")

    # Define seeds
    seed = 1
    random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    info_file_name1 = "database_fratture_vertebrali_rx.csv"
    dicom_folder_name1 = "RX colonne anonoimizzate/"
    dataset_name1 = "xray_dataset"

    # Initialize and store dataset
    # dataset1 = XrayDataset(working_dir=working_dir1)
    # dataset1.read_info_file(info_file_name=info_file_name1)
    # dataset1.read_dicom_folder(dicom_folder_name=dicom_folder_name1)

    # Load  dataset
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Divide dataset
    train_perc1 = 0.8
    # dataset1.divide_dataset(train_perc=train_perc1)

    # Store dataset
    dataset1.store_dataset(dataset_name=dataset_name1)

    # Compute statistics
    print()
    # dataset1.count_data()

    print()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Training set:")
    # dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1, set_type=SetType.TRAIN)
    # dataset1.count_data()
    print()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Validation set:")
    # dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1, set_type=SetType.VAL)
    # dataset1.count_data()
    print()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Test set:")
    # dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1, set_type=SetType.TEST)
    # dataset1.count_data()

    # Show items
    ind1 = 1
    # dataset1.show_item(ind=ind1)

    pt_id1 = 66
    # dataset1.show_patient(pt_id=pt_id1)
