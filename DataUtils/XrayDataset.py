# Import packages
import cv2
import pandas as pd
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from torch.utils.data import Dataset
from datetime import datetime

from DataUtils.PatientInstance import PatientInstance
from DataUtils.ExtraPatientInstance import ExtraPatientInstance
from Enumerators.ProjectionType import ProjectionType
from Enumerators.SetType import SetType
from Enumerators.ExtraDatasetType import ExtraDatasetType


# Class
class XrayDataset(Dataset):
    # Define class attributes
    data_fold = "dati_s_matteo/"
    results_fold = "results/"
    models_fold = "models/"
    preliminary_fold = "preliminary_analysis/"
    jai_fold = "jai_results/"
    extra_data_fold = "data_extra/"

    info_names = ["id", "sex", "birth", "segments", "projections", "spondylarthrosis_present", "label",
                  "fracture_position", "clinical_report", "ct_present", "ct_date", "ct_report", "mri_present",
                  "mri_date", "mri_report", "notes"]
    segment_dict = {"C": "cervical", "D": "thoracic", "L": "lumbar", "S": "sacral-coccygeal"}
    segment_id_list = list(segment_dict.keys())
    segment_dict_ita = {"C": "cervicale", "D": "dorsale", "L": "lombare", "S": "sacro-coccigeo"}
    classes = ["fracture absent", "fracture present"]

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.data_dir = working_dir + self.data_fold
        self.results_dir = working_dir + self.results_fold
        self.preliminary_dir = self.results_dir + self.preliminary_fold
        self.jai_dir = self.results_dir + self.jai_fold
        self.extra_data_dir = working_dir + self.extra_data_fold

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
        item, _ = self.__getitem__(ind)
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

    def count_data(self, is_extra=False, is_cropped=False):
        # Define directory for preliminary evaluation
        if self.set_type is None:
            self.preliminary_dir += "pooled/"
        else:
            if is_extra:
                addon = "extra_"
            elif is_cropped:
                addon = "cropped_"
            else:
                addon = ""
            self.preliminary_dir += addon + self.set_type.value + "/"

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
                flag = False
                for projection in segment:
                    if projection[2] != "":
                        flag = True
                if flag:
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

    def get_data_from_name(self, name):
        ind = self.dicom_instances.index(name)
        return self.__getitem__(ind)

    def complement_with_extra_data(self, extra_dataset_type):
        dataset_name = extra_dataset_type.get_dataset_name()
        if extra_dataset_type != ExtraDatasetType.CROPPED:
            sub_fold = self.extra_data_fold
        else:
            sub_fold = self.data_fold
            sub_fold += "xray_dataset_" + self.set_type.value + "_cropped_imgs/"
        dataset_path = self.working_dir + sub_fold + dataset_name + "/"

        if extra_dataset_type != ExtraDatasetType.CROPPED:
            ref_id_start = extra_dataset_type.get_ref_id_start()
            img_names = []
            img_proj_ids = []
            csv_pool = []
            for proj_id in os.listdir(dataset_path):
                if proj_id.endswith(".csv"):
                    csv_pool.append(pd.read_csv(dataset_path + proj_id))
                    continue
                files = os.listdir(dataset_path + proj_id)
                img_names += files
                img_proj_ids += [proj_id] * len(files)
            if len(csv_pool) > 0:
                csv_pool = pd.concat(csv_pool, ignore_index=True)

            # Detect subject IDs
            if extra_dataset_type == ExtraDatasetType.BUU:
                img_segm_ids = ["L"] * len(img_names)
                pt_ids = np.unique([int(name[:4]) for name in img_names])

                new_instances = []
                for pt_id in pt_ids:
                    new_id = ref_id_start + pt_id
                    new_instances.append(str(new_id) + "l")
                    pt_instances = [name for name in img_names if int(name[:4]) == pt_id]
                    img_segm_pt = [segm_id for i, segm_id in enumerate(img_segm_ids) if img_names[i] in pt_instances]
                    img_proj_pt = [proj_id for i, proj_id in enumerate(img_proj_ids) if img_names[i] in pt_instances]
                    pt_info = {"id": new_id, "sex": pt_instances[0][5], "age": int(pt_instances[0][7:10]), "segments": ["L"],
                               "label": 0}
                    self.patient_data.append(ExtraPatientInstance(pt_info, pt_instances, dataset_path, img_proj_pt,
                                                                  img_segm_pt))

            elif extra_dataset_type == ExtraDatasetType.AASCE:
                img_segm_ids = ["D"] * len(img_names)
                pt_ids = []
                for name in img_names:
                    tmp_id = name.split("-")[-1][:-4]
                    if "coronal" in name:
                        tmp_id = tmp_id[7:]
                    pt_ids.append(int(tmp_id.split(" ")[0]))
                pt_ids = np.unique(pt_ids)

                new_instances = []
                for pt_id in pt_ids:
                    new_id = ref_id_start + pt_id
                    new_instances.append(str(new_id) + "d")
                    pt_instances = []
                    for name in img_names:
                        tmp_id = name.split("-")[-1][:-4]
                        if "coronal" in name:
                            tmp_id = tmp_id[7:]
                        if int(tmp_id.split(" ")[0]) == pt_id:
                            pt_instances.append(name)
                    img_segm_pt = [segm_id for i, segm_id in enumerate(img_segm_ids) if img_names[i] in pt_instances]
                    img_proj_pt = [proj_id for i, proj_id in enumerate(img_proj_ids) if img_names[i] in pt_instances]

                    date = "-".join(pt_instances[0].split("-")[-4:-1])
                    try:
                        date = datetime.strptime(date, "%d-%B-%Y")
                    except ValueError:
                        date = datetime.strptime(date, "%d-%b-%Y")
                    date = date.strftime("%d/%m/%y")
                    pt_info = {"id": new_id, "segments": ["D"], "acquisition_date": date, "label": 0}
                    self.patient_data.append(ExtraPatientInstance(pt_info, pt_instances, dataset_path, img_proj_pt,
                                                                  img_segm_pt))

            elif extra_dataset_type == ExtraDatasetType.DD:
                pt_ids = []
                img_segm_ids = []
                img_labels = []
                vertebra_names = []
                for name in img_names:
                    if name[0] == "x":
                        pt_ids.append(int(name.split("_")[2].split(".")[0]) + 500)
                        img_segm_ids.append("L")
                        img_labels.append("L")
                        vertebra_names.append("L")
                    else:
                        sep = "-" if "-" in name else "_"
                        pt_ids.append(int(name.split(sep)[0]))
                        tmp_label = name.split("_")[-1][:-4]
                        if tmp_label[0] == "T":
                            tmp_label = "D" + tmp_label[1:]
                        img_segm_ids.append(tmp_label[0])

                        label = bool(csv_pool.loc[csv_pool["image name"] == name, "ground truth"].values)
                        img_labels.append(label)
                        vertebra_names.append(tmp_label)
                pt_ids = np.unique(pt_ids)

                new_instances = []
                for pt_id in pt_ids:
                    new_id = ref_id_start + pt_id
                    pt_instances = []
                    for name in img_names:
                        if name[0] == "x":
                            tmp_id = int(name.split("_")[2].split(".")[0]) + 500
                        else:
                            sep = "-" if "-" in name else "_"
                            tmp_id = int(name.split(sep)[0])
                        if tmp_id == pt_id:
                            pt_instances.append(name)

                    img_segm_pt = []
                    img_proj_pt = []
                    img_label_pt = []
                    vertebra_name_pt = []
                    for i in range(len(img_segm_ids)):
                        if img_names[i] in pt_instances:
                            img_segm_pt.append(img_segm_ids[i])
                            img_proj_pt.append(img_proj_ids[i])
                            img_label_pt.append(img_labels[i])
                            vertebra_name_pt.append(vertebra_names[i])
                    segments = np.unique(img_segm_ids).tolist()
                    for segm_id in segments:
                        new_instances.append(str(new_id) + segm_id.lower())

                    pt_info = {"id": new_id, "segments": segments, "label": img_label_pt, "fracture_position": vertebra_names}
                    self.patient_data.append(ExtraPatientInstance(pt_info, pt_instances, dataset_path, img_proj_pt,
                                                                  img_segm_pt))
            else:
                new_instances = []
                print(extra_dataset_type, "is not available!")

            self.dicom_instances += new_instances
            self.len += len(new_instances)
            self.training_pts = [pt_datum.id for pt_datum in self.patient_data]

        else:
            cropping_ref = pd.read_csv(dataset_path + "pooled_cropping_ref.csv")
            new_patient_data = self.patient_data.copy()
            for i, patient in enumerate(self.patient_data):
                new_pt_data = []
                for j, segment in enumerate(patient.segments):
                    new_segment_data = []
                    instance_name = f"{patient.id:03}" + segment.lower()
                    desired_patches = cropping_ref[cropping_ref.segment == instance_name]
                    if len(desired_patches) > 0:
                        for patch_ref in desired_patches.itertuples(index=False):
                            projection_type = ProjectionType.AP if patch_ref.projection_type == "antero-posterior" \
                                else ProjectionType.LAT
                            img = cv2.imread(dataset_path + patch_ref.file_name, cv2.IMREAD_GRAYSCALE)
                            w = patch_ref.x_max - patch_ref.x_min + 1
                            h = patch_ref.y_max - patch_ref.y_min + 1
                            img = cv2.resize(img, (w, h))
                            label = "" if not patch_ref.fracture_present else patch_ref.vertebra_name
                            new_segment_data.append((projection_type, img, label, str(patch_ref)))
                        new_pt_data.append(new_segment_data)
                    else:
                        new_pt_data.append(self.patient_data[i].pt_data[j])
                new_patient_data[i].pt_data = new_pt_data
            self.patient_data = new_patient_data

    @staticmethod
    def load_dataset(working_dir, dataset_name, set_type=None, s3=None, selected_segments=None,
                     selected_projection=None, correct_mistakes=True):
        file_path = working_dir + XrayDataset.data_fold + dataset_name + ".pt"
        file = open(file_path, "rb") if s3 is None else s3.open(file_path, "rb")
        dataset = pickle.load(file)
        print("The dataset", dataset_name, "have been loaded!")

        # Select segment
        if selected_segments is not None:
            instances = []
            for segm in selected_segments:
                instances = [instance for instance in dataset.dicom_instances if segm in list(instance)[3:]]
            dataset.dicom_instances = instances
            dataset.len = len(dataset.dicom_instances)

        # Correct mistakes in previously stored classes
        dataset.working_dir = working_dir
        for attr in dataset.__dict__.keys():
            val = dataset.__dict__[attr]
            if isinstance(val, str):
                if val.startswith("./../../"):
                    dataset.__dict__[attr] = working_dir + val[8:]
                elif val.startswith("./../"):
                    dataset.__dict__[attr] = working_dir + val[5:]
        dataset.len = len(dataset.dicom_instances)

        # Correct mistakes in previously stored files
        if correct_mistakes:
            for i in range(dataset.len):
                instance = dataset.dicom_instances[i]
                instance_list = list(instance)
                if "Copia" in instance:
                    dataset.dicom_instances[i] = "".join(instance_list[:4])
                elif len(instance) > 4:
                    dataset.dicom_instances[i] = "".join(instance_list[:3] + [instance_list[4]])

        # Correct mistakes in the original data files
        if correct_mistakes:
            for datum in dataset.patient_data:
                if datum.fracture_position is not None:
                    frac_pos = []
                    for pos in datum.fracture_position:
                        new_pos = pos.replace(" ", "").replace("...", "")
                        if "Tratto" in pos:
                            poses = pos.split("Tratto")
                            frac_pos.append(poses[0])
                            new_pos = poses[1]
                        frac_pos.append(new_pos)

                for i in range(len(datum.pt_data)):
                    segm = datum.pt_data[i]
                    for j in range(len(segm)):
                        temp = list(segm[j])
                        temp[2] = temp[2].replace("...", "").replace(" ", "")
                        if "Tratto" in temp[2]:
                            temp1 = temp[2].split("Tratto")
                            temp[2] = temp1[0] if datum.segments[i] in temp1[0] else temp1[1]
                        datum.pt_data[i][j] = tuple(temp)
                if datum.id == 81:
                    datum.segments = [datum.segments[0], "L"]
                if datum.id == 335:
                    datum.segments = ["C", datum.segments[0]]
                if datum.id == 112:
                    datum.pt_data = [datum.pt_data[0], datum.pt_data[2]]
                if datum.id == 117:
                    datum.segments = ["C"] + datum.segments[:2]
                if datum.id == 161:
                    datum.pt_data[3].append(datum.pt_data[2][2])
                    datum.pt_data[3].append(datum.pt_data[2][3])
                    datum.pt_data[2] = datum.pt_data[2][:2]

        # Select projection
        if selected_projection is not None:
            temp_patient_data = []
            for instance in dataset.patient_data:
                temp_instance = []
                temp_segments = []
                for i, segm in enumerate(instance.pt_data):
                    temp_segm = [proj for proj in segm if proj[0] == selected_projection]
                    temp_instance_name = f"{instance.id:03}" + instance.segments[i].lower()
                    if len(temp_segm) == 0:
                        if temp_instance_name in dataset.dicom_instances:
                            dataset.dicom_instances.remove(f"{instance.id:03}" + instance.segments[i].lower())
                    else:
                        temp_instance.append(temp_segm)
                        temp_segments.append(instance.segments[i])
                instance.pt_data = temp_instance
                instance.segments = temp_segments
                temp_patient_data.append(instance)
            dataset.patient_data = temp_patient_data
            dataset.len = len(dataset.patient_data)

        if set_type is not None:
            dataset.define_set_type(set_type)
            dataset.store_dataset(dataset_name=dataset_name + "_" + set_type.value)
        
        if s3 is not None:
            dataset.working_dir = "s3://dd-s-matteo-dev-resources/"
            dataset.data_dir = dataset.working_dir + dataset.data_fold
            dataset.results_dir = dataset.working_dir + dataset.results_fold
            dataset.preliminary_dir = dataset.results_dir + dataset.preliminary_fold

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
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    info_file_name1 = "database_fratture_vertebrali_rx.csv"
    dicom_folder_name1 = "RX colonne anonoimizzate/"
    dataset_name1 = "xray_dataset"

    # Initialize and store dataset
    # dataset1 = XrayDataset(working_dir=working_dir1)
    # dataset1.read_info_file(info_file_name=info_file_name1)
    # dataset1.read_dicom_folder(dicom_folder_name=dicom_folder_name1)

    # Load  dataset
    # dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Divide dataset
    train_perc1 = 0.8
    # dataset1.divide_dataset(train_perc=train_perc1)

    # Store dataset
    # dataset1.store_dataset(dataset_name=dataset_name1)

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

    # Load an already split datasets
    dataset_name1 = "xray_dataset_training"
    '''dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1, selected_segments=None,
                                        selected_projection=None)'''

    # Show items
    ind1 = 1
    # dataset1.show_item(ind=ind1)

    pt_id1 = 161
    # dataset1.show_patient(pt_id=pt_id1)

    # Extend training set
    extra_dataset_types1 = [ExtraDatasetType.BUU, ExtraDatasetType.AASCE, ExtraDatasetType.DD]
    # extra_dataset_types1 = [ExtraDatasetType.CROPPED]
    '''for extra_dataset_type1 in extra_dataset_types1:
        print("Processing", extra_dataset_type1.value + "...")
        dataset1.complement_with_extra_data(extra_dataset_type=extra_dataset_type1)'''
    addon1 = "cropped_" if ExtraDatasetType.CROPPED in extra_dataset_types1 else "extended_"
    # dataset1.store_dataset(dataset_name=addon1 + dataset_name1)

    print("-----------------------------------------------------------------------------------------------------------")
    addon2 = "Cropped" if ExtraDatasetType.CROPPED in extra_dataset_types1 else "Extended"
    print(addon2, "training set:")
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon1 + dataset_name1,
                                        selected_segments=None, selected_projection=None, correct_mistakes=False)

    is_cropped1 = ExtraDatasetType.CROPPED in extra_dataset_types1
    is_extra1 = not is_cropped1
    dataset1.count_data(is_extra=is_extra1, is_cropped=is_cropped1)
