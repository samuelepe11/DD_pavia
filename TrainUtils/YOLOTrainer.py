# Import packages
import os
import pandas as pd
import cv2
import yaml
import torch
from ultralytics import YOLO

from TrainUtils.NetworkTrainer import NetworkTrainer
from DataUtils.XrayDataset import XrayDataset
from Enumerators.SetType import SetType


# Class
class YOLOTrainer:
    def __init__(self, working_dir, train_data, val_data, test_data, model_name, n_classes=2):
        # Define variables
        self.working_dir = working_dir
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model_name = model_name
        self.n_classes = n_classes

        # Create YOLO directory and dataset
        self.yolo_dir = self.working_dir + "yolo_dir/"
        self.data_dir = self.yolo_dir + "dataset/"
        self.results_dir = self.yolo_dir + "results/"
        if "yolo_dir" not in os.listdir(working_dir):
            self.create_yolo_dirs_and_dataset()
            self.create_yaml_file()

        # Define model
        self.model = YOLO("yolov8n.pt")

    def train(self, epochs, img_size, batch, use_cuda=False):
        use_cuda = use_cuda and torch.cuda.is_available()
        if not use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.model.train(data=self.data_dir + "data.yaml", epochs=epochs, imgsz=img_size, batch=batch, device="cpu",
                         rect=True, workers=0)

    def create_yolo_dirs_and_dataset(self):
        print("Creating YOLO directories...")
        os.mkdir(self.yolo_dir)
        os.mkdir(self.data_dir)
        os.mkdir(self.data_dir + "images/")
        os.mkdir(self.data_dir + "labels/")
        os.mkdir(self.results_dir)

        for set_type in SetType:
            print("Creating YOLO", set_type.value, " set...")
            img_dir = self.data_dir + "images/" + set_type.value + "/"
            os.mkdir(img_dir)
            label_dir = self.data_dir + "labels/" + set_type.value + "/"
            os.mkdir(label_dir)

            # Get data and cropping references
            if set_type == SetType.TRAIN:
                data = self.train_data
            elif set_type == SetType.VAL:
                data = self.val_data
            else:
                data = self.test_data
            cropping_ref = pd.read_csv(
                self.working_dir + XrayDataset.data_fold + set_type.value + "_cropping_ref.csv")

            for patient in data.patient_data:
                for i, segment in enumerate(patient.segments):
                    instance_name = f"{patient.id:03}" + segment.lower()
                    for j, projection in enumerate(patient.pt_data[i]):
                        # Store image
                        cv2.imwrite(img_dir + instance_name + "_proj" + str(j) + ".png", projection[1])
                        # Store labels
                        desired_patches = cropping_ref[
                            (cropping_ref.segment == instance_name) & (cropping_ref.projection == j)]
                        with open(label_dir + instance_name + "_proj" + str(j) + ".txt", "a") as f:
                            for patch in desired_patches.itertuples(index=False):
                                full_width = patch.width
                                full_height = patch.height
                                x_center = (patch.x_min + patch.x_max) / 2
                                x_center /= full_width
                                y_center = (patch.y_min + patch.y_max) / 2
                                y_center /= full_height
                                width = patch.x_max - patch.x_min
                                width /= full_width
                                height = patch.y_max - patch.y_min
                                height /= full_height
                                f.write(f"{int(patch.fracture_present)} {x_center} {y_center} {width} {height}\n")

    def create_yaml_file(self):
        data_dict = {"path": self.data_dir, "train": "images/training", "val": "images/validation",
                     "test": "images/test", "nc": self.n_classes, "project": self.results_dir, "name": self.model_name}
        if self.n_classes == 2:
            data_dict["names"] = XrayDataset.classes
        else:
            data_dict["names"] = {"vertebra"}
        with open(self.data_dir + "data.yaml", "w") as f:
            yaml.dump(data_dict, f, sort_keys=False)


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"

    # Load data
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training")
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation")
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test")

    # Define trainer
    n_classes1 = 2
    trainer1 = YOLOTrainer(working_dir=working_dir1, train_data=train_data1, val_data=val_data1, test_data=test_data1,
                           n_classes=n_classes1)

    # Train model
    epochs1 = 100
    img_size1 = 1024
    batch1 = 16
    use_cuda1 = True
    trainer1.train(epochs=epochs1, img_size=img_size1, batch=batch1, use_cuda=use_cuda1)
