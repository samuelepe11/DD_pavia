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
    instances_to_rotate_ccw = ["112c_1", "180d_1", "258c_1", "298c_1", "329c_1", "376c_1", "475d_1", "161l_1", "321d_1",
                               "321l_1", "069c_1", "069d_1", "069l_1"]
    instances_to_rotate_cw = ["329l_1"]
    instances_to_rotate_180 = ["305c_1"]
    instances_to_flip_v = ["329l_1"]
    instances_to_flip_h = []

    def __init__(self, working_dir, model_name, n_classes=2, train_data=None, val_data=None, test_data=None,
                 selected_model="yolov8n.pt", augment=False):
        # Define variables
        self.working_dir = working_dir
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model_name = model_name
        self.n_classes = n_classes
        self.selected_model = selected_model
        self.augment = augment

        # Create YOLO directory and dataset
        self.yolo_dir = self.working_dir + "yolo_dir/"
        os.makedirs(self.yolo_dir, exist_ok=True)
        data_fold = "dataset" if n_classes == 2 else "dataset_one_class"
        self.data_dir = self.yolo_dir + data_fold + "/"
        if data_fold not in os.listdir(self.yolo_dir):
            self.create_yolo_dirs_and_dataset()
        self.results_dir = self.yolo_dir + "results/"
        os.makedirs(self.results_dir, exist_ok=True)

        # Define model
        self.model = YOLO(selected_model)

    def train(self, epochs, img_size, batch, use_cuda=False, tune=False):
        if tune:
            self.model_name += "_tune"
        self.create_yaml_file()

        use_cuda = use_cuda and torch.cuda.is_available()
        if not use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = "cpu"
        else:
            device = 0

        if tune:
            self.model.tune(data=self.data_dir + self.model_name + ".yaml", epochs=50, iterations=100, imgsz=img_size,
                            batch=batch, device=device, project=self.results_dir, name=self.model_name)

            with open(self.results_dir + self.model_name + "/best_hyperparameters.yaml", "r") as f:
                best_params = yaml.safe_load(f)
            self.model.train(data=self.data_dir + self.model_name + ".yaml", epochs=epochs, imgsz=img_size, batch=batch,
                             device=device, project=self.results_dir, name=self.model_name,
                             single_cls=self.n_classes == 1, augment=self.augment, pretrained=True, **best_params)
        else:
            self.model.train(data=self.data_dir + self.model_name + ".yaml", epochs=epochs, imgsz=img_size, batch=batch,
                             device=device, project=self.results_dir, name=self.model_name, single_cls=self.n_classes==1,
                             augment=self.augment, degrees=90.0, translate=0.15, scale=0.3, shear=0.0, fliplr=0.4,
                             flipud=0.4, mosaic=0.3, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, erasing=0.2, patience=50,
                             pretrained=True, cos_lr=True, lr0=0.05)

    def create_yolo_dirs_and_dataset(self):
        print("Creating YOLO directories...")
        os.mkdir(self.data_dir)
        os.mkdir(self.data_dir + "images/")
        os.mkdir(self.data_dir + "labels/")
        os.mkdir(self.data_dir + "gt/")

        for set_type in SetType:
            print("Creating YOLO", set_type.value, " set...")
            img_dir = self.data_dir + "images/" + set_type.value + "/"
            os.mkdir(img_dir)
            label_dir = self.data_dir + "labels/" + set_type.value + "/"
            os.mkdir(label_dir)
            gt_dir = self.data_dir + "gt/" + set_type.value + "/"
            os.mkdir(gt_dir)

            # Get data and cropping references
            if set_type == SetType.TRAIN:
                data = self.train_data
            elif set_type == SetType.VAL:
                data = self.val_data
            else:
                data = self.test_data
            cropping_ref = pd.read_csv(
                self.working_dir + XrayDataset.data_fold + "xray_dataset_" + set_type.value + "_cropped_imgs/" + "pooled/pooled_cropping_ref.csv")

            for patient in data.patient_data:
                for i, segment in enumerate(patient.segments):
                    instance_name = f"{patient.id:03}" + segment.lower()

                    for j, projection in enumerate(patient.pt_data[i]):
                        # Store image
                        img = projection[1]
                        filename = instance_name + "_proj" + str(j) + ".png"
                        cv2.imwrite(img_dir + filename, img)

                        # Store labels
                        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        box_img = img.copy()
                        desired_patches = cropping_ref[
                            (cropping_ref.segment == instance_name) & (cropping_ref.projection == j)]
                        with open(label_dir + instance_name + "_proj" + str(j) + ".txt", "a") as f:
                            for patch in desired_patches.itertuples(index=False):
                                label = int(patch.fracture_present) if self.n_classes == 2 else 0
                                full_width = patch.width
                                full_height = patch.height
                                x_min = int(patch.x_min)
                                y_min = int(patch.y_min)
                                x_max = int(patch.x_max)
                                y_max = int(patch.y_max)

                                instance_key = instance_name + "_" + str(j)
                                if instance_key in self.instances_to_rotate_cw:
                                    tmp1 = x_min
                                    x_min = full_height - y_max
                                    tmp2 = y_min
                                    y_min = tmp1
                                    tmp3 = x_max
                                    x_max = full_height - tmp2
                                    y_max = tmp3
                                    tmp4 = full_width
                                    full_width = full_height
                                    full_height = tmp4
                                elif instance_key in self.instances_to_rotate_ccw:
                                    tmp1 = x_min
                                    x_min = y_min
                                    tmp2 = y_max
                                    y_min = full_width - x_max
                                    x_max = tmp2
                                    y_max = full_width - tmp1
                                    tmp4 = full_width
                                    full_width = full_height
                                    full_height = tmp4
                                elif instance_key in self.instances_to_rotate_180:
                                    tmp1 = x_min
                                    x_min = full_width - x_max
                                    tmp2 = y_min
                                    y_min = full_height - y_max
                                    x_max = full_width - tmp1
                                    y_max = full_height - tmp2

                                if instance_key in self.instances_to_flip_v:
                                    tmp1 = y_min
                                    y_min = full_height - y_max
                                    y_max = full_height - tmp1

                                if instance_key in self.instances_to_flip_h:
                                    tmp1 = x_min
                                    x_min = full_width - x_max
                                    x_max = full_width - tmp1

                                if instance_key == "329c_1":
                                    y_min -= 300
                                    y_max -= 300

                                x_center = (x_min + x_max) / 2
                                x_center /= full_width
                                y_center = (y_min + y_max) / 2
                                y_center /= full_height
                                width = patch.x_max - patch.x_min
                                width /= full_width
                                height = patch.y_max - patch.y_min
                                height /= full_height
                                f.write(f"{label} {x_center} {y_center} {width} {height}\n")

                                # Draw boundig-box in the images
                                color = (0, 0, 255) if label == 1 else (0, 255, 0)  # red = fracture, green = normal
                                cv2.rectangle(box_img, (x_min, y_min), (x_max, y_max), color, thickness=3,
                                              lineType=cv2.LINE_8)

                            alpha = 0.2
                            box_img = cv2.addWeighted(img, alpha, box_img, 1 - alpha, 0)
                            cv2.imwrite(gt_dir + filename, box_img)

    def create_yaml_file(self):
        if self.model_name + ".yaml" not in os.listdir(self.data_dir):
            print("Creating YAML file...")
            data_dict = {"path": self.data_dir, "train": "images/training", "val": "images/validation",
                         "test": "images/test", "nc": self.n_classes}
            if self.n_classes == 2:
                data_dict["names"] = XrayDataset.classes
            else:
                data_dict["names"] = ["vertebra"]
            with open(self.data_dir + self.model_name + ".yaml", "w") as f:
                yaml.dump(data_dict, f, sort_keys=False)

    def save_predictions(self, dataset_name, tune=False):
        addon = "_tune" if tune else ""
        pred_dir = self.results_dir + self.model_name + addon + "/pred/" + dataset_name + "/"
        os.makedirs(pred_dir, exist_ok=True)
        img_dir = self.data_dir + "/images/" + dataset_name
        results = self.model.predict(source=img_dir, stream=True)
        for r in results:
            img = r.orig_img.copy()
            img = cv2.normalize(img[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            box_img = img.copy()
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if self.n_classes == 2:
                        cls = int(box.cls[0])
                        color = (0, 0, 255) if cls == 1 else (0, 255, 0)
                    else:
                        color = (0, 0, 255) if conf < 0.3 else (255, 0, 0) if conf > 0.7 else (0, 255, 0)
                    cv2.rectangle(box_img, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_8)
                    cv2.putText(box_img, f"{conf:.2f}", (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                                2, color, 3)
            alpha = 0.2
            out = cv2.addWeighted(img, alpha, box_img, 1 - alpha, 0)
            filename = os.path.basename(r.path)
            cv2.imwrite(pred_dir + filename, out)

    @staticmethod
    def load_model(working_dir, model_name, n_classes):
        trainer = YOLOTrainer(working_dir=working_dir, model_name=model_name, n_classes=n_classes)
        weights_path = os.path.join(trainer.results_dir, model_name, "weights", "best.pt")
        trainer.model = YOLO(weights_path)
        return trainer

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
    model_name1 = "yolo"
    n_classes1 = 1
    selected_model1 = "yolov8x.pt"
    augment1 = True
    trainer1 = YOLOTrainer(working_dir=working_dir1, train_data=train_data1, val_data=val_data1, test_data=test_data1,
                           model_name=model_name1, n_classes=n_classes1, selected_model=selected_model1,
                           augment=augment1)

    # Train model
    epochs1 = 300
    img_size1 = 1024
    batch1 = 8
    use_cuda1 = True
    tune1 = False
    #trainer1.train(epochs=epochs1, img_size=img_size1, batch=batch1, use_cuda=use_cuda1, tune=tune1)

    # Load and test model
    trainer1 = YOLOTrainer.load_model(working_dir=working_dir1, model_name=model_name1, n_classes=n_classes1)
    trainer1.save_predictions("training", tune=tune1)
    trainer1.save_predictions("validation", tune=tune1)
    trainer1.save_predictions("test", tune=tune1)