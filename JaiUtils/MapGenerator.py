# Import packages
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from signal_grad_cam import TorchCamBuilder

from Enumerators.SetType import SetType
from Enumerators.ProjectionType import ProjectionType
from DataUtils.XrayDataset import XrayDataset
from DataUtils.PatientInstance import PatientInstance
from TrainUtils.NetworkTrainer import NetworkTrainer


# Class
class MapGenerator:
    def __init__(self, working_dir, model_name, trial_n, use_cuda=False, projection_dataset=False,
                 selected_segments=None, selected_projection=None):
        # Initialize attributes
        self.working_dir = working_dir
        self.jai_dir = working_dir + XrayDataset.results_fold + XrayDataset.jai_fold
        self.model_name = model_name
        self.trial_n = trial_n
        self.use_cuda = use_cuda
        self.projection_dataset = projection_dataset

        # Load data
        self.train_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name="xray_dataset_training",
                                                   selected_segments=selected_segments,
                                                   selected_projection=selected_projection)
        self.val_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name="xray_dataset_validation",
                                                 selected_segments=selected_segments,
                                                 selected_projection=selected_projection)
        self.test_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name="xray_dataset_test",
                                                  selected_segments=selected_segments,
                                                  selected_projection=selected_projection)

        # Load model
        self.trainer = NetworkTrainer.load_model(working_dir=working_dir, model_name=model_name, trial_n=trial_n,
                                                 use_cuda=use_cuda, train_data=self.train_data, val_data=self.val_data,
                                                 test_data=self.test_data, projection_dataset=projection_dataset)

        # Assess model
        self.trainer.summarize_performance(show_test=True, show_process=True, show_cm=True, assess_calibration=True)
        self.aggregate_evals()

        # Define CAM builder
        self.cam_builder = TorchCamBuilder(model=self.trainer.net, transform_fn=NetworkTrainer.preprocess_fn,
                                           class_names=self.train_data.classes, input_transposed=True, use_gpu=use_cuda)

    def aggregate_evals(self):
        if self.projection_dataset:
            for type in SetType:
                filepath = self.trainer.results_dir + type.value + "_classification_results.csv"
                df = pd.read_csv(filepath)
                df["segm_descr"] = df["descr"].str.extract(r"^([a-zA-Z0-9]+)_")
                df["y_pred"] = df["y_pred"].astype(int)
                df["y_true"] = df["y_true"].astype(int)
                results = df.groupby("segm_descr").agg(
                    y_true=("y_true", "first"), majority_vote=("y_pred", MapGenerator.majority_vote),
                    worst_case_opt=("y_pred", lambda x: MapGenerator.worst_case_vote(x, mode="optimistic")),
                    worst_case_pess=("y_pred", lambda x: MapGenerator.worst_case_vote(x, mode="pessimistic")),
                    ).reset_index()

                # Store results
                self.trainer.results_dir + type.value + "aggregated_classification_results.csv"
                results.to_csv(filepath, index=False)

                # Display aggregated accuracies
                print()
                print("Voting performances on the " + type.value.upper() + " set:")
                majority_acc = (results["majority_vote"] == results["y_true"]).mean()
                majority_f1 = f1_score(results["y_true"], results["majority_vote"], average="binary")
                majority_mcc = matthews_corrcoef(results["y_true"], results["majority_vote"])
                print(f" - Majority vote accuracy = {majority_acc * 100:.2f}%, F1-score = {majority_f1 * 100:.2f}%, and"
                      f" MCC = {majority_mcc:.3f}")
                opt_acc = (results["worst_case_opt"] == results["y_true"]).mean()
                opt_f1 = f1_score(results["y_true"], results["worst_case_opt"], average="binary")
                opt_mcc = matthews_corrcoef(results["y_true"], results["worst_case_opt"])
                print(f" - Optimistic vote accuracy = {opt_acc * 100:.2f}%, F1-score = {opt_f1 * 100:.2f}%, and MCC = "
                      f"{opt_mcc:.3f}")
                pess_acc = (results["worst_case_pess"] == results["y_true"]).mean()
                pess_f1 = f1_score(results["y_true"], results["worst_case_pess"], average="binary")
                pess_mcc = matthews_corrcoef(results["y_true"], results["worst_case_pess"])
                print(f" - Pessimistic case accuracy = {pess_acc * 100:.2f}%, F1-score = {pess_f1 * 100:.2f}%, and MCC "
                      f"= {pess_mcc:.3f}")

    def get_cam(self, set_type, target_classes, explainer_types, target_layers):
        # Choose data
        data, _, _ = self.trainer.select_dataset(set_type)
        loader, _ = self.trainer.load_data(data, shuffle=False)
        all_names = data.dicom_instances if not self.projection_dataset else data.dicom_projection_instances

        # Request cams
        for i, instance in enumerate(loader):
            item, extra = instance
            projection_type_batch, projection_batch, _ = item
            data_list = list(projection_batch[np.newaxis, :, :, :])
            data_labels = list((item[-1] != "").astype(int))
            extra_preprocess_inputs_list = [self.trainer, projection_type_batch, extra, set_type]
            extra_inputs_list = [extra[1], projection_type_batch]
            data_names = all_names[i * self.trainer.net_params["batch_size"] : (i + 1) * self.trainer.net_params["batch_size"]]

            # Get shapes
            data_shape_list = []
            for name in data_names:
                instance_name, ind = name.split("_")
                pt_id, segment_id = PatientInstance.get_patient_and_segment(instance_name)
                pt_instance = data.get_patient(pt_id)
                segment_data = pt_instance.get_segment_images(segment_id=segment_id)
                data_shape_list.append(segment_data[int(ind)][1].shape)

            # Get CAMs
            self.cam_builder.get_cam(data_list, data_labels, target_classes, explainer_types, target_layers,
                                     softmax_final=True, data_names=data_names, results_dir_path=self.jai_dir,
                                     data_shape_list=data_shape_list,
                                     extra_preprocess_inputs_list=extra_preprocess_inputs_list,
                                     extra_inputs_list=extra_inputs_list)

    @staticmethod
    def majority_vote(x):
        counts = x.value_counts()
        if len(counts) > 1:
            total = len(x)
            for val, count in counts.items():
                if count >= total / 2:
                    return val
        else:
            return counts.index[0]

    @staticmethod
    def worst_case_vote(x, mode):
        ref = 1 if mode == "pessimistic" else 0
        return int((x == ref).any())

# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "projection_resnext101_optuna"
    trial_n1 = 55
    use_cuda1 = True
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None

    # Define generator
    generator1 = MapGenerator(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1, use_cuda=use_cuda1,
                              projection_dataset=projection_dataset1, selected_segments=selected_segments1,
                              selected_projection=selected_projection1)

    # Draw maps
    set_type1 = SetType.VAL
    target_classes1 = [0, 1]
    explainer_types1 = "Grad-CAM"
    target_layers1 = "feature_extractor.features.7.2.conv3"
    generator1.get_cam(set_type=set_type1, target_classes=target_classes1, explainer_types=explainer_types1,
                       target_layers=target_layers1)
