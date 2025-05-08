# Import packages
import os

from flatbuffers.flexbuffers import Object

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from signal_grad_cam import TorchCamBuilder

from Enumerators.SetType import SetType
from DataUtils.XrayDataset import XrayDataset
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
        '''self.trainer.summarize_performance(show_test=True, show_process=True, show_cm=True, assess_calibration=True)
        self.aggregate_evals()'''

        # Define CAM builder
        model = self.trainer.net.to("cuda")
        self.cam_builder = TorchCamBuilder(model=model, transform_fn=MapGenerator.preprocess_fn,
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
        data_list = []
        data_labels = []
        extras1 = []
        projection_types = []
        extra_preprocess_inputs_list = []
        for i, instance in enumerate(data):
            item, extra = instance
            extras1.append(extra[1])
            projection_type, projection, _ = item[0]
            data_list.append(projection[np.newaxis, :, :])
            projection_types.append(projection_type)
            data_labels.append(int((item[0][-1] != "")))
            extra_preprocess_inputs_list.append([self.trainer, projection_type, extra, set_type])
        extra_inputs_list = [extras1, projection_types]

        # Get CAMs
        self.cam_builder.get_cam(data_list, data_labels, target_classes, explainer_types, target_layers,
                                 softmax_final=True, data_names=all_names, results_dir_path=self.jai_dir,
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

    @staticmethod
    def preprocess_fn(item, trainer, projection_type, extra, set_type):
        projection_batch = item[np.newaxis, :, :, :]
        projection_type_batch = np.array([projection_type], dtype=Object)
        extra = [[extra_element] for extra_element in extra]
        input = trainer.preprocess_fn(projection_batch, projection_type_batch, extra, set_type)

        return input[0]

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
