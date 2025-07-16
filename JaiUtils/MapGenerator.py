# Import packages
import os

from flatbuffers.flexbuffers import Object

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import f1_score, matthews_corrcoef
from signal_grad_cam import TorchCamBuilder

from Enumerators.SetType import SetType
from DataUtils.XrayDataset import XrayDataset
from TrainUtils.NetworkTrainer import NetworkTrainer
from Networks.PretrainedFeatureExtractor import PretrainedFeatureExtractor


# Class
class MapGenerator:
    def __init__(self, working_dir, model_name, trial_n, use_cuda=False, projection_dataset=False,
                 selected_segments=None, selected_projection=None):
        # Initialize attributes
        self.working_dir = working_dir
        self.jai_dir = working_dir + XrayDataset.results_fold + XrayDataset.jai_fold
        self.model_name = model_name
        if model_name not in os.listdir(self.jai_dir):
            os.mkdir(self.jai_dir + model_name)
        self.jai_dir += model_name + "/"

        self.trial_n = trial_n
        self.use_cuda = use_cuda
        self.projection_dataset = projection_dataset

        # Load data
        '''self.train_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name="xray_dataset_training",
                                                   selected_segments=selected_segments,
                                                   selected_projection=selected_projection)'''
        self.val_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name="xray_dataset_validation",
                                                 selected_segments=selected_segments,
                                                 selected_projection=selected_projection)
        '''self.test_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name="xray_dataset_test",
                                                  selected_segments=selected_segments,
                                                  selected_projection=selected_projection)'''

        # Load model
        self.trainer = NetworkTrainer.load_model(working_dir=working_dir, model_name=model_name, trial_n=trial_n,
                                                 use_cuda=use_cuda, train_data=self.val_data, val_data=self.val_data,
                                                 test_data=self.val_data, projection_dataset=projection_dataset)
        PretrainedFeatureExtractor.freeze_layers(self.trainer.net, [])

        # Assess model
        '''self.trainer.summarize_performance(show_test=True, show_process=True, show_cm=True, assess_calibration=True)
        self.aggregate_evals()'''

        # Define CAM builder
        model = self.trainer.net.to("cuda")
        if not self.projection_dataset:
            self.cam_builder = MultiInputTorchCamBuilder(model=model, transform_fn=MapGenerator.preprocess_fn,
                                                         class_names=self.val_data.classes, input_transposed=False,
                                                         use_gpu=use_cuda)
        else:
            self.cam_builder = TorchCamBuilder(model=model, transform_fn=MapGenerator.preprocess_fn,
                                               class_names=self.val_data.classes, input_transposed=False,
                                               use_gpu=use_cuda)

    def aggregate_evals(self):
        if self.projection_dataset:
            for set_type in SetType:
                filepath = self.trainer.results_dir + set_type.value + "_classification_results.csv"
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
                self.trainer.results_dir + set_type.value + "aggregated_classification_results.csv"
                results.to_csv(filepath, index=False)

                # Display aggregated accuracies
                print()
                print("Voting performances on the " + set_type.value.upper() + " set:")
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
        if self.projection_dataset:
            segm_names = None
            data_names = data.dicom_projection_instances
        else:
            segm_names = data.dicom_instances
            data_names = []
        addon = "_multi_projection" if not self.projection_dataset else "_single_projection"
        cam_dir = self.jai_dir + set_type.value + addon + "/"

        # Request cams
        img_dim = self.trainer.net.input_dim
        data_list = []
        data_labels = []
        data_shape_list = []
        extras1 = []
        extras = []
        projection_types = []
        extra_preprocess_inputs_list = []
        original_imgs = []
        masks = []
        for i, instance in enumerate(data):
            if i > 2:  # <=8 ###########################
                continue
            item, extra = instance
            extras1.append(extra[1])
            extras.append(extra)
            projection_type = []
            resized_img = []
            instance_name = f"{extra[0]:03d}" + extra[1].lower()
            fold = self.trainer.preprocessor.segmentation_dir + set_type.value
            for j in range(len(item)):
                projection_type_j, projection_j, frac_label = item[j]
                projection_type.append(projection_type_j)
                original_imgs.append(np.stack([projection_j / np.max(projection_j)] * 3, axis=-1))
                resized_img.append(cv2.resize(projection_j, (img_dim, img_dim))[np.newaxis, :, :])
                data_shape_list.append(projection_j.shape)
                data_labels.append(int(frac_label != ""))
                if segm_names is not None:
                    data_names.append(segm_names[i] + "_" + str(j))

                # Get masks for visualization
                try:
                    projection_id = extra[2]
                except IndexError:
                    projection_id = j
                filepath = fold + "/" + instance_name + "/projection" + str(projection_id) + ".png"
                mask_j = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                mask_j = cv2.resize(mask_j, (projection_j.shape[1], projection_j.shape[0])) / 255
                masks.append(cv2.blur(mask_j, (101, 101)))
            data_list.append(resized_img)
            projection_types.append(projection_type)

        if len(projection_types) > 1:
            max_proj_num = np.max([len(projection_type) for projection_type in projection_types])
            proj_types_tmp = []
            for k, segment_data in enumerate(data_list):
                extra_proj = max_proj_num - len(segment_data)
                proj_type_tmp = [proj_type for proj_type in projection_types[k]] + [projection_types[k][-1]] * extra_proj
                proj_types_tmp.append(proj_type_tmp)
            projection_types = proj_types_tmp
        else:
            max_proj_num = None

        for i, projection_type in enumerate(projection_types):
            extra_preprocess_inputs_list.append([self.trainer, projection_type, extras[i], set_type, max_proj_num])

        data_list = [np.concatenate(segment_data, axis=0) for segment_data in data_list]
        extra_inputs_list = [extras1, projection_types, True]

        # Get CAMs
        cams_dict, predicted_probs_dict, bar_ranges_dict = self.cam_builder.get_cam(data_list, data_labels,
                                                                                    target_classes, explainer_types,
                                                                                    target_layers, softmax_final=False,
                                                                                    data_names=data_names,
                                                                                    results_dir_path=cam_dir,
                                                                                    extra_preprocess_inputs_list=
                                                                                    extra_preprocess_inputs_list,
                                                                                    extra_inputs_list=extra_inputs_list,
                                                                                    data_shape_list=data_shape_list)

        # Display overlapped
        comparison_classes = target_classes
        comparison_algorithms = explainer_types
        for comparison_class in comparison_classes:
            for comparison_algorithm in comparison_algorithms:
                for i, img in enumerate(original_imgs):
                    prob = {k: v[i][np.newaxis] for k, v in predicted_probs_dict.items()}
                    cam = {k: [v[i] * masks[i]] for k, v in cams_dict.items()}
                    bar = {k: (v[0][i], v[1][i]) for k, v in bar_ranges_dict.items()}
                    self.cam_builder.overlapped_output_display(data_list=[img], data_labels=[data_labels[i]],
                                                               predicted_probs_dict=prob, cams_dict=cam,
                                                               explainer_types=comparison_algorithm,
                                                               target_classes=comparison_class,
                                                               target_layers=target_layers, data_names=[data_names[i]],
                                                               bar_ranges_dict=bar, fig_size=(10, 6),
                                                               results_dir_path=cam_dir + data_names[i] + "/")

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
    def preprocess_fn(item, trainer, projection_type, extra, set_type, max_proj_num=None):
        projection_batch = item[np.newaxis, :, :, :]
        projection_type_batch = np.array([projection_type], dtype=Object)
        extra = [[extra_element] for extra_element in extra]
        input = trainer.preprocess_fn(projection_batch, projection_type_batch, extra, set_type, max_proj_num)

        return input[0]


class MultiInputTorchCamBuilder(TorchCamBuilder):
    def __init__(self, model, transform_fn=None, class_names=None, time_axs=1, input_transposed=False,
                 ignore_channel_dim=False, is_regression_network=False, model_output_index=None,
                 extend_search=False, use_gpu=False, padding_dim=None, seed=11):

        # Initialize attributes
        super(MultiInputTorchCamBuilder, self).__init__(model, transform_fn, class_names, time_axs, input_transposed,
                                                        ignore_channel_dim, is_regression_network, model_output_index,
                                                        extend_search, use_gpu, padding_dim, seed)

    def _create_raw_batched_cams(self, data_list, target_class, target_layer, explainer_type, softmax_final,
                                 extra_inputs_list=None, eps=1e-6):

        # Register hooks
        _ = target_layer.register_forward_hook(self._TorchCamBuilder__get_activation_forward_hook, prepend=False)
        _ = target_layer.register_forward_hook(self._TorchCamBuilder__get_gradient_forward_hook, prepend=False)

        # Data batching
        if not isinstance(data_list[0], torch.Tensor):
            data_list = [torch.Tensor(x) for x in data_list]
        if self.padding_dim is not None:
            padded_data_list = []
            for item in data_list:
                pad_size = self.padding_dim - item.shape[self.time_axs]
                if not self.time_axs:
                    zeros = torch.zeros((pad_size, item.shape[1]), dtype=item.dtype,
                                        device=item.device)
                else:
                    zeros = torch.zeros((item.shape[0], pad_size), dtype=item.dtype,
                                        device=item.device)
                padded_data_list.append(torch.cat((item, zeros), dim=self.time_axs))
            data_list = padded_data_list

        is_2d_layer = self._is_2d_layer(target_layer)
        if not self.ignore_channel_dim and (is_2d_layer and len(data_list[0].shape) == 2 or not is_2d_layer
                                            and len(data_list[0].shape) == 1):
            data_list = [x.unsqueeze(0) for x in data_list]
        data_batch = torch.stack(data_list)

        # Set device
        self.model = self.model.to(self.device)
        data_batch = data_batch.to(self.device)

        extra_inputs_list = extra_inputs_list or []
        outputs = self.model(data_batch, *extra_inputs_list)
        if isinstance(outputs, tuple):
            outputs = outputs[self.model_output_index]

        if softmax_final:
            target_probs = outputs
            if len(outputs.shape) == 2 and outputs.shape[1] > 1:
                # Approximate Softmax inversion formula logit = log(prob) + constant, as the constant is negligible
                # during derivation. Clamp probabilities before log application to avoid null maps for maximum
                # confidence.
                target_scores = torch.log(torch.clamp(outputs, min=eps, max=1 - eps))
            else:
                # Adjust results for binary networks
                target_scores = torch.logit(outputs, eps=eps)
                if len(outputs.shape) == 1:
                    target_scores = torch.stack([-target_scores, target_scores], dim=1)
                    target_probs = torch.stack([1 - target_probs, target_probs], dim=1)
                else:
                    target_scores = torch.cat([-target_scores, target_scores], dim=1)
                    target_probs = torch.cat([1 - target_probs, target_probs], dim=1)
        else:
            target_scores = outputs
            if len(outputs.shape) == 2 and outputs.shape[1] > 1:
                target_probs = torch.softmax(target_scores, dim=1)
            else:
                p = torch.sigmoid(outputs)
                if len(outputs.shape) == 1:
                    target_scores = torch.stack([-outputs, outputs], dim=1)
                    target_probs = torch.stack([1 - p, p], dim=1)
                elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                    target_scores = torch.cat([-outputs, outputs], dim=1)
                    target_probs = torch.cat([1 - p, p], dim=1)

        target_probs_out = []
        cam_list = []
        count = -1
        for i in range(len(data_list)):
            datum = data_list[i]
            for j in range(datum.shape[0]):
                count += 1
                if torch.max(datum[j]) == torch.min(datum[j]) and datum[j][0][0] == 0:
                    continue
                self.model.zero_grad()
                target_score = target_scores[i, target_class]
                target_score.backward(retain_graph=True)
                target_probs_out.append(target_probs[i, target_class].cpu().detach().numpy())

                if explainer_type == "HiResCAM":
                    cam = self._get_hirescam_map(is_2d_layer=is_2d_layer, batch_idx=count)
                else:
                    cam = self._get_gradcam_map(is_2d_layer=is_2d_layer, batch_idx=count)
                cam_list.append(cam.cpu().detach().numpy())

        return cam_list, np.array(target_probs_out)


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    # working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "projection_resnext101_optuna"
    trial_n1 = 60
    use_cuda1 = False
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None

    # Define generator
    generator1 = MapGenerator(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1, use_cuda=use_cuda1,
                              projection_dataset=projection_dataset1, selected_segments=selected_segments1,
                              selected_projection=selected_projection1)

    # Draw maps
    set_type1 = SetType.VAL
    target_classes1 = [1]
    explainer_types1 = ["Grad-CAM"]
    target_layers1 = ["feature_extractor.features.0"]
    generator1.get_cam(set_type=set_type1, target_classes=target_classes1, explainer_types=explainer_types1,
                       target_layers=target_layers1)
