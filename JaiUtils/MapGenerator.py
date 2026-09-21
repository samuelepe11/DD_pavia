# Import packages
import os

from ultralytics.engine.results import Boxes

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True)

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re
import gc
from sklearn.metrics import f1_score, matthews_corrcoef
from signal_grad_cam import TorchCamBuilder
from flatbuffers.flexbuffers import Object
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama
from Enumerators.SetType import SetType
from matplotlib.backends.backend_pdf import PdfPages
from DataUtils.XrayDataset import XrayDataset
from TrainUtils.NetworkTrainer import NetworkTrainer
from Networks.PretrainedFeatureExtractor import PretrainedFeatureExtractor


# Class
class MapGenerator:
    def __init__(self, working_dir, model_name, trial_n, use_cuda=False, projection_dataset=False,
                 selected_segments=None, selected_projection=None, is_cropped=False, yolo_cropping=False):
        # Initialize attributes
        self.working_dir = working_dir
        self.jai_dir = working_dir + XrayDataset.results_fold + XrayDataset.jai_fold
        self.data_dir = working_dir + XrayDataset.data_fold
        self.model_name = model_name
        if model_name not in os.listdir(self.jai_dir):
            os.mkdir(self.jai_dir + model_name)
        self.jai_dir += model_name + "/"

        self.trial_n = trial_n
        self.use_cuda = use_cuda
        self.projection_dataset = projection_dataset
        self.is_cropped = is_cropped
        self.yolo_cropping = yolo_cropping

        # Load data
        addon = "" if not is_cropped else "cropped_"
        if yolo_cropping:
            addon = "yolo_" + addon
        self.selected_segments = selected_segments
        self.selected_projection = selected_projection
        if not yolo_cropping:
            removable_train = "removable_instances_training.txt"
            removable_val = "removable_instances_validation.txt"
            removable_test = "removable_instances_test.txt"
        else:
            removable_train = None
            removable_val = None
            removable_test = None

        self.train_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name=addon + "xray_dataset_training",
                                                   selected_segments=selected_segments,
                                                   selected_projection=selected_projection,
                                                   removable_instances_txt=removable_train)
        self.val_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name=addon + "xray_dataset_validation",
                                                 selected_segments=selected_segments,
                                                 selected_projection=selected_projection,
                                                 removable_instances_txt=removable_val)
        self.test_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name=addon + "xray_dataset_test",
                                                  selected_segments=selected_segments,
                                                  selected_projection=selected_projection,
                                                  removable_instances_txt=removable_test)

        # Load model
        self.trainer = NetworkTrainer.load_model(working_dir=working_dir, model_name=model_name, trial_n=trial_n,
                                                 use_cuda=use_cuda, train_data=self.train_data, val_data=self.val_data,
                                                 test_data=self.test_data, projection_dataset=projection_dataset,
                                                 is_cropped=self.is_cropped)

        PretrainedFeatureExtractor.freeze_layers(self.trainer.net, [])

        # Assess model
        if not yolo_cropping:
            self.trainer.summarize_performance(show_test=True, show_process=True, show_cm=True, assess_calibration=True)
            if not is_cropped:
                self.aggregate_evals()

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

    def get_cam(self, set_type, target_classes, explainer_types, target_layers, desired_instances=None):
        # Choose data
        data, _, _ = self.trainer.select_dataset(set_type)
        if isinstance(desired_instances, dict):
            desired_instances = (desired_instances["block 1"] + desired_instances["block 2"] +
                                 desired_instances["block 3"])

        if self.projection_dataset:
            segm_names = None
            data_names = data.dicom_projection_instances if desired_instances is None else desired_instances
        else:
            segm_names = data.dicom_instances
            data_names = []
        addon = "_multi_projection" if not self.projection_dataset else "_single_projection"
        if self.yolo_cropping:
            addon += "_yolo_cropping"
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
        data_names_tmp = []
        box_coords = []
        data_names = desired_instances if desired_instances is not None else data.dicom_instances
        for data_name in data_names:
            for i, instance in enumerate(data):
                item, extra = instance
                projection_type = []
                resized_img = []
                instance_name = f"{extra[0]:03d}" + extra[1].lower()
                if instance_name != data_name:
                    continue
                extras1.append(extra[1])
                extras.append(extra)

                fold = self.trainer.preprocessor.segmentation_dir + set_type.value
                for j in range(len(item)):
                    if not self.is_cropped:
                        projection_type_j, projection_j, frac_label = item[j]
                    else:
                        projection_type_j, projection_j, frac_label, extra_info = item[j]
                    projection_type.append(projection_type_j)
                    original_imgs.append(np.stack([projection_j / np.max(projection_j)] * 3, axis=-1))
                    resized_img.append(cv2.resize(projection_j, (img_dim, img_dim))[np.newaxis, :, :])
                    data_shape_list.append(projection_j.shape)
                    data_labels.append(int(frac_label != ""))
                    if segm_names is not None:
                        data_names.append(segm_names[i] + "_" + str(j))
                    else:
                        data_names_tmp.append(extra_info.split("file_name=")[-1].split(",")[0][1:-5])

                    # Save ranges
                    tmp = extra_info.split("x_min=")[1].split(", x_max=")
                    x_min = int(tmp[0])
                    tmp = tmp[1].split(", y_min=")
                    x_max = int(tmp[0])
                    tmp = tmp[1].split(", y_max=")
                    y_min = int(tmp[0])
                    y_max = int(tmp[1].split(", fracture_present")[0])
                    box_coords.append({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})

                    # Get masks for visualization
                    if not self.is_cropped:
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
        max_batch_size = 2
        if desired_instances is not None and len(desired_instances) < max_batch_size:
            cams_dict, predicted_probs_dict, bar_ranges_dict = self.cam_builder.get_cam(data_list, data_labels,
                                                                                        target_classes, explainer_types,
                                                                                        target_layers, softmax_final=False,
                                                                                        data_names=data_names_tmp,
                                                                                        results_dir_path=cam_dir,
                                                                                        extra_preprocess_inputs_list=
                                                                                        extra_preprocess_inputs_list,
                                                                                        extra_inputs_list=extra_inputs_list,
                                                                                        data_shape_list=data_shape_list)
        else:
            cams_dict = {}
            predicted_probs_dict = {}
            bar_ranges_dict = {}
            for start_idx in range(0, len(data_list), max_batch_size):
                end_idx = min(start_idx + max_batch_size, len(data_list))
                print(f"\nGenerating CAMs for {start_idx}:{end_idx}/{len(data_list)}\n")
                data_list_batch = data_list[start_idx:end_idx]
                data_labels_batch = data_labels[start_idx:end_idx]
                data_names_batch = data_names_tmp[start_idx:end_idx]
                extra_preprocess_inputs_batch = extra_preprocess_inputs_list[start_idx:end_idx]
                extra_inputs_batch = [extra_inputs_list[0][start_idx:end_idx], extra_inputs_list[1][start_idx:end_idx], extra_inputs_list[2]]
                data_shape_batch = data_shape_list[start_idx:end_idx]
                cams_tmp, predicted_probs_tmp, bar_ranges_tmp = self.cam_builder.get_cam(data_list_batch,
                                                                                         data_labels_batch,
                                                                                         target_classes,
                                                                                         explainer_types, target_layers,
                                                                                         softmax_final=False,
                                                                                         data_names=data_names_batch,
                                                                                         results_dir_path=cam_dir,
                                                                                         extra_preprocess_inputs_list=extra_preprocess_inputs_batch,
                                                                                         extra_inputs_list=extra_inputs_batch,
                                                                                         data_shape_list=data_shape_batch)
                for k in cams_tmp.keys():
                    if k not in cams_dict.keys():
                        cams_dict.update({k: cams_tmp[k]})
                        predicted_probs_dict.update({k: predicted_probs_tmp[k]})
                        bar_ranges_dict.update({k: bar_ranges_tmp[k]})
                    else:
                        cams_dict.update({k: cams_dict[k] + cams_tmp[k]})
                        predicted_probs_dict.update({k: np.concatenate([predicted_probs_dict[k], predicted_probs_tmp[k]])})
                        bar_ranges_dict.update({k: (np.concatenate([bar_ranges_dict[k][0], bar_ranges_tmp[k][0]]), np.concatenate([bar_ranges_dict[k][1], bar_ranges_tmp[k][1]]))})
                del data_list_batch, data_labels_batch, data_names_batch, extra_preprocess_inputs_batch, extra_inputs_batch, data_shape_batch
                del cams_tmp, predicted_probs_tmp, bar_ranges_tmp
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Store raw images
        for i, data_name in enumerate(data_names_tmp):
            if "raw_image.png" not in os.listdir(cam_dir + data_name):
                plt.figure()
                plt.imshow(original_imgs[i], "gray")
                plt.xticks([], [])
                plt.yticks([], [])
                plt.savefig(cam_dir + data_name + "/raw_image.png", format="png", bbox_inches="tight",
                            pad_inches=0, dpi=500)
                plt.close()

        # Display overlapped
        comparison_classes = target_classes
        comparison_algorithms = explainer_types
        for comparison_class in comparison_classes:
            for comparison_algorithm in comparison_algorithms:
                for i, img in enumerate(original_imgs):
                    prob = {k: v[i][np.newaxis] for k, v in predicted_probs_dict.items()}
                    if not self.is_cropped:
                        cam = {k: [v[i] * masks[i]] for k, v in cams_dict.items()}
                    else:
                        cam = {k: [v[i]] for k, v in cams_dict.items()}
                    bar = {k: (v[0][i], v[1][i]) for k, v in bar_ranges_dict.items()}
                    self.cam_builder.overlapped_output_display(data_list=[img], data_labels=[data_labels[i]],
                                                               predicted_probs_dict=prob, cams_dict=cam,
                                                               explainer_types=comparison_algorithm,
                                                               target_classes=comparison_class,
                                                               target_layers=target_layers, data_names=[data_names_tmp[i]],
                                                               bar_ranges_dict=bar, fig_size=(20, 12),
                                                               results_dir_path=cam_dir + data_names_tmp[i] + "/")

                    with open(cam_dir + data_names_tmp[i] + "/" + "model_predictions.txt", "w", encoding="utf-8") as f:
                        f.write(f"x_min: {box_coords[i]['x_min']}\nx_max: {box_coords[i]['x_max']}\n"
                                f"y_min: {box_coords[i]['y_min']}\ny_max: {box_coords[i]['y_max']}\n")
                        for k, v in prob.items():
                            f.write(f"prob{k[-1]}: {v[0]:.5f}\n")

        return cams_dict, predicted_probs_dict, bar_ranges_dict

    def get_overlapped_radiography(self, cams_dict, predicted_probs_dict, bar_ranges_dict, set_type, target_classes,
                                   explainer_types, target_layers, desired_instances=None, box_thickness=3, blur=False):
        # Choose data
        data, _, _ = self.trainer.select_dataset(set_type)
        if isinstance(desired_instances, dict):
            desired_instances = (desired_instances["block 1"] + desired_instances["block 2"] +
                                 desired_instances["block 3"])

        full_dataset = XrayDataset.load_dataset(working_dir=self.working_dir,
                                                dataset_name="xray_dataset_" + set_type.value,
                                                selected_segments=self.selected_segments,
                                                selected_projection=self.selected_projection)
        data_names = full_dataset.dicom_instances if desired_instances is None else desired_instances
        addon = "_multi_projection" if not self.projection_dataset else "_single_projection"
        if self.yolo_cropping:
            addon += "_yolo_cropping"
        cam_dir = self.jai_dir + set_type.value + addon + "/"
        box_keys = ["x_min", "x_max", "y_min", "y_max"]

        # Reconstruct radiography
        for comparison_class in target_classes:
            for comparison_algorithm in explainer_types:
                for comparison_layer in target_layers:
                    cam_key = comparison_algorithm + "_" + comparison_layer + "_class" + str(comparison_class)
                    counter = 0
                    for data_name in data_names:
                        original_item, _ = full_dataset.get_data_from_name(data_name)
                        frac_labels = []
                        for j in range(len(original_item)):
                            # Get full radiography
                            _, original_projection_j, frac_label_j = original_item[j]
                            original_img = np.stack([original_projection_j / np.max(original_projection_j)] * 3,
                                                    axis=-1)
                            frac_labels.append(frac_label_j)

                            # Get cropped patch
                            flag = True
                            boxes = []
                            box_probs = []
                            for cropped_item, cropped_extra in data:
                                if not f"{cropped_extra[0]:03}" + cropped_extra[1].lower() == data_name:
                                    continue
                                else:
                                    _, _, _, cropped_info = cropped_item[0]
                                    if "proj" + str(j) not in cropped_info:
                                        continue

                                    if flag:
                                        full_width = int(cropped_info.split("width=")[-1].split(",")[0])
                                        full_height = int(cropped_info.split("height=")[-1].split(",")[0])
                                        original_img = cv2.resize(original_img, (full_width, full_height))
                                        full_cam = np.zeros_like(original_img)[:, :, 0]
                                        flag = False

                                    box = {}
                                    for key in box_keys:
                                        box.update({key: int(cropped_info.split(key + "=")[-1].split(",")[0])})
                                    if box["y_max"] >= full_cam.shape[0]:
                                        new_max = full_cam.shape[0] - 1
                                        box["y_min"] -= box["y_max"] - new_max
                                        box["y_max"] = new_max
                                    boxes.append(box)
                                    box_probs.append(float(np.asarray(predicted_probs_dict[cam_key][counter]).squeeze()))

                                    max_val = bar_ranges_dict[cam_key][1][counter][0]
                                    min_val = bar_ranges_dict[cam_key][0][counter][0]
                                    vertebra_cam = cams_dict[cam_key][counter] / 255.0 * (max_val - min_val) + min_val
                                    counter += 1
                                    for row in range(box["y_min"], (box["y_max"] + 1)):
                                        for col in range(box["x_min"], (box["x_max"] + 1)):
                                            full_h, full_w = full_cam.shape[:2]
                                            x_min = max(0, box["x_min"])
                                            y_min = max(0, box["y_min"])
                                            x_max = min(full_w, box["x_max"])
                                            y_max = min(full_h, box["y_max"])
                                            box_h = y_max - y_min
                                            box_w = x_max - x_min
                                            cam_h, cam_w = vertebra_cam.shape[:2]
                                            usable_h = min(box_h, cam_h)
                                            usable_w = min(box_w, cam_w)
                                            full_cam[y_min:y_min + usable_h, x_min:x_min + usable_w] \
                                                = vertebra_cam[:usable_h, :usable_w]

                            # Draw bounding boxes
                            original_img_tmp = original_img.copy()
                            if box_thickness != 0:
                                cmap = plt.get_cmap("tab10")
                                colors = [cmap(i)[:3] for i in range(len(boxes))]
                                for i, box in enumerate(boxes):
                                    x_min, x_max, y_min, y_max = box["x_min"], box["x_max"], box["y_min"], box["y_max"]
                                    original_img_tmp[y_min:y_min + box_thickness, x_min:x_max + 1] = colors[i]
                                    original_img_tmp[y_max - box_thickness + 1:y_max + 1, x_min:x_max + 1] = colors[i]
                                    original_img_tmp[y_min:y_max + 1, x_min:x_min + box_thickness] = colors[i]
                                    original_img_tmp[y_min:y_max + 1, x_max - box_thickness + 1:x_max + 1] = colors[i]

                            # Display overlapped
                            '''prob = {cam_key: np.mean(predicted_probs_dict[cam_key])[np.newaxis]}'''
                            full_cam, bar_ranges = self.cam_builder._CamBuilder__normalize_cams(
                                full_cam[np.newaxis, :, :], True, False)
                            full_cam = full_cam[0]
                            if blur:
                                full_cam = cv2.GaussianBlur(full_cam, (101, 101), 0)
                            '''cam = {cam_key: [full_cam]}
                            bar = {cam_key: (bar_ranges[0][0][0], bar_ranges[1][0][0])}
                            data_label = int(any([frac_label != "" for frac_label in frac_labels]))'''
                            data_name_tmp = data_name + "_proj" + str(j)
                            if data_name_tmp not in os.listdir(cam_dir):
                                os.mkdir(cam_dir + data_name_tmp)

                            # GT box visualization
                            if "predicted_boxes.png" not in os.listdir(cam_dir + data_name_tmp + "/"):
                                box_img = np.zeros_like(original_img)
                                candidates = []
                                for box, prob in zip(boxes, box_probs):
                                    if prob >= 0.01:
                                        candidates.append({"box": box, "prob": prob})
                                filtered_candidates = []
                                for i, cand_i in enumerate(candidates):
                                    box_i = cand_i["box"]
                                    prob_i = cand_i["prob"]
                                    keep = True
                                    for j, cand_j in enumerate(candidates):
                                        if i == j:
                                            continue
                                        box_j = cand_j["box"]
                                        prob_j = cand_j["prob"]
                                        inside = (box_i["x_min"] >= box_j["x_min"] and
                                                  box_i["x_max"] <= box_j["x_max"] and
                                                  box_i["y_min"] >= box_j["y_min"] and
                                                  box_i["y_max"] <= box_j["y_max"])
                                        if inside and prob_j > prob_i:
                                            keep = False
                                            break
                                    if keep:
                                        filtered_candidates.append(cand_i)
                                for cand in filtered_candidates:
                                    box = cand["box"]
                                    prob = cand["prob"]
                                    x_min, x_max, y_min, y_max = box["x_min"], box["x_max"], box["y_min"], box["y_max"]
                                    color = (0, 0, 255) if prob >= 0.5 else (0, 165, 255)
                                    cv2.rectangle(box_img, (x_min, y_min), (x_max, y_max), color, thickness=8)
                                    label = f"{prob:.3f}"
                                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                                                                                 5)
                                    if x_max + 12 + text_w < box_img.shape[1]:
                                        text_x = x_max + 12
                                    else:
                                        text_x = max(5, x_min - text_w - 12)
                                    text_y = y_min + ((y_max - y_min) + text_h) // 2
                                    cv2.putText(box_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                                2.4, color, 5, cv2.LINE_AA)
                                cv2.imwrite(cam_dir + data_name_tmp + "/predicted_boxes.png", box_img)

                            # Raw image
                            if "raw_image.png" not in os.listdir(cam_dir + data_name_tmp + "/"):
                                plt.figure()
                                plt.imshow(original_img_tmp)
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(cam_dir + data_name_tmp + "/raw_image.png", format="png",
                                            bbox_inches="tight", pad_inches=0, dpi=500)
                                plt.close()

                            if "xray_dataset_" + set_type.value + "_masks" in os.listdir(self.data_dir):
                                gt_mask = cv2.imread(self.data_dir + "xray_dataset_" + set_type.value +
                                                     "_masks/gt_masks/" + data_name + "/projection" + str(j) + ".jpg",
                                                     cv2.IMREAD_GRAYSCALE)
                                gt_mask = cv2.resize(gt_mask, (original_img_tmp.shape[1], original_img_tmp.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)
                                gt_mask = (gt_mask >= 128).astype(np.uint8) * 255
                                if "gt_image.png" not in os.listdir(cam_dir + data_name_tmp + "/"):
                                    plt.figure()
                                    plt.imshow(original_img_tmp)
                                    map = plt.imshow(gt_mask, cmap="inferno", norm=None)
                                    map.set_alpha(0.3)
                                    plt.xticks([], [])
                                    plt.yticks([], [])
                                    plt.savefig(cam_dir + data_name_tmp + "/gt_image.png", format="png",
                                                bbox_inches="tight", pad_inches=0, dpi=500)
                                    plt.close()
                            else:
                                gt_mask = None

                            # CAM alone
                            norm = self.cam_builder._CamBuilder__get_norm(full_cam)
                            cam_norm = norm(full_cam) if norm is not None else full_cam
                            filename = (comparison_algorithm + "_" + re.sub(r"\W", "_", comparison_layer) +
                                        "_class" + str(comparison_class) + ".png")
                            cv2.imwrite(cam_dir + data_name_tmp + "/" + filename, cam_norm)

                            # CAM overlapped
                            plt.figure()
                            plt.imshow(original_img_tmp)
                            map = plt.imshow(full_cam, cmap="inferno", norm=norm)
                            map.set_alpha(0.3)
                            plt.xticks([], [])
                            plt.yticks([], [])
                            filename = (cam_dir + data_name_tmp + "/" + "results_" + filename)
                            plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0, dpi=500)
                            plt.close()
                            '''self.cam_builder.overlapped_output_display(data_list=[original_img_tmp], data_labels=[data_label],
                                                                       predicted_probs_dict=prob, cams_dict=cam,
                                                                       explainer_types=comparison_algorithm,
                                                                       target_classes=comparison_class,
                                                                       target_layers=target_layers,
                                                                       data_names=[data_name_tmp],
                                                                       bar_ranges_dict=bar, fig_size=(20, 12),
                                                                       results_dir_path=cam_dir + data_name_tmp + "/")'''

    def compute_validation_metrics(self, set_type, target_classes, explainer_types, target_layers, desired_instances=None):
        # Choose data
        data, _, _ = self.trainer.select_dataset(set_type)
        full_dataset = XrayDataset.load_dataset(working_dir=self.working_dir,
                                                dataset_name="xray_dataset_" + set_type.value,
                                                selected_segments=self.selected_segments,
                                                selected_projection=self.selected_projection)
        data_names = full_dataset.dicom_instances if desired_instances is None else desired_instances
        addon = "_multi_projection" if not self.projection_dataset else "_single_projection"
        if self.yolo_cropping:
            addon += "_yolo_cropping"
        cam_dir = self.jai_dir + set_type.value + addon + "/"

        for comparison_class in target_classes:
            for comparison_algorithm in explainer_types:
                for comparison_layer in target_layers:
                    iou_list = []
                    iogt_list = []
                    for data_name in data_names:
                        original_item, _ = full_dataset.get_data_from_name(data_name)
                        for j in range(len(original_item)):
                            data_name_tmp = data_name + "_proj" + str(j)
                            # Get CAM
                            filename = (comparison_algorithm + "_" + re.sub(r"\W", "_", comparison_layer) +
                                        "_class" + str(comparison_class) + ".png")
                            cam = cv2.imread(cam_dir + data_name_tmp + "/" + filename, cv2.IMREAD_GRAYSCALE)
                            if cam is None:
                                continue
                            cam = cam / np.max(cam)

                            # GT mask
                            gt_mask = cv2.imread(self.data_dir + "xray_dataset_" + set_type.value +
                                                 "_masks/gt_masks/" + data_name + "/projection" + str(j) + ".jpg",
                                                 cv2.IMREAD_GRAYSCALE)
                            gt_mask = cv2.resize(gt_mask, (cam.shape[1], cam.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                            gt_mask = (gt_mask >= 128).astype(np.uint8) * 255

                            # Compute validation metrics
                            if gt_mask is not None and cam is not None:
                                cam_black = np.all(np.abs(cam) < 1e-8)
                                gt_black = np.all(gt_mask == 0)
                                if cam_black and gt_black:
                                    iou = 1.0
                                    iogt = 1.0
                                elif cam_black or gt_black:
                                    iou = 0.0
                                    iogt = 0.0
                                else:
                                    m = np.mean(cam)
                                    s = np.std(cam)
                                    full_cam_bin = cam >= m + s
                                    intersection = np.logical_and(gt_mask, full_cam_bin).sum()
                                    union = np.logical_or(gt_mask, full_cam_bin).sum()
                                    gt_area = gt_mask.sum() / 255
                                    iou = intersection / union if union > 0 else 0.0
                                    iogt = intersection / gt_area if gt_area > 0 else 0.0
                                iou_list.append(iou)
                                iogt_list.append(iogt)

                    # Store final IoU and IoGT
                    with open(cam_dir + "/" + "validation.txt", "a", encoding="utf-8") as f:
                        f.write(f"Class = {comparison_class}, algorithm = {comparison_algorithm}, layer = {comparison_layer}\n")
                        f.write(f"IoU = {np.mean(iou_list)}\n")
                        f.write(f"IoGT = {np.mean(iogt_list)}\n\n")

    def generate_gt_prediction_pdf(self, set_type, desired_instances=None, output_path=None, threshold=0.5,
                                   box_thickness=8, max_batch_size=32):
        data, _, _ = self.trainer.select_dataset(set_type)
        gt_data = XrayDataset.load_dataset(working_dir=self.working_dir,
                                           dataset_name="cropped_xray_dataset_" + set_type.value,
                                           selected_segments=self.selected_segments,
                                           selected_projection=self.selected_projection)
        full_dataset = XrayDataset.load_dataset(working_dir=self.working_dir,
                                                dataset_name="xray_dataset_" + set_type.value,
                                                selected_segments=self.selected_segments,
                                                selected_projection=self.selected_projection)

        data_names = full_dataset.dicom_instances if desired_instances is None else desired_instances
        data_names_set = set(data_names)

        if output_path is None:
            output_path = self.jai_dir + set_type.value + "_gt_vs_model.pdf"

        img_dim = self.trainer.net.input_dim
        box_keys = ["x_min", "x_max", "y_min", "y_max"]
        records = []

        for cropped_item, cropped_extra in data:
            instance_name = f"{cropped_extra[0]:03d}" + cropped_extra[1].lower()
            if instance_name not in data_names_set:
                continue

            for projection_type_j, projection_j, _, extra_info in cropped_item:
                match = re.search(r"proj(\d+)", extra_info)
                if match is None:
                    raise ValueError(f"Projection number not found in YOLO crop: {extra_info}")

                projection_id = int(match.group(1))
                box = {key: int(extra_info.split(key + "=")[-1].split(",")[0]) for key in box_keys}
                full_width = int(extra_info.split("width=")[-1].split(",")[0])
                full_height = int(extra_info.split("height=")[-1].split(",")[0])

                records.append(
                    {"instance": instance_name, "projection_id": projection_id, "projection_type": projection_type_j,
                     "projection": projection_j, "extra": cropped_extra, "box": box, "full_width": full_width,
                     "full_height": full_height})

        gt_records = []

        for cropped_item, cropped_extra in gt_data:
            instance_name = f"{cropped_extra[0]:03d}" + cropped_extra[1].lower()
            if instance_name not in data_names_set:
                continue

            for _, _, frac_label, extra_info in cropped_item:
                match = re.search(r"proj(\d+)", extra_info)
                if match is None:
                    raise ValueError(f"Projection number not found in GT crop: {extra_info}")

                projection_id = int(match.group(1))
                box = {key: int(extra_info.split(key + "=")[-1].split(",")[0]) for key in box_keys}
                gt_records.append(
                    {"instance": instance_name, "projection_id": projection_id, "gt": int(frac_label != ""),
                     "box": box})

        net = self.trainer.net
        device = next(net.parameters()).device
        net.device = device

        was_training = net.training
        net.eval()

        with torch.inference_mode():
            for start_idx in range(0, len(records), max_batch_size):
                end_idx = min(start_idx + max_batch_size, len(records))
                batch_records = records[start_idx:end_idx]

                batch_inputs = []
                batch_segments = []
                batch_projection_types = []

                for record in batch_records:
                    resized_img = cv2.resize(record["projection"], (img_dim, img_dim))[np.newaxis, :, :]
                    input_tensor = self.preprocess_fn(resized_img, self.trainer, [record["projection_type"]],
                                                      record["extra"], set_type, None)

                    batch_inputs.append(input_tensor)
                    batch_segments.append(record["extra"][1])
                    batch_projection_types.append([record["projection_type"]])

                batch_inputs = torch.stack(batch_inputs).to(device)
                outputs = net(batch_inputs, batch_segments, batch_projection_types, True)
                probs = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)

                for record, prob in zip(batch_records, probs):
                    record["prob"] = float(prob)

        if was_training:
            net.train()

        for record in records:
            del record["projection"]
            del record["extra"]
            del record["projection_type"]

        records_by_projection = {}
        for record in records:
            key = (record["instance"], record["projection_id"])
            if key not in records_by_projection:
                records_by_projection[key] = []
            records_by_projection[key].append(record)

        gt_records_by_projection = {}
        for record in gt_records:
            key = (record["instance"], record["projection_id"])
            if key not in gt_records_by_projection:
                gt_records_by_projection[key] = []
            gt_records_by_projection[key].append(record)

        with PdfPages(output_path) as pdf:
            for data_name in data_names:
                original_item, _ = full_dataset.get_data_from_name(data_name)
                n_projections = len(original_item)

                fig, axes = plt.subplots(n_projections, 2, figsize=(14, 6 * n_projections), squeeze=False)
                fig.suptitle("Instance " + data_name, fontsize=18)

                for j in range(n_projections):
                    _, original_projection, _ = original_item[j]
                    projection_records = records_by_projection.get((data_name, j), [])
                    projection_gt_records = gt_records_by_projection.get((data_name, j), [])

                    max_val = np.max(original_projection)
                    original_projection = original_projection / max_val if max_val > 0 else original_projection
                    original_img = np.stack([original_projection] * 3, axis=-1)

                    if len(projection_records) > 0:
                        full_width = projection_records[0]["full_width"]
                        full_height = projection_records[0]["full_height"]
                        original_img = cv2.resize(original_img, (full_width, full_height))

                    original_img = (np.clip(original_img, 0, 1) * 255).astype(np.uint8)
                    gt_img = original_img.copy()
                    pred_img = original_img.copy()

                    color = (255, 0, 0)

                    for record in projection_gt_records:
                        if record["gt"] != 1:
                            continue

                        box = record["box"]
                        x_min = max(0, min(gt_img.shape[1] - 1, box["x_min"]))
                        x_max = max(0, min(gt_img.shape[1] - 1, box["x_max"]))
                        y_min = max(0, min(gt_img.shape[0] - 1, box["y_min"]))
                        y_max = max(0, min(gt_img.shape[0] - 1, box["y_max"]))
                        cv2.rectangle(gt_img, (x_min, y_min), (x_max, y_max), color, thickness=box_thickness)

                    for record in projection_records:
                        if record["prob"] >= threshold:
                            pred_color = (255, 0, 0)
                        elif 0.01 < record["prob"] < threshold:
                            pred_color = (255, 165, 0)
                        else:
                            continue

                        box = record["box"]
                        x_min = max(0, min(pred_img.shape[1] - 1, box["x_min"]))
                        x_max = max(0, min(pred_img.shape[1] - 1, box["x_max"]))
                        y_min = max(0, min(pred_img.shape[0] - 1, box["y_min"]))
                        y_max = max(0, min(pred_img.shape[0] - 1, box["y_max"]))

                        cv2.rectangle(pred_img, (x_min, y_min), (x_max, y_max), pred_color, thickness=box_thickness)

                        label = f"{record['prob']:.3f}"
                        font_scale = 1.2
                        text_thickness = 3
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              text_thickness)

                        text_x = x_max + 12 if x_max + 12 + text_w < pred_img.shape[1] else max(5, x_min - text_w - 12)
                        text_y = y_min + ((y_max - y_min) + text_h) // 2
                        text_y = max(text_h + 5, min(pred_img.shape[0] - 5, text_y))

                        cv2.putText(pred_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pred_color,
                                    text_thickness, cv2.LINE_AA)

                    axes[j, 0].imshow(gt_img)
                    axes[j, 0].set_title("Ground truth", fontsize=14)
                    axes[j, 0].axis("off")

                    axes[j, 1].imshow(pred_img)
                    axes[j, 1].set_title("Prediction", fontsize=14)
                    axes[j, 1].axis("off")

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print("PDF stored in:", output_path)
        return output_path

    def get_textual_explainer(self):
        role = ("Sei un sistema di supporto alla decisione medica esperto nella valutazione di radiografie della colonna"
                " vertebrale. Il tuo compito è trasformare le regioni selezionate da un classificatore in una breve "
                "argomentazione testuale rivolta a un medico, descrivendo esclusivamente caratteristiche anatomiche o "
                "morfologiche realmente osservabili in un frammento di immagine radiografica originale e valutandole "
                "come evidenza a favore della classe proposta. L'argomentazione deve essere persuasiva attraverso la "
                "precisione e la concretezza delle osservazioni, senza amplificarne il significato e senza introdurre "
                "reperti non visibili.")
        model = Ollama(id="qwen3-vl:4b", host="http://localhost:11434", options={"temperature": 0.1})
        instructions = [
        "Rispondi esclusivamente in italiano.",
        "Produci un unico paragrafo di massimo cinque frasi.",
        "Non utilizzare elenchi, titoli, numerazioni o formattazione Markdown.",
        "Restituisci soltanto la risposta finale e non mostrare il ragionamento seguito, i passaggi intermedi o un riassunto delle istruzioni ricevute.",
        "Nella risposta finale non utilizzare mai le parole heatmap, mappa, colormap, Grad-CAM, giallo, arancione, grigio, nero, colore, intelligenza artificiale, modello o classificatore.",
        "Le tre immagini devono essere interpretate sempre secondo il loro ordine e secondo la funzione assegnata a ciascuna di esse.",
        "Usa l'IMMAGINE 2 esclusivamente per individuare posizione, estensione e importanza relativa delle regioni maggiormente selezionate.",
        "Usa l'IMMAGINE 3 esclusivamente per associare le regioni individuate nell'IMMAGINE 2 alle strutture anatomiche corrispondenti nell'immagine radiografica.",
        "Usa l'IMMAGINE 1 come riferimento finale per verificare che ogni caratteristica anatomica o morfologica descritta nella risposta sia realmente visibile.",
        "Qualsiasi caratteristica anatomica o morfologica riportata nella risposta deve essere direttamente verificabile nell'IMMAGINE 1.",
        "Non interpretare la sola intensità o estensione della regione selezionata come evidenza di frattura o assenza di frattura: essa indica esclusivamente quali regioni sono state considerate maggiormente rilevanti per la classe proposta.",
        "Considera prioritariamente le regioni con maggiore intensità nell'IMMAGINE 2 e la loro corrispondenza anatomica nell'IMMAGINE 3.",
        "Puoi utilizzare internamente le informazioni visive delle IMMAGINI 2 e 3 per localizzare le regioni selezionate, ma nella risposta finale devi descrivere esclusivamente la loro posizione anatomica e le caratteristiche radiografiche effettivamente visibili.",
        "Ignora completamente titolo, nome del file, etichette, percentuali, barra laterale, valori numerici e qualsiasi testo visibile nelle immagini, soprattuto in IMMAGINI 2 e 3 dove un titolo e una colorbar sono sempre presenti.",
        "Non utilizzare eventuale testo presente nelle immagini per determinare la classe, la diagnosi, la confidenza o il significato delle regioni selezionate.",
        "Non identificare il livello vertebrale specifico o il tratto spinale.",
        "Descrivi inizialmente la posizione relativa della regione selezionata utilizzando, quando appropriato, termini come superiore, inferiore, centrale, periferica, anteriore o posteriore.",
        "Identifica una struttura anatomica specifica soltanto quando essa è chiaramente riconoscibile nell'IMMAGINE 1.",
        "Non chiamare una struttura piatto vertebrale, corticale, parete vertebrale, peduncolo o altra struttura specifica quando la sua identificazione è incerta.",
        "Non completare mentalmente strutture parzialmente rappresentate e non inferire la loro morfologia nelle porzioni non visibili.",
        "Descrivi come regolare, conservata, continua o integra una struttura soltanto quando la caratteristica pertinente è chiaramente visualizzabile nell'IMMAGINE 1.",
        "Non dichiarare la presenza di depressione, deformazione, cedimento, discontinuità, irregolarità corticale o altri segni di frattura se tali caratteristiche non sono chiaramente osservabili nell'IMMAGINE 1.",
        "Per la classe 'frattura assente', descrivi una struttura come conservata, regolare, continua o integra soltanto se la caratteristica pertinente è chiaramente visualizzabile nell'IMMAGINE 1.",
        "La classe proposta è un'ipotesi da sostenere attraverso l'evidenza visiva e non costituisce essa stessa una prova.",
        "Se il collegamento tra la regione selezionata e un reperto morfologico favorevole alla classe proposta è debole, indiretto, parziale o non completamente valutabile, non inventare ulteriori caratteristiche e utilizza la conclusione 'supporta parzialmente'.",
        "Utilizza la conclusione 'supporta' soltanto quando la regione selezionata corrisponde chiaramente a una caratteristica morfologica visibile e pertinente alla classe proposta.",
        "Valuta l'associazione tra le caratteristiche osservate e la classe proposta scegliendo esclusivamente una delle due conclusioni: 'supporta' oppure 'supporta parzialmente'.",
        "Non discutere la classe alternativa.",
        "Non formulare una diagnosi autonoma.",
        "Non aggiungere informazioni cliniche, anatomiche o diagnostiche non direttamente osservabili.",
        "Non menzionare la confidenza della previsione, la probabilità della classe, il ground truth o la correttezza complessiva della classificazione."]
        self.text_explainer = Agent(role=role, model=model, tools=[], markdown=False, instructions=instructions)
        self.base_prompt = ("Analizza una singola proiezione radiografica contenente approssimativamente una sola "
                            "vertebra, una sua porzione oppure l'intero tratto sacro-coccigeo. La proiezione può essere"
                            " antero-posteriore o latero-laterale. Ti vengono fornite esattamente tre immagini e il "
                            "loro ordine è sempre significativo. IMMAGINE 1: radiografia originale senza "
                            "sovrapposizioni; utilizzala per riconoscere le strutture anatomiche e verificare le "
                            "caratteristiche morfologiche effettivamente visibili. IMMAGINE 2: rappresentazione isolata"
                            " delle regioni selezionate dal classificatore come evidenza per la classe proposta; utilizzala "
                            "esclusivamente per determinare dove si concentra maggiormente l'informazione rilevante e "
                            "quanto sono estese le regioni selezionate. IMMAGINE 3: sovrapposizione dell'IMMAGINE 2 "
                            "alla radiografia originale (IMMAGINE 1); utilizzala esclusivamente per stabilire a quali "
                            "strutture anatomiche visibili nell'IMMAGINE 1 corrispondono le regioni selezionate nell'"
                            "IMMAGINE 2. Segui questa sequenza concettuale: individua la regione mediante l'IMMAGINE 2,"
                            " determina la sua corrispondenza anatomica mediante l'IMMAGINE 3 e verifica ogni "
                            "osservazione morfologica direttamente nell'IMMAGINE 1. L'IMMAGINE 1 rappresenta sempre il "
                            "riferimento finale per stabilire ciò che può essere affermato nella risposta. Ricorda che "
                            "il medico che stai supportanto avrà accesso esclusivamente all'IMMAGINE 1.")
        self.ita_classes = ["frattura assente", "frattura presente"]

    def textually_explain(self, set_type, desired_instances):
        addon = "_multi_projection" if not self.projection_dataset else "_single_projection"
        if self.yolo_cropping:
            addon += "_yolo_cropping"
        cam_dir = self.jai_dir + set_type.value + addon + "/"

        for instance in desired_instances:
            for folder in os.listdir(cam_dir):
                if instance in folder and len(folder.split("_")) == 3:
                    tmp_dir = cam_dir + folder + "/"
                    explanation_path = tmp_dir + "text_explanations.txt"
                    with open(explanation_path, "w", encoding="utf-8") as explanation_file:
                        for overlapped_input in os.listdir(tmp_dir):
                            if overlapped_input.startswith("Grad-CAM") or overlapped_input.startswith("HiResCAM") or overlapped_input.endswith("txt") or overlapped_input == "raw_image.png" or overlapped_input == "gt_image.png":
                                continue

                            # Complement prompt with class information
                            predicted_class = int(overlapped_input.split("_class")[-1].split(".")[0])
                            predicted_class = self.ita_classes[predicted_class]
                            prompt = (self.base_prompt + (f" La classe proposta da valutare è '{predicted_class}'. "
                                                          "Individua innanzitutto mediante l'IMMAGINE 2 le regioni "
                                                          "maggiormente selezionate dell'immagine, utilizza l'IMMAGINE "
                                                          "3 per stabilire a quali parti della vertebra o strutture "
                                                          "anatomiche esse corrispondono e verifica infine nell'"
                                                          "IMMAGINE 1 quali caratteristiche morfologiche siano "
                                                          "effettivamente presenti in quelle stesse regioni. Costruisci"
                                                          " un'argomentazione breve e persuasiva a favore della classe "
                                                          f"'{predicted_class}' utilizzando esclusivamente "
                                                          "caratteristiche realmente osservabili nell'IMMAGINE 1 e "
                                                          "spazialmente corrispondenti alle regioni selezionate nelle "
                                                          "IMMAGINI 2 e 3. La persuasività deve derivare dalla "
                                                          "precisione delle osservazioni e non dall'introduzione di "
                                                          "reperti non visibili. Se il rapporto tra quanto selezionato "
                                                          f"e la classe '{predicted_class}' è chiaro e sostenuto da una"
                                                          " caratteristica morfologica direttamente osservabile, "
                                                          "concludi che l'evidenza 'supporta' la classe proposta; se "
                                                          "tale rapporto è soltanto parziale, indiretto, debole o non "
                                                          "completamente valutabile, descrivi soltanto ciò che è "
                                                          "effettivamente visibile e concludi che l'evidenza 'supporta "
                                                          "parzialmente' la classe proposta. Non discutere la classe "
                                                          "alternativa, non utilizzare la classe proposta per dedurre "
                                                          "automaticamente ciò che dovrebbe essere presente nell'"
                                                          "immagine e non formulare una diagnosi indipendente. La "
                                                          "classe proposta potrebbe non corrispondere alla diagnosi "
                                                          "reale."))

                            # Process input
                            img_input = [Image(filepath=tmp_dir + "raw_image.png", detail="high"),
                                         Image(filepath=tmp_dir + overlapped_input[8:], detail="high"),
                                         Image(filepath=tmp_dir + overlapped_input, detail="high")]

                            # Explain
                            print(f"Textually explaining {folder}/{overlapped_input}...")
                            response = self.text_explainer.run(prompt, images=img_input)
                            explanation_file.write(overlapped_input + "\n")
                            explanation_file.write(response.content.strip().replace("\n\n", " ").replace("\n", " ") + "\n\n")

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
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "cropped_projection_resnext50_simpler_transpose_equalize"
    trial_n1 = 2
    use_cuda1 = True
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None
    is_cropped1 = True

    # Define generator
    yolo_cropping1 = True
    generator1 = MapGenerator(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1, use_cuda=use_cuda1,
                              projection_dataset=projection_dataset1, selected_segments=selected_segments1,
                              selected_projection=selected_projection1, is_cropped=is_cropped1,
                              yolo_cropping=yolo_cropping1)

    # Draw maps
    set_type1 = SetType.VAL
    target_classes1 = [1]
    explainer_types1 = ["Grad-CAM", "HiResCAM"]
    target_layers1 = ["feature_extractor.features.7.1.conv3"]#feature_extractor.features.6.5.conv3"]#"feature_extractor.features.7.2.conv3"]#, "feature_extractor.features.7.1.conv3",
    desired_instances1 = None # ["032d"]#, "032l", "446l"]
    '''desired_instances1 = {"block 1": ["474l", "378d", "405l", "281c", "413l", "297l", "170l", "093l"],
                          "block 2": ["433d", "308c", "312l", "152l", "150l", "330l", "459d", "413d"],
                          "block 3": ["338l", "229c", "386l", "123l", "226l", "354d", "174l", "113l"]}'''
    cams_dict1, predicted_probs_dict1, bar_ranges_dict1 = generator1.get_cam(set_type=set_type1,
                                                                             target_classes=target_classes1,
                                                                             explainer_types=explainer_types1,
                                                                             target_layers=target_layers1,
                                                                             desired_instances=desired_instances1)

    # Overlap radiography
    generator1.get_overlapped_radiography(cams_dict1, predicted_probs_dict1, bar_ranges_dict1, set_type=set_type1,
                                          target_classes=target_classes1, explainer_types=explainer_types1,
                                          target_layers=target_layers1, desired_instances=desired_instances1,
                                          box_thickness=0, blur=True)

    # Compute validation metrics
    generator1.compute_validation_metrics(set_type=set_type1, target_classes=target_classes1,
                                          explainer_types=explainer_types1, target_layers=target_layers1,
                                          desired_instances=desired_instances1)

    # Generate PDF for data selection
    # generator1.generate_gt_prediction_pdf(set_type=set_type1, desired_instances=desired_instances1)

    # Textual explainer
    # generator1.get_textual_explainer()
    # generator1.textually_explain(set_type1, desired_instances1)