# Import packages
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True)

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score, matthews_corrcoef
from signal_grad_cam import TorchCamBuilder
from flatbuffers.flexbuffers import Object
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama
from Enumerators.SetType import SetType
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
        self.train_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name=addon + "xray_dataset_training",
                                                   selected_segments=selected_segments,
                                                   selected_projection=selected_projection)
        self.val_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name=addon + "xray_dataset_validation",
                                                 selected_segments=selected_segments,
                                                 selected_projection=selected_projection)
        self.test_data = XrayDataset.load_dataset(working_dir=working_dir, dataset_name=addon + "xray_dataset_test",
                                                  selected_segments=selected_segments,
                                                  selected_projection=selected_projection)

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
        for i, instance in enumerate(data):
            item, extra = instance
            projection_type = []
            resized_img = []
            instance_name = f"{extra[0]:03d}" + extra[1].lower()
            if instance_name not in desired_instances:
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
        cams_dict, predicted_probs_dict, bar_ranges_dict = self.cam_builder.get_cam(data_list, data_labels,
                                                                                    target_classes, explainer_types,
                                                                                    target_layers, softmax_final=False,
                                                                                    data_names=data_names_tmp,
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
                        for k, v in prob.items():
                            f.write(f"{k[-1]}: {v[0]:.5f}\n")

        return cams_dict, predicted_probs_dict, bar_ranges_dict

    def get_overlapped_radiography(self, cams_dict, predicted_probs_dict, bar_ranges_dict, set_type, target_classes,
                                   explainer_types, target_layers, desired_instances=None, box_thickness=3, blur=False):
        # Choose data
        data, _, _ = self.trainer.select_dataset(set_type)
        full_dataset = XrayDataset.load_dataset(working_dir=self.working_dir,
                                                dataset_name="xray_dataset_" + set_type.value,
                                                selected_segments=self.selected_segments,
                                                selected_projection=self.selected_projection)
        data_names = full_dataset.dicom_projection_instances if desired_instances is None else desired_instances
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
                    for data_name in data_names:
                        original_item, _ = full_dataset.get_data_from_name(data_name)
                        frac_labels = []
                        for j in range(len(original_item)):
                            # Get full radiography
                            _, original_projection_j, frac_label_j = original_item[j]
                            original_img = np.stack([original_projection_j / np.max(original_projection_j)] * 3, axis=-1)
                            frac_labels.append(frac_label_j)

                            # Get cropped patch
                            flag = True
                            boxes = []
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

                                    max_val = bar_ranges_dict[cam_key][1][0][0]
                                    min_val = bar_ranges_dict[cam_key][0][0][0]
                                    vertebra_cam = cams_dict[cam_key][cropped_extra[2]] / 255.0 * (max_val - min_val) + min_val
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
                            prob = {cam_key: np.mean(predicted_probs_dict[cam_key])[np.newaxis]}
                            full_cam, bar_ranges = self.cam_builder._CamBuilder__normalize_cams(full_cam[np.newaxis, :, :], True, False)
                            full_cam = full_cam[0]
                            if blur:
                                full_cam = cv2.GaussianBlur(full_cam, (101, 101), 0)
                            cam = {cam_key: [full_cam]}
                            bar = {cam_key: (bar_ranges[0][0][0], bar_ranges[1][0][0])}
                            data_label = int(any([frac_label != "" for frac_label in frac_labels]))
                            data_name_tmp = data_name + "_proj" + str(j)
                            if data_name_tmp not in os.listdir(cam_dir):
                                os.mkdir(cam_dir + data_name_tmp)

                            if "raw_image.png" not in os.listdir(cam_dir + data_name_tmp + "/"):
                                plt.figure()
                                plt.imshow(original_img_tmp)
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(cam_dir + data_name_tmp + "/raw_image.png", format="png", bbox_inches="tight", pad_inches=0, dpi=500)

                            plt.figure()
                            plt.imshow(original_img_tmp)
                            norm = self.cam_builder._CamBuilder__get_norm(full_cam)
                            map = plt.imshow(full_cam, cmap="inferno", norm=norm)
                            map.set_alpha(0.3)
                            plt.xticks([], [])
                            plt.yticks([], [])
                            filename = (cam_dir + data_name_tmp + "/" + "results_" + comparison_algorithm + "_" +
                                        re.sub(r"\W", "_", comparison_layer) + "_class" + str(comparison_class) + ".png")
                            plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0, dpi=500)
                            '''self.cam_builder.overlapped_output_display(data_list=[original_img_tmp], data_labels=[data_label],
                                                                       predicted_probs_dict=prob, cams_dict=cam,
                                                                       explainer_types=comparison_algorithm,
                                                                       target_classes=comparison_class,
                                                                       target_layers=target_layers,
                                                                       data_names=[data_name_tmp],
                                                                       bar_ranges_dict=bar, fig_size=(20, 12),
                                                                       results_dir_path=cam_dir + data_name_tmp + "/")'''

    def get_textual_explainer(self):
        role = ("Sei un radiologo muscoloscheletrico incaricato di analizzare una porzione di radiografia vertebrale e "
                "di ricavare quali sono gli elementi a favore di una classe proposta (frattura assente o frattura "
                "presente) tra quelli evidenziati da un modello AI tramite una mappa di calore con colorazione inferno "
                "dove i colori scuri (nero o grigio) identificano le parti di minore interesse mentre i colori brillanti"
                " (giallo o arancio) identificano le parti di maggiore importanza secondo il modello.")
        model = Ollama(id="qwen3-vl:4b", host="http://localhost:11434",
                       options={"temperature": 0.1})
        instructions = [
            "Rispondi esclusivamente in italiano.",
            "Produci un unico paragrafo di massimo tre frasi.",
            "Non utilizzare elenchi, titoli, numerazioni o formattazione Markdown.",
            "Restituisci soltanto la risposta finale e non mostrare il ragionamento seguito né riassumi i prompt forniti.",
            "Nella risposta non utilizzare mai le parole heatmap, mappa, colormap, giallo, arancione, colore, intelligenza artificiale, modello o classificatore.",
            "Considera esclusivamente le regioni maggiormente selezionate dalla mappa di calore (in giallo/arancione brillante) nella parte radiografica dell'immagine.",
            "Ignora completamente titolo, nome del file, etichette, percentuali, barra laterale, valori numerici e qualsiasi testo visibile nell'immagine.",
            "Non identificare il livello vertebrale o il tratto spinale.",
            "Descrivi inizialmente la posizione relativa della regione selezionata dalla mappa di calore usando termini come superiore, inferiore, centrale, periferica, anteriore o posteriore.",
            "Non chiamare una struttura piatto vertebrale, corticale, parete vertebrale o peduncolo quando la sua identificazione è incerta.",
            "Non usare la semplice assenza di un'anomalia visibile come prova della classe frattura assente.",
            "Valuta l'associazione tra un fenoeno e la classe proposta scegliendo esclusivamente una delle seguenti conclusioni: supporta, supporta parzialmente, è insufficiente oppure contraddice.",
            "Devi cercare necessariamente elementi favorevoli alla classe proposta.",
            "La classe proposta è soltanto l'ipotesi da valutare per cui tu devi cercare evidenza visiva.",
            "Non aggiungere informazioni cliniche, anatomiche o diagnostiche non direttamente osservabili.",
            "Non fornire una diagnosi definitiva."]
        self.text_explainer = Agent(role=role, model=model, tools=[], markdown=False, instructions=instructions)
        self.base_prompt = ("Analizza una singola proiezione radiografica contenente approssimativame una sola vertebra, "
                            "o una sua porzione o l'intero tratto sacro-coccigeo. La proiezione può essere antero-posteriore "
                            "o latero-laterale. Considera esclusivamente le regioni maggiormente selezionate nella parte "
                            "radiografica e ignora completamente ogni testo, titolo, barra, o valore numerico nell'immagine. "
                            "Prima descrivi soltanto ciò che è direttamente osservabile; successivamente valuta il rapporto "
                            "tra tale osservazione e la classe proposta")
        self.ita_classes = ["frattura assente", "frattura presente"]

    def textually_explain(self, set_type, desired_instances):
        addon = "_multi_projection" if not self.projection_dataset else "_single_projection"
        if self.yolo_cropping:
            addon += "_yolo_cropping"
        cam_dir = self.jai_dir + set_type.value + addon + "/"

        for instance in desired_instances:
            for folder in os.listdir(cam_dir):
                if instance in folder:
                    tmp_dir = cam_dir + folder + "/"
                    explanation_path = tmp_dir + "text_explanations.txt"
                    with open(explanation_path, "w", encoding="utf-8") as explanation_file:
                        for overlapped_input in os.listdir(tmp_dir):
                            if overlapped_input.startswith("Grad-CAM") or overlapped_input.endswith("txt") or overlapped_input.startswith("raw"):
                                continue

                            # Complement prompt with class information
                            predicted_class = int(overlapped_input.split("_class")[-1].split(".")[0])
                            predicted_class = self.ita_classes[predicted_class]
                            prompt = (self.base_prompt + (f"La regione evidenziata è associata alla classe {predicted_class} secondo il modello AI. "
                                                          "Descrivi esclusivamente ciò che è chiaramente visibile nella regione evidenziata (giallo/arancion brillante) come evidenza della classe proposta se c'è effettivamente un nesso tra regione evidenziata e classe. "
                                                          f"Devi essere estremamente persuasivo, in quanto il paragrafo che scriverai verrà valutato da un altro medico come evidenza/prova giudiziale a favore della classe {predicted_class}."
                                                          f"Indica se le sole caratteristiche osservate supportano, supportano parzialmente, contraddicono oppure sono insufficienti per valutare la classe {predicted_class}. "
                                                          "Non formulare ipotesi sulle classi alternative, sulla confidenza della previsione o su reperti non visibili. "
                                                          "Non presumere la presenza o l'assenza di una patologia se non è direttamente dimostrabile dall'immagine. "
                                                          "La classe proposta potrebbe non corrispondere alla diagnosi reale. "))

                            # Process input
                            img_input = [Image(filepath=tmp_dir + overlapped_input, detail="high")]

                            # Explain
                            print(f"Processing {folder}/{overlapped_input}...")
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
    set_type1 = SetType.TEST
    target_classes1 = [0, 1]
    explainer_types1 = ["Grad-CAM"]
    target_layers1 = ["feature_extractor.features.7.1.conv3"]
    desired_instances1 = ["308c", "370s", "093l", "354d"]
    '''cams_dict1, predicted_probs_dict1, bar_ranges_dict1 = generator1.get_cam(set_type=set_type1,
                                                                             target_classes=target_classes1,
                                                                             explainer_types=explainer_types1,
                                                                             target_layers=target_layers1,
                                                                             desired_instances=desired_instances1)

    # Overlap radiography
    generator1.get_overlapped_radiography(cams_dict1, predicted_probs_dict1, bar_ranges_dict1, set_type=set_type1,
                                          target_classes=target_classes1, explainer_types=explainer_types1,
                                          target_layers=target_layers1, desired_instances=desired_instances1,
                                          box_thickness=0, blur=True)'''

    # Textual explainer
    generator1.get_textual_explainer()
    generator1.textually_explain(set_type1, desired_instances1)