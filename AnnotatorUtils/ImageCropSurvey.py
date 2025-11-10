# Import packages
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
import pytz
import ast
import shutil
from datetime import datetime

from DataUtils.XrayDataset import XrayDataset
from AnnotatorUtils.MaskSurvey import MaskSurvey


# Class
class ImageCropSurvey(MaskSurvey):
    # Define class attributes
    mask_fold = "cropped_imgs/"
    empty_mask_update = gr.update(value={"image": np.zeros((1000, 1000)), "boxes": []}, show_label=True,
                                  interactive=True)

    intro_msg = """
                # Judicial AI in Riabilitazione: Sviluppo di Sistemi di Supporto Decisionale Evidence-First per Diagnosi e Valutazione di Terapie
                #### Dopo aver inserito il tuo nominativo, vedrai in successione delle immagini radiografiche relative a più soggetti: ogni gruppo di immagini rappresenta più proiezioni dello stesso segmento vertebrale di ogni paziente.
                #### In ogni immagine saranno presenti più vertebre: ti chiediamo di evidenziarle con riquadri differenti. Ti invitiamo a contornare la zona con adeguata precisione. Sarà necessario contornare solo le vertebre interamente presenti nell'immagine.
                #### Qualora non fosse possibile identificare vertebre in nessuna immagine del set, segnalalo con l'apposito checkbox in fondo alla pagina.\n
                #### ATTENZIONE! Se la piattaforma dovesse segnalare un errore nel caricamento di una o più componenti (_Error_ sulla componente) oppure un problema di connessione (_Connection errored out._ come pop-up in alto a destra), ciò potrebbe essere legato ad un'instabilità della nostra rete di laboratorio: per risolvere il problema basterà ricaricare la pagina e ripetere l'autenticazione.
                ### BUON LAVORO!
                """
    vertebra_names = (["C generica (NO frattura)"] + ["C generica (frattura)"] + [f"C{i}" for i in range(1, 8)] +
                      ["D generica (NO frattura)"] + ["D generica (frattura)"] + [f"D{i}" for i in range(1, 13)]
                      + ["L generica (NO frattura)"] + ["L generica (frattura)"] + [f"L{i}" for i in range(1, 6)] +
                      ["Sacro/Coccige"])
    vertebra_values = [n if len(n) <= 3 else n[0] for n in vertebra_names]
    for i in range(len(vertebra_values)):
        if i > 0 and vertebra_values[i] == vertebra_values[i - 1]:
            vertebra_values[i] += "f"
    vertebra_dict = dict(zip(vertebra_names, vertebra_values))
    checkbox_label = "Non ho individuato alcuna vertebra in nessuna delle proiezioni"
    cropping_ref_name = "cropping_ref.csv"

    def __init__(self, dataset, desired_instances, dataset_name=None, blur=False, annotators_list=None, extra_col=0):
        super(ImageCropSurvey, self).__init__(dataset, desired_instances, blur, dataset_name, is_crop_gui=True,
                                              annotators_list=annotators_list, extra_col=extra_col)

    def process_mask(self, name, count, count_all, mask_id, adjust_specifics, mask):
        if self.annotators_list is None:
            desired_instances = self.desired_instances
        else:
            desired_instances = self.assignments[name]

        user_folder = self.mask_dir + name + "/"
        if self.cropping_ref_name not in os.listdir(user_folder):
            cropping_ref = pd.DataFrame(columns=["file_name", "segment", "projection", "projection_type", "width",
                                                 "height", "vertebra_name", "x_min", "x_max", "y_min", "y_max",
                                                 "fracture_present", "annotator", "timestamp", "annotator_counter",
                                                 "global_counter"])
        else:
            cropping_ref = pd.read_csv(user_folder + self.cropping_ref_name)

        if count != len(desired_instances):
            # Define mask name
            img_name = desired_instances[count]

            # Get image and selected boxes
            project_id, img, label = self.get_img(mask_id, count_all=count_all, return_label=True)
            img = np.rot90(img, k=adjust_specifics[mask_id, 0])
            if adjust_specifics[mask_id, 1]:
                img = np.fliplr(img)
            if adjust_specifics[mask_id, 2]:
                img = np.flipud(img)
            img = np.stack([img] * 3, axis=-1)
            boxes = mask["boxes"]

            # Set image classification label
            vertebra_names = [self.vertebra_dict[box["label"]] for box in boxes]
            fracture_presents = ["f" in vn for vn in vertebra_names]
            if label != "":
                fractured_vertebrae = label.split("-")
                fractured_segments = np.unique([fv[0] for fv in fractured_vertebrae]).tolist()
                fractured_vertebrae = [fv for fv in fractured_vertebrae if len(fv) > 1]
                if "S" in fractured_segments:
                    fractured_vertebrae += ["S"]

                for i, vertebra_name in enumerate(vertebra_names):
                    if "f" in vertebra_name:
                        continue
                    if len(vertebra_name) > 1 and vertebra_name in fractured_vertebrae:
                        fracture_presents[i] = True
                    elif (len(vertebra_name) == 1 and vertebra_name in fractured_segments and
                          vertebra_name + "f" not in fractured_vertebrae):
                        fracture_presents[i] = True

            # Store boxes
            for i, box in enumerate(boxes):
                vertebra_name = vertebra_names[i]
                x_min = box["xmin"]
                x_max = box["xmax"]
                y_min = box["ymin"]
                y_max = box["ymax"]

                # Store cropped images
                img_patch = img[y_min:y_max+1, x_min:x_max+1]
                img_patch = self.remove_adjust(img_patch, mask_id, adjust_specifics)
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
                ax.axis("off")
                plt.imshow(img_patch, "gray")
                cropped_img_name = img_name + "_proj" + str(mask_id) + "_" + vertebra_name.replace("/", "")
                same_name_imgs = []
                for im in os.listdir(user_folder):
                    if im[-3:] != "png":
                        continue
                    if im.startswith(cropped_img_name):
                        if "f" not in im and "f" not in cropped_img_name or "f" in im and "f" in cropped_img_name:
                            same_name_imgs.append(im)
                cropped_img_name += "_" + str(len(same_name_imgs)) + ".png"
                plt.savefig(user_folder + "/" + cropped_img_name, bbox_inches="tight", pad_inches=0, dpi=600)
                plt.close(fig)

                # Store cropping references
                now = datetime.now(tz=pytz.timezone("Europe/Rome"))
                now = now.strftime("%m-%d-%Y %H:%M:%S")
                tmp_vertebra_name = vertebra_name.strip("f")
                row = {"file_name": cropped_img_name, "segment": img_name, "projection": mask_id,
                       "projection_type": project_id.value, "width": img.shape[1], "height": img.shape[0],
                       "vertebra_name": tmp_vertebra_name, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                       "fracture_present": fracture_presents[i], "annotator": name, "timestamp": now,
                       "annotator_counter": count, "global_counter": count_all}
                print("-----------------------------------------------------------------------------------------------")
                print(row)
                cropping_ref.loc[len(cropping_ref)] = row

            # Store cropping references
            cropping_ref.to_csv(user_folder + self.cropping_ref_name, index=False)

            button = gr.update(icon="./icons/checked.png")
            ok_flag = True
        else:
            button = gr.update()
            ok_flag = False
        return button, name, ok_flag

    def check_collected_masks(self, round_precision=2):
        # Read command line output
        command_line_output = []
        with open(self.mask_dir + "../command_line_output.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    row = ast.literal_eval(line)
                    command_line_output.append(row)
        command_line_output = pd.DataFrame(command_line_output)
        command_line_output = command_line_output.drop_duplicates(ignore_index=True)

        # Recover missing references
        for user in self.annotators_list:
            user_folder = self.mask_dir + user + "/"
            cropped_patches = [file for file in os.listdir(user_folder) if ".csv" not in file]
            cropping_ref = pd.read_csv(user_folder + self.cropping_ref_name)
            original_n_rows = len(cropping_ref)
            unreferenced_patches = set(cropped_patches) - set(cropping_ref.file_name)
            print(user.upper(), "has", len(unreferenced_patches), "missing references...")

            for patch_name in unreferenced_patches:
                missing_row = command_line_output[command_line_output["file_name"] == patch_name]
                if len(missing_row) == 0:
                    print(patch_name, "not found!")
                else:
                    cropping_ref = pd.concat([cropping_ref, missing_row], ignore_index=True)
            print(" >", len(cropping_ref) - original_n_rows, "references recovered for", user.upper())

            cropping_ref = cropping_ref.sort_values(by=["global_counter", "file_name"], ascending=[True, True])
            cropping_ref.to_csv(user_folder + "adjusted_" + self.cropping_ref_name, index=False)

    def pool_collected_masks(self):
        print("\nPooling procedure starting...")
        mask_dir = self.mask_dir + "pooled/"
        if "pooled" not in os.listdir(self.mask_dir):
            os.mkdir(mask_dir)

        cropping_refs = []
        for user in self.annotators_list:
            user_folder = self.mask_dir + user + "/"
            cropping_ref = pd.read_csv(user_folder + "adjusted_" + self.cropping_ref_name)
            for row in cropping_ref.itertuples(index=False):
                vertebra_name = row.vertebra_name
                file_name = row.file_name
                if vertebra_name == "S" or len(vertebra_name) > 1:
                    root = "_".join(file_name.split("_")[:-1])
                    most_recent_patch = cropping_ref[cropping_ref["file_name"].str.startswith(root)].file_name.max()
                    if file_name != most_recent_patch:
                        print(" >", file_name, "discharged")
                        continue
                src_path = os.path.join(user_folder, file_name)
                dst_path = os.path.join(mask_dir, file_name)
                shutil.copy2(src_path, dst_path)
                cropping_refs.append(row)
        cropping_refs = pd.DataFrame(cropping_refs)
        cropping_refs.to_csv(mask_dir + "pooled_" + self.cropping_ref_name, index=False)


# Main
if __name__ == "__main__":
    # Set seed
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    dataset_name1 = "xray_dataset_training"

    # Define data
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Launch app
    blur1 = False
    annotators_list1 = ["salina", "ciccone", "brevi"]
    survey = ImageCropSurvey(dataset=dataset1, desired_instances=dataset1.dicom_instances, dataset_name=dataset_name1,
                             blur=blur1, annotators_list=annotators_list1, extra_col=1)
    print("Add '?__theme=dark' at the end of the link")
    # survey.build_app()

    # Check masks
    survey.check_collected_masks()

    # Pool masks
    survey.pool_collected_masks()
