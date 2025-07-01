# Import packages
import random
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import os
import pandas as pd

from DataUtils.XrayDataset import XrayDataset
from AnnotatorUtils.MaskSurvey import MaskSurvey


# Class
class ImageCropSurvey(MaskSurvey):
    # Define class attributes
    mask_fold = "cropped_imgs/"
    empty_mask_update = gr.update(value={"image": np.zeros((1000, 1000)), "boxes": []}, show_label=False,
                                  interactive=False)
    intro_msg = """
                # Judicial AI in Riabilitazione: Sviluppo di Sistemi di Supporto Decisionale Evidence-First per Diagnosi e Pianificazione di Terapie
                #### Dopo aver inserito il tuo nominativo, vedrai in successione delle immagini radiografiche relative a più soggetti: ogni gruppo di immagini rappresenta più proiezioni dello stesso segmento vertebrale di ogni paziente.
                #### Nell'immagine saranno presenti più vertebre: ti chiediamo di evidenziarle con riquadri differenti. Ti invitiamo a contornare la zona con adeguata precisione. Sarà necessario contornare solo le vertebre interamente presenti nell'immagine.
                #### Qualora non vi fossero vertebre nell'immagine o fosse impossibile identificarle e contornarle, segnalalo con l'apposito checkbox in fondo alla pagina.\n
                #### ATTENZIONE! Se la piattaforma dovesse segnalare un errore nel caricamento di una o più componenti (_Error_ sulla componente) oppure un problema di connessione (_Connection errored out._ come pop-up in alto a destra), ciò potrebbe essere legato ad un'instabilità della nostra rete di laboratorio: per risolvere il problema basterà ricaricare la pagina e ripetere l'autenticazione.
                ### BUON LAVORO!
                """
    vertebra_names = (["C generica"] + [f"C{i}" for i in range(1, 8)] + ["D generica"] + [f"D{i}" for i in range(1, 13)]
                      + ["L generica"] + [f"L{i}" for i in range(1, 6)] + ["Sacro/Coccige"])
    vertebra_values = [n if len(n) <= 3 else n[0] for n in vertebra_names]
    vertebra_dict = dict(zip(vertebra_names, vertebra_values))

    def __init__(self, dataset, desired_instances, dataset_name=None):
        super(ImageCropSurvey, self).__init__(dataset, desired_instances, False, dataset_name, True)

    def process_mask(self, name, count, mask_id, adjust_specifics, mask):
        cropping_ref_name = "cropping_ref.csv"
        if cropping_ref_name not in os.listdir(self.mask_dir):
            cropping_ref = pd.DataFrame(columns=["file_name", "segment", "projection", "projection_type", "width",
                                                 "height", "vertebra_name", "x_min", "x_max", "y_min", "y_max",
                                                 "fracture_present", "annotator"])
        else:
            cropping_ref = pd.read_csv(self.mask_dir + cropping_ref_name)

        if count != len(self.desired_instances):
            # Define mask name
            img_name = self.desired_instances[count]

            # Get image and selected boxes
            project_id, img, label = self.get_img(mask_id, count=count, return_label=True)
            img = np.rot90(img, k=adjust_specifics[mask_id, 0])
            if adjust_specifics[mask_id, 1]:
                img = np.fliplr(img)
            if adjust_specifics[mask_id, 2]:
                img = np.flipud(img)
            img = np.stack([img] * 3, axis=-1)
            boxes = mask["boxes"]

            # Set image classification label
            vertebra_names = [self.vertebra_dict[box["label"]] for box in boxes]
            fracture_presents = [False for _ in vertebra_names]
            if label != "":
                fractured_vertebrae = label.split("-")
                fractured_segments = [fv[0] for fv in fractured_vertebrae]
                if "S" in fractured_segments:
                    fractured_vertebrae += ["S"]

                flag = False
                for i, vertebra_name in enumerate(vertebra_names):
                    if vertebra_name in fractured_vertebrae:
                        fracture_presents[i] = True
                        flag = True
                if not flag:
                    for i, vertebra_name in enumerate(vertebra_names):
                        if vertebra_name in fractured_segments:
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
                cropped_img_name = img_name + "_proj" + str(mask_id) + "_" + vertebra_name.replace("/", "") + ".png"
                plt.savefig(self.mask_dir + "/" + cropped_img_name, bbox_inches="tight", pad_inches=0, dpi=600)
                plt.close(fig)

                # Store cropping references
                cropping_ref = cropping_ref[cropping_ref["file_name"] != cropped_img_name]
                row = {"file_name": cropped_img_name, "segment": img_name, "projection": mask_id,
                       "projection_type": project_id.value, "width": img.shape[1], "height": img.shape[0],
                       "vertebra_name": vertebra_name, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                       "fracture_present": fracture_presents[i], "annotator": name}
                cropping_ref.loc[len(cropping_ref)] = row

            # Store cropping references
            cropping_ref.to_csv(self.mask_dir + cropping_ref_name, index=False)

            button = gr.update(icon="./icons/checked.png")
            ok_flag = True
        else:
            button = gr.update()
            ok_flag = False
        return button, name, ok_flag


# Main
if __name__ == "__main__":
    # Set seed
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    dataset_name1 = "xray_dataset_validation"

    # Define data
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Launch app
    survey = ImageCropSurvey(dataset=dataset1, desired_instances=dataset1.dicom_instances, dataset_name=dataset_name1)
    print("Add '?__theme=dark' at the end of the link")
    survey.build_app()
