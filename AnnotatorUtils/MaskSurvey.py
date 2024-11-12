# Import packages
import random
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import os
import pytz
import cv2
import pandas as pd
from datetime import datetime

from DataUtils.XrayDataset import XrayDataset
from DataUtils.PatientInstance import PatientInstance
from Enumerators.ProjectionType import ProjectionType
from TrainUtils.NetworkTrainer import NetworkTrainer
from AnnotateMasks import AnnotateMasks


# Class
class MaskSurvey:
    # Define class attributes
    mask_fold = "masks/"
    empty_mask_update = gr.update(value={"background": None, "layers": [], "composite": None}, show_label=False,
                                  interactive=False)
    allow_interaction = gr.update(interactive=True)
    avoid_interaction = gr.update(interactive=False)
    normal_scale_value = 1

    def __init__(self, dataset, desired_instances, blur=False, dataset_name=None):
        self.users = None
        self.dataset = dataset
        if dataset_name is not None:
            self.mask_fold = dataset_name + "_" + self.mask_fold
        self.mask_dir = dataset.data_dir + self.mask_fold
        self.max_projection_number = dataset.max_projection_number()

        self.desired_instances = desired_instances
        random.shuffle(self.desired_instances)
        print("Item name:")
        for i, name in enumerate(self.desired_instances):
            print(" " + str(i + 1) + ") " + name)

        self.blur = blur

    def process_mask(self, name, count, mask_id, adjust_specifics, mask):
        if count != len(self.desired_instances):
            # Define mask name
            pt_folder = self.mask_dir + name
            img_name = self.desired_instances[count]
            segment_dir = pt_folder + "/" + img_name
            if img_name not in os.listdir(pt_folder):
                os.mkdir(segment_dir)
            projections = os.listdir(segment_dir)

            # Store mask
            gt_mask = mask["layers"][0]
            gt_mask = MaskSurvey.remove_adjust(gt_mask, mask_id, adjust_specifics)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
            ax.axis("off")
            plt.imshow(gt_mask, "gray")

            projection_name = "projection" + str(mask_id)
            images = [img for img in projections if projection_name in img]
            plt.savefig(segment_dir + "/" + projection_name + "_vers" + str(len(images)) + ".jpg",
                        format="tif", bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            button = gr.update(icon="./icons/checked.png")
            ok_flag = True
        else:
            button = gr.update()
            ok_flag = False
        return button, name, ok_flag

    def avoid_clear_action(self, count, mask_id, adjust_specifics):
        _, img = self.get_img(mask_id=mask_id, count=count)

        mask = gr.update(value={"background": img, "layers": [], "composite": None})
        bright = self.normal_scale_value
        contrast = self.normal_scale_value

        return mask, bright, contrast, adjust_specifics

    def adjust_img(self, bright, contrast, adjust_specifics, count, mask_id):
        _, img = self.get_img(mask_id=mask_id, count=count)

        # Adjust brightness
        img = np.int32(img + 255 * (bright - self.normal_scale_value) / self.normal_scale_value)
        img = np.clip(img, 0, 255)

        # Adjust contrast
        mean = np.mean(img)
        img = np.int32((contrast / self.normal_scale_value) * (img - mean) + mean)
        img = np.clip(img, 0, 255)

        # Rotate
        img = np.rot90(img, k=adjust_specifics[mask_id, 0])

        # Flip vertically
        if adjust_specifics[mask_id, 1]:
            img = np.fliplr(img)

        # Flip horizontally
        if adjust_specifics[mask_id, 2]:
            img = np.flipud(img)

        mask = gr.update(value={"background": img, "layers": [], "composite": None})
        return mask

    def rotate_img(self, bright, contrast, adjust_specifics, count, mask_id):
        rotate_id = adjust_specifics[mask_id, 0] + 1
        adjust_specifics[mask_id, 0] = rotate_id % 4

        # Undo flips
        if adjust_specifics[mask_id, 1]:
            adjust_specifics[mask_id, 1] = 0
            gr.Warning("ATTENZIONE! L'operazione 'Specchia' è stata annullata. Ti consiglio di definire prima "
                       "l'orientazione dell'immmagine e specchiarla o capovolgerla solo in seguito.")
        if adjust_specifics[mask_id, 2]:
            adjust_specifics[mask_id, 2] = 0
            gr.Warning("ATTENZIONE! L'operazione 'Capovolgi' è stata annullata. Ti consiglio di definire prima "
                       "l'orientazione dell'immmagine e specchiarla o capovolgerla solo in seguito.")

        mask = self.adjust_img(bright, contrast, adjust_specifics, count, mask_id)
        return mask, adjust_specifics

    def flip_vert_img(self, bright, contrast, adjust_specifics, count, mask_id):
        mask, adjust_specifics = self.flip_img(1, bright, contrast, adjust_specifics, count, mask_id)

        return mask, adjust_specifics

    def flip_horiz_img(self, bright, contrast, adjust_specifics, count, mask_id):
        mask, adjust_specifics = self.flip_img(2, bright, contrast, adjust_specifics, count, mask_id)

        return mask, adjust_specifics

    def flip_img(self, direction_ind, bright, contrast, adjust_specifics, count, mask_id):
        adjust_specifics[mask_id, direction_ind] = 1 - adjust_specifics[mask_id, direction_ind]
        mask = self.adjust_img(bright, contrast, adjust_specifics, count, mask_id)

        return mask, adjust_specifics

    def get_img(self, mask_id, item=None, count=None, first_display=False):
        if first_display:
            projection_id = ProjectionType.AP
            img = np.zeros((10, 10))
        else:
            if item is None:
                item, _ = self.dataset.__getitem__(count)
            projection_id, img, _ = item[mask_id]

            img = np.int32(img / np.max(img) * 255)
            if self.blur:
                img = cv2.GaussianBlur(img.astype(np.uint8), (105, 105), 0)
        return projection_id, img

    def next_img(self, count, name, box, ok_flag):
        if count != len(self.desired_instances):
            # Verify if task is completed
            if not box and not ok_flag:
                # Generate warning
                gr.Warning("ATTENZIONE! Per proseguire dovresti evidenziare la frattura in almeno una proiezione altrimenti"
                           " dovresti segnalarne l'assenza con l'apposito checkbox.")
            else:
                if box and ok_flag:
                    gr.Warning("ATTENZIONE! Nel set di immagini numero " + str(count + 1) + " hai inviato una o più "
                               + "maschere, ma hai anche contrassegnato il checkbox per l'assenza di frattura. Vista "
                               + "l'inconsistenza, ti chiediamo di comunicarci via mail se le maschere caricate sono "
                               + "errate.")
                elif box and not ok_flag:
                    # Create an empty folder for the non-fracture segments
                    img_name = self.desired_instances[count]
                    pt_folder = self.mask_dir + name
                    if img_name not in os.listdir(pt_folder):
                        os.mkdir(pt_folder + "/" + img_name)

                count += 1

        # Get next images
        if count == len(self.desired_instances):
            out_txt = "GRAZIE PER IL TUO CONTRIBUTO! Ora puoi chiudere l'applicazione."
            mask_blocks = self.max_projection_number * (5 * [self.avoid_interaction] + [self.empty_mask_update] +
                                                        [gr.update(icon="icons/unchecked.png", interactive=True)])
        else:
            instance = self.desired_instances[count]
            _, segment = PatientInstance.get_patient_and_segment(instance)
            out_txt = self.get_label(count, segment)
            item, _ = self.dataset.__getitem__(count)
            mask_blocks = []
            for j in range(self.max_projection_number):
                try:
                    projection_id, img = self.get_img(mask_id=j, item=item)
                    label = "Vista: " + projection_id.translated_value()

                    # Enable tools and update image
                    mask_blocks += (5 * [self.allow_interaction] +
                                    [gr.update(value={"background": img, "layers": [], "composite": None}, label=label,
                                               show_label=True, interactive=True)] +
                                    [gr.update(icon="icons/unchecked.png", interactive=True)])
                except IndexError:
                    # Disable tools and update image
                    mask_blocks += (5 * [self.avoid_interaction] + [self.empty_mask_update] +
                                    [gr.update(icon="icons/unchecked.png", interactive=False)])

        adjust_specifics = np.zeros((self.max_projection_number, 3))
        box = False
        ok_flag = False
        return out_txt, count, box, ok_flag, adjust_specifics, *mask_blocks

    def change_page(self, name):
        # Check for name correctness
        for char in ["^", "~", "\"", "#", "%", "&", "*", ":", "<", ">", "?", "/", "\\", "{", "}", "|"]:
            if char in name:
                gr.Warning("ATTENZIONE! Il tuo nominativo non può contenere nessuno dei seguenti caratteri:\n"
                           "^ ~ \" # % & * : < > ? / \\ { } |")
                return name, -1, gr.update(), gr.update()

        # Create folder
        if name == "":
            now = datetime.now(tz=pytz.timezone("Europe/Rome"))
            name = "session_" + now.strftime("%m-%d-%Y_%H'%M'%S")
        pt_folder = self.mask_dir + name
        if name not in os.listdir(self.mask_dir):
            os.mkdir(pt_folder)

        # Assess already evaluated instances
        annotated_segments = []
        for instance in self.desired_instances:
            if instance in os.listdir(pt_folder):
                annotated_segments.append(instance)
        count = len(annotated_segments)
        if count > 0:
            count -= 2
        else:
            count = -1

        tab1 = gr.update(interactive=False)
        tab2 = gr.update(interactive=True)
        return name, count, tab1, tab2

    def display_images(self, block):
        count = gr.State(0)

        # Add initial fields
        with gr.Tab(label="Autenticazione utente", interactive=True) as tab1:
            name = gr.Textbox(placeholder="Inserisci qui il tuo cognome...", show_label=False, max_lines=1)
            start = gr.Button(value="Vai alla prossima sezione", icon="icons/next.png")

        with gr.Tab(label="Annotazione immagini", interactive=False) as tab2:
            # Add segment number
            txt = gr.Text(value="", show_label=False, max_lines=1)
            ok_flag = gr.State(False)

            # Add image fields
            adjust_specifics = gr.State(np.zeros((self.max_projection_number, 3)))
            mask_blocks = []
            with gr.Row():
                for i in range(self.max_projection_number):
                    with gr.Column(min_width=200):
                        # Get image
                        projection_id, img = self.get_img(mask_id=i, first_display=True)
                        label = "Vista: " + projection_id.translated_value()
                        show_label = True
                        interactive = True

                        # Add identifier to the image
                        mask_id = gr.State(i)

                        # Add adjust brightness and contrast sliders
                        bright = gr.Slider(minimum=0, maximum=2 * self.normal_scale_value,
                                           value=self.normal_scale_value,
                                           step=self.normal_scale_value / 100, label="Correggi luminosità",
                                           show_label=True, container=True, interactive=interactive, visible=True)
                        mask_blocks.append(bright)
                        contrast = gr.Slider(minimum=0, maximum=10 * self.normal_scale_value,
                                             value=self.normal_scale_value,
                                             step=self.normal_scale_value / 100, label="Correggi contrasto",
                                             show_label=True, container=True, interactive=interactive, visible=True)
                        mask_blocks.append(contrast)

                        # Add rotate button
                        with gr.Row():
                            rotate = gr.Button(value="Ruota", icon="icons/rotate.png", interactive=interactive,
                                               min_width=66)
                            mask_blocks.append(rotate)

                            # Add flip buttons
                            flip_vert = gr.Button(value="Specchia", icon="icons/flip_vert.png", interactive=interactive,
                                                  min_width=66)
                            mask_blocks.append(flip_vert)
                            flip_horiz = gr.Button(value="Capovolgi", icon="icons/flip_horiz.png", interactive=interactive,
                                                   min_width=66)
                            mask_blocks.append(flip_horiz)

                        # Add image editor
                        fig = {"background": img, "layers": [], "composite": None}
                        mask = gr.ImageEditor(value=fig, image_mode="L", sources=None, type="numpy",
                                              label=label, show_label=show_label, show_download_button=False,
                                              container=True, interactive=interactive, visible=True,
                                              show_share_button=False, eraser=gr.Eraser(default_size="auto"),
                                              brush=gr.Brush(default_size="auto", colors=["#ff0000"], color_mode="fixed"),
                                              layers=False, transforms=(), elem_id="output_image")
                        mask_blocks.append(mask)

                        # Add submit button
                        button = gr.Button(value="Invia maschera", icon="icons/unchecked.png", interactive=interactive)
                        mask_blocks.append(button)

                        # Add action listeners
                        if img is not None:
                            bright.release(fn=self.adjust_img, inputs=[bright, contrast, adjust_specifics, count, mask_id],
                                           outputs=[mask])
                            contrast.release(fn=self.adjust_img,
                                             inputs=[bright, contrast, adjust_specifics, count, mask_id],
                                             outputs=[mask])
                            rotate.click(fn=self.rotate_img,
                                         inputs=[bright, contrast, adjust_specifics, count, mask_id],
                                         outputs=[mask, adjust_specifics])
                            flip_vert.click(fn=self.flip_vert_img,
                                            inputs=[bright, contrast, adjust_specifics, count, mask_id],
                                            outputs=[mask, adjust_specifics])
                            flip_horiz.click(fn=self.flip_horiz_img,
                                             inputs=[bright, contrast, adjust_specifics, count, mask_id],
                                             outputs=[mask, adjust_specifics])
                            mask.clear(fn=self.avoid_clear_action, inputs=[count, mask_id, adjust_specifics],
                                       outputs=[mask, bright, contrast, adjust_specifics])
                            button.click(fn=self.notify_click, inputs=[mask_id])
                            button.click(fn=self.process_mask, inputs=[name, count, mask_id, adjust_specifics, mask],
                                         outputs=[button, name, ok_flag])

            # Add final buttons
            box = gr.Checkbox(label="Non ho individuato alcuna frattura")
            next_button = gr.Button(value="Vai al prossimo gruppo di immagini", icon="icons/next.png")
            next_button.click(fn=self.next_img, inputs=[count, name, box, ok_flag],
                              outputs=[txt, count, box, ok_flag, adjust_specifics] + mask_blocks)
        start.click(fn=self.change_page, inputs=[name], outputs=[name, count, tab1, tab2], concurrency_id="start",
                    concurrency_limit=1)
        start.click(fn=self.next_img, inputs=[count, name, gr.State(False), gr.State(True)],
                    outputs=[txt, count, box, ok_flag, adjust_specifics] + mask_blocks, concurrency_id="start",
                    concurrency_limit=1)

    def get_label(self, counter, segment):
        return ("Segmento  " + str(counter + 1) + " / " + str(len(self.desired_instances)) +
                "   -   Segmento " + XrayDataset.get_segment_name_ita(segment))

    def build_app(self):
        # Set up the application
        with gr.Blocks(gr.themes.Soft()) as block:
            gr.Markdown(
                """
                # Judicial AI in Riabilitazione: Sviluppo di Sistemi di Supporto Decisionale Basati su Evidenza per Diagnosi e Pianificazione di Terapie
                #### Dopo aver inserito il tuo nominativo, vedrai in successione delle immagini radiografiche relative a più soggetti: ogni gruppo di immagini rappresenta più proiezioni dello stesso segmento vertebrale di ogni paziente.
                #### Se è presente una frattura, ti chiediamo di evidenziarla colorando la zona interessata in tutte le immagini dove la frattura risulta effettivamente osservabile (anche se parzialmente). Ti invitiamo a contornare o colorare tale zona con adeguata precisione.
                #### Qualora la frattura non fosse presente, segnalalo con l'apposito checkbox in fondo alla pagina.\n
                #### ATTENZIONE! Se la piattaforma dovesse segnalare un errore nel caricamento di una o più componenti (_Error_ sulla componente) oppure un problema di connessione (_Connection errored out._ come pop-up in alto a destra), ciò potrebbe essere legato ad un'instabilità della nostra rete di laboratorio: per risolvere il problema basterà ricaricare la pagina e ripetere l'autenticazione.
                ### BUON LAVORO!
                """
            )
            self.display_images(block)

        # Launch the application
        block.launch(share=True)

    def check_collected_masks(self, round_precision=2):
        # Collect classification outcome
        print("-------------------------------------------------------------------------------------------------------")
        self.users = [x for x in os.listdir(self.mask_dir) if "." not in x and x not in AnnotateMasks.annotator_labels
                      and AnnotateMasks.result_folder not in x]
        col_names = ["Item name", "Fracture position"] + self.users
        all_labels = []
        acc = [0] * len(self.users)
        user_tp = dict(zip(self.users, acc))
        user_tn = dict(zip(self.users, acc))
        user_fp = dict(zip(self.users, acc))
        user_fn = dict(zip(self.users, acc))
        and_accuracy = 0
        or_accuracy = 0
        majority_accuracy = 0
        n_instances = len(self.desired_instances)
        for i in range(n_instances):
            item_name = self.desired_instances[i]
            instance, _ = self.dataset.__getitem__(i)
            labels = [item_name, instance[0][-1]]
            true_label = int(labels[1] != "")
            for user in self.users:
                user_label = int(len(os.listdir(self.mask_dir + "/" + user + "/" + item_name)) > 0)
                labels.append(user_label)
                if true_label:
                    if user_label:
                        user_tp[user] += 1
                    else:
                        user_fn[user] += 1
                else:
                    if user_label:
                        user_fp[user] += 1
                    else:
                        user_tn[user] += 1

            all_labels.append(labels)

            user_labels = np.array(labels[2:])
            gt_label = np.array(true_label)
            if np.all(user_labels == gt_label):
                and_accuracy += 1
            if np.any(user_labels == gt_label):
                or_accuracy += 1
            if np.sum(user_labels == gt_label) >= len(self.users) / 2:
                majority_accuracy += 1
        df = pd.DataFrame(all_labels, columns=col_names)
        df.to_csv(self.mask_dir + "classification_results.csv", index=False)

        # Normalize accuracies
        user_accuracies = {user: MaskSurvey.normalize_acc(user_tp[user] + user_tn[user], n_instances) for user in
                           self.users}
        for user in self.users:
            print("Accuracy for " + user + ": " + str(user_accuracies[user]) + "%")

            # Draw confusion matrices
            cm = [[user_tp[user], user_fp[user]], [user_fn[user], user_tn[user]]]
            filename = user.replace(" ", "_").lower()
            NetworkTrainer.draw_multiclass_confusion_matrix(cm, XrayDataset.classes, self.mask_dir + filename + "_cm.jpg")
        print()
        and_accuracy = MaskSurvey.normalize_acc(and_accuracy, n_instances)
        print("Accuracy if every user is right: " + str(and_accuracy) + "%")
        or_accuracy = MaskSurvey.normalize_acc(or_accuracy, n_instances)
        print("Accuracy if at least one user is right: " + str(or_accuracy) + "%")
        majority_accuracy = MaskSurvey.normalize_acc(majority_accuracy, n_instances)
        print("Accuracy if the majority of users are right: " + str(majority_accuracy) + "%")

        # Check mask-projection correspondence
        print("-------------------------------------------------------------------------------------------------------")
        for user in self.users:
            print("Correspondence check for " + user + "...")
            for i in range(n_instances):
                instance, _ = self.dataset.__getitem__(i)
                instance_name = self.desired_instances[i]
                instance_path = self.mask_dir + user + "/" + instance_name
                masks = os.listdir(instance_path)
                if len(masks) == 0:
                    continue
                unique_projections = np.unique(np.array([int(mask[10]) for mask in masks]))
                if not np.isin(unique_projections, np.arange(len(instance))).all():
                    print(" > There are too many masks for " + instance_name)
                for mask in masks:
                    proj_id = int(mask[10])
                    # Check version numbers
                    if "vers0" not in mask:
                        num_vers = int(mask[-5])
                        for vers in range(num_vers):
                            if mask[:-5] + str(vers) + ".jpg" not in masks:
                                print(" > Version " + str(vers) + " of projection " + str(proj_id) + " is missing for " + instance_name)

                    # Check mask sizes
                    mask_shape = cv2.imread(instance_path + "/" + mask).shape[:-1]
                    try:
                        img_shape = instance[proj_id][1].shape
                        w_ratio = mask_shape[1] / img_shape[1]
                        h_ratio = mask_shape[0] / img_shape[0]
                        wh_ratio = w_ratio / h_ratio
                        if round(wh_ratio, round_precision) != 1:
                            print(" > " + mask + " does not correspond in shape to the original projection for " + instance_name)
                            self.find_shape_matches(mask_shape, round_precision)
                    except IndexError:
                        print(" > " + mask + " has no correspondent projection for " + instance_name)
                        self.find_shape_matches(mask_shape, round_precision)
            print()

    def find_shape_matches(self, mask_shape, round_precision=2):
        for i in range(len(self.desired_instances)):
            instance, _ = self.dataset.__getitem__(i)
            for j in range(len(instance)):
                img_shape = instance[j][1].shape
                w_ratio = mask_shape[1] / img_shape[1]
                h_ratio = mask_shape[0] / img_shape[0]
                wh_ratio = w_ratio / h_ratio
                if round(wh_ratio, round_precision) == 1:
                    print("    - Possible match found in projection " + str(j) + " of " + self.desired_instances[i])

    @staticmethod
    def remove_adjust(img, mask_id, adjust_specifics):
        if adjust_specifics[mask_id, 2]:
            img = np.flipud(img)
        if adjust_specifics[mask_id, 1]:
            img = np.fliplr(img)
        img = np.rot90(img, k=adjust_specifics[mask_id, 0] * -1)

        return img

    @staticmethod
    def notify_click(mask_id):
        gr.Info("Invio maschera " + str(mask_id + 1) + " in corso...")

    @staticmethod
    def normalize_acc(x, n_instances):
        return np.round(x / n_instances * 100, 2)


# Main
if __name__ == "__main__":
    # Set seed
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    dataset_name1 = "xray_dataset_validation"
    blur1 = False

    # Define data
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Launch app
    survey = MaskSurvey(dataset=dataset1, desired_instances=dataset1.dicom_instances, blur=blur1,
                        dataset_name=dataset_name1)
    print("Add '?__theme=dark' at the end of the link")
    # survey.build_app()

    # Check masks
    survey.check_collected_masks()

    print()
