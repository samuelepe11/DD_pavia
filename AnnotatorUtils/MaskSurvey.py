# Import packages
import random
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import os
import pytz
from datetime import datetime

from DataUtils.XrayDataset import XrayDataset
from DataUtils.PatientInstance import PatientInstance


# Class
class MaskSurvey:
    # Define class attributes
    mask_fold = "masks/"
    empty_mask_update = gr.update(value={"background": None, "layers": [], "composite": None}, show_label=False,
                                  interactive=False)
    allow_interaction = gr.update(interactive=True)
    avoid_interaction = gr.update(interactive=False)
    normal_scale_value = 1

    def __init__(self, dataset, desired_instances):
        self.dataset = dataset
        self.mask_dir = dataset.data_dir + MaskSurvey.mask_fold
        self.max_projection_number = dataset.max_projection_number()

        self.desired_instances = desired_instances
        random.shuffle(self.desired_instances)

    def process_mask(self, name, count, mask_id, adjust_specifics, mask):
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

    def get_img(self, mask_id, item=None, count=None):
        if item is None:
            item = self.dataset.__getitem__(count)
        projection_id, img, _ = item[mask_id]

        img = np.int32(img / np.max(img) * 255)
        return projection_id, img

    def next_img(self, count, name, box, ok_flag):
        # Verify if task is completed
        if not box and not ok_flag:
            gr.Warning("ATTENZIONE! Per proseguire dovresti evidenziare la frattura in almeno una proiezione altrimenti"
                       " dovresti segnalarne l'assenza con l'apposito checkbox.")
        else:
            # Create an empty folder for the no-fracture segment
            if box:
                img_name = self.desired_instances[count]
                pt_folder = self.mask_dir + name
                if img_name not in os.listdir(pt_folder):
                    os.mkdir(pt_folder + "/" + img_name)

            # Update counter
            if not (box and ok_flag):  # Avoid counter incrementation after pressing of Start button
                count += 1

        # Get next images
        if count == len(self.desired_instances) - 1:
            out_txt = "GRAZIE PER IL TUO CONTRIBUTO! Ora puoi chiudere l'applicazione."
            mask_blocks = self.max_projection_number * (5 * [self.avoid_interaction] + [self.empty_mask_update] +
                                                        [gr.update(icon="icons/unchecked.png", interactive=True)])
        else:
            instance = self.desired_instances[count]
            _, segment = PatientInstance.get_patient_and_segment(instance)
            out_txt = self.get_label(count, segment)
            item = self.dataset.__getitem__(count)
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
            item = self.dataset.__getitem__(count.value)
            mask_blocks = []
            with gr.Row():
                for i in range(self.max_projection_number):
                    with gr.Column(min_width=200):
                        # Get image
                        try:
                            projection_id, img = self.get_img(mask_id=i, item=item)
                            label = "Vista: " + projection_id.translated_value()
                            show_label = True
                            interactive = True
                        except IndexError:
                            label = ""
                            img = None
                            show_label = False
                            interactive = False

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
                            button.click(fn=self.process_mask, inputs=[name, count, mask_id, adjust_specifics, mask],
                                         outputs=[button, name, ok_flag])

            # Add final buttons
            box = gr.Checkbox(label="Non ho individuato alcuna frattura")
            next_button = gr.Button(value="Vai alla prossima immagine", icon="icons/next.png")
            next_button.click(fn=self.next_img, inputs=[count, name, box, ok_flag],
                              outputs=[txt, count, box, ok_flag, adjust_specifics] + mask_blocks)
        start.click(self.change_page, inputs=[name], outputs=[name, count, tab1, tab2], concurrency_id="start",
                    concurrency_limit=1)
        start.click(fn=self.next_img, inputs=[count, name, gr.State(True), gr.State(True)],
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
                #### Per favore, con l'aiuto delle diverse proiezioni mostrate, individua la frattura in ogni immagine proposta, colorando la zona interessata.
                #### Qualora la frattura non fosse presente, segnalalo con l'apposito checkbox.\n
                #### BUON LAVORO!
                """
            )
            self.display_images(block)

        # Launch the application
        x = block.launch(share=True)

    @staticmethod
    def remove_adjust(img, mask_id, adjust_specifics):
        if adjust_specifics[mask_id, 2]:
            img = np.flipud(img)
        if adjust_specifics[mask_id, 1]:
            img = np.fliplr(img)
        img = np.rot90(img, k=adjust_specifics[mask_id, 0] * -1)

        return img


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
    print("Add '?__theme=dark' at the end of the link")
    survey = MaskSurvey(dataset=dataset1, desired_instances=dataset1.dicom_instances)
    survey.build_app()
