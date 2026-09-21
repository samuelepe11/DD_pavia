# Import packages
import random
import cv2
import gradio as gr
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import base64
import warnings

from fontTools.unicodedata import block
from ipywidgets import interactive
from opt_einsum.paths import branch
from sympy.physics.units import current

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# NEEDS Gradio 4.44.1
# Classes
class ImageReadError(Exception):
    pass


class SurveyCreator:
    # Define class attributes
    results_fold = "results/"
    survey_fold = "survey_results/"
    data_fold = "jai_results/"
    results_file_name = "survey_results.csv"
    binary_answers = ["No", "Sì"]
    sex_levels = ["Maschio", "Femmina", "Non binario", "Altro", "Preferisco non rispondere"]
    career_levels = ["Non ho ancora iniziato", "I anno", "II anno", "III anno", "IV anno", "V anno", "Ho terminato la specializzazione"]
    likert_choices = ["Fortemente in disaccordo", "In disaccordo", "Parzialmente in disaccordo", "Parzialmente d'accordo", "D'accordo",
                      "Fortemente d'accordo"]
    confidence_levels = ["Molto bassa", "Bassa", "Abbastanza bassa", "Abbastanza alta", "Alta", "Molto alta"]
    diagnoses = ["Non ci sono fratture", "Il paziente presenta fratture"]
    branches = ["Supporto bounding-box", "Supporto heatmap (calda)", "Supporto heatmap (fredda)"]
    colormaps = ["inferno", "gist_earth"]

    # Change this depending on the selected strategy
    data_fold += "cropped_projection_resnext50_simpler_transpose_equalize/test_single_projection_yolo_cropping/"
    cam_names = ["HiResCAM_feature_extractor_features_7_1_conv3_class", "HiResCAM_feature_extractor_features_7_2_conv3_class"]
    max_projection_number = 4

    # Update attributes
    allow_interaction = gr.update(interactive=True)
    avoid_interaction = gr.update(interactive=False)
    empty_radio = gr.update(value=None)
    normal_scale_value = 1

    # Messages
    intro_msg = """#### Istruzioni:
                   - Dovrai <span style='color:#f97316;'>effettuare l'accesso scegliendo uno username</span> (ad esempio, il tuo cognome). Questo ti consentirà di interrompere il questionario e riprenderlo successivamente, se necessario.
                   - Ti verrà somministrato un breve <span style='color:#f97316;'>questionario di profilazione</span>.
                   - Visualizzerai i dati RX di 24 pazienti, che includono almeno due proiezioni per ognuno (antero-posteriore e/o laterale). 
                     * <span style='color:#f97316;'>Dovrai effetturare una valutazione, supportato dall'IA</span>: in ordine casuale valuterai (a) 8 casi con supporto esplicito (bounding-box), (b) 8 casi con un supporto basato su mappa di calore a colorazione calda, (c) 8 casi con un secondo supporto indiretto basato su mappa di calore a colori freddi. Ogni sistema verrà meglio dettagliato nella schermata dedicata.
                     * <span style='color:#f97316;'>Per ciascun RX dovrai fornire una diagnosi</span> - elencando se il paziente presenta fratture e, in caso, quali vertebre sono riguardate e dove - e valutare la tua confidenza diagnostica, la complessità del caso e l'utilità del supporto.
                   - Dopo l'utilizzo di ciascun sistema, compilerai un breve <span style='color:#f97316;'>questionario per valutare la tua esperienza complessiva</span>.
                """
    sub_descriptions = {
        branches[0]: branches[0] + "\n"
                     "Per ogni proiezione associata al caso clinico, verranno evidenziate mediante un <span style='color:#f97316;'>riquadro rosso "
                     "solo le vertebre per cui il modello IA predice la presenza di una frattura</span>. "
                     "Per ciascuna vertebra evidenziata verrà inoltre mostrata la confidenza del modello, espressa come probabilità "
                     "(dove 1.000 rappresenta la massima certezza). <span style='color:#f97316;'>Se non vedi box significa che il modello non ha rilevato alcuna frattura</span>.",
        branches[1]: branches[1] + "\n"
                     "Per ogni proiezione associata al caso clinico non verranno mostrate né la predizione del modello IA né la relativa confidenza. "
                     "Verrà invece visualizzata una <span style='color:#f97316;'>mappa di calore che evidenzia le aree dell'immagine considerate più importanti dal modello per la classificazione di 'frattura presente'</span>. "
                     "Le aree più scure (grigio/viola) indicano una minore importanza, mentre quelle più brillanti (arancione/giallo) indicano una maggiore importanza per la classificazione.",
        branches[2]: branches[2] + "\n"
                     "Per ogni proiezione associata al caso clinico non verranno mostrate né la predizione del modello IA né la relativa confidenza. "
                     "Verrà invece visualizzata una <span style='color:#f97316;'>mappa di calore che evidenzia le aree dell'immagine considerate più importanti dal modello per la classificazione di 'frattura presente'</span>. "
                     "Le aree più scure (grigio/blu) indicano una minore importanza, mentre quelli più brillanti (giallo/bianco) indicano una maggiore importanza per la classificazione."
    }
    scroll_js = """() => {
                    setTimeout(() => {
                        const tab3Button = document.getElementById("tab3-button");
                        const tab4Button = document.getElementById("tab4-button");
                
                        if (
                            tab3Button?.getAttribute("aria-selected") === "true" ||
                            tab4Button?.getAttribute("aria-selected") === "true"
                        ) {
                            window.scrollTo({top: 0, left: 0, behavior: "auto"});
                        }
                    }, 50);
                }"""

    def __init__(self, working_dir, desired_instances, debug_mode=False):
        # Define attributes
        self.working_dir = working_dir
        self.results_dir = working_dir + self.results_fold
        self.data_dir = self.results_dir + self.data_fold
        self.survey_dir = self.results_dir + self.survey_fold
        if self.survey_fold[:-1] not in os.listdir(self.results_dir):
            os.mkdir(self.survey_dir)

        if not isinstance(desired_instances, dict):
            self.n_instances = len(desired_instances)
            self.desired_instances = desired_instances.copy()
            random.shuffle(self.desired_instances)
            print("Presented patients:")
            for i, name in enumerate(self.desired_instances):
                print(" " + str(i + 1) + ") " + name)
        else:
            self.n_instances = len(desired_instances["block 1"]) * 3
            self.desired_instances = {}
            for k, v in desired_instances.items():
                print(k.upper() + ":")
                block = v.copy()
                random.shuffle(block)
                self.desired_instances.update({k: block})
                for i, name in enumerate(block):
                    print(" " + str(i + 1) + ") " + name)
        self.debug_mode = debug_mode

    def avoid_clear_action(self, proj_idx, state_dict):
        img, exp = self.get_img(state_dict=state_dict, proj_idx=proj_idx)
        img = self.overlap_input(img, exp, state_dict)
        img_display = gr.update(value=img)
        bright = self.normal_scale_value
        contrast = self.normal_scale_value
        adjust_specifics = state_dict["adjust_specifics"]
        adjust_specifics[proj_idx, :] = 0
        state_dict.update({"adjust_specifics": adjust_specifics})
        return img_display, bright, contrast, state_dict

    def adjust_img(self, bright, contrast, proj_idx, state_dict):
        img, exp = self.get_img(state_dict=state_dict, proj_idx=proj_idx)
        adjust_specifics = state_dict["adjust_specifics"]

        # Adjust brightness
        img = np.int32(img + 255 * (bright - self.normal_scale_value) / self.normal_scale_value)
        img = np.clip(img, 0, 255)

        # Adjust contrast
        mean = np.mean(img)
        img = np.int32((contrast / self.normal_scale_value) * (img - mean) + mean)
        img = np.clip(img, 0, 255)

        # Rotate
        img = np.rot90(img, k=adjust_specifics[proj_idx, 0])

        # Flip vertically
        if adjust_specifics[proj_idx, 1]:
            img = np.fliplr(img)

        # Flip horizontally
        if adjust_specifics[proj_idx, 2]:
            img = np.flipud(img)

        # Adjust explanation
        exp = np.rot90(exp, k=adjust_specifics[proj_idx, 0])
        if adjust_specifics[proj_idx, 1]:
            exp = np.fliplr(exp)
        if adjust_specifics[proj_idx, 2]:
            exp = np.flipud(exp)

        # Hide/show evidence
        if not adjust_specifics[proj_idx, 3]:
            img = self.overlap_input(img, exp, state_dict)

        img_display = gr.update(value=img)
        return img_display

    def rotate_img(self, bright, contrast, proj_idx, state_dict):
        adjust_specifics = state_dict["adjust_specifics"]
        rotate_id = adjust_specifics[proj_idx, 0] + 1
        adjust_specifics[proj_idx, 0] = rotate_id % 4

        # Undo flips
        if adjust_specifics[proj_idx, 1]:
            adjust_specifics[proj_idx, 1] = 0
            gr.Warning("ATTENZIONE! L'operazione 'Specchia' è stata annullata. Ti consiglio di definire prima "
                       "l'orientazione dell'immmagine e specchiarla o capovolgerla solo in seguito.")
        if adjust_specifics[proj_idx, 2]:
            adjust_specifics[proj_idx, 2] = 0
            gr.Warning("ATTENZIONE! L'operazione 'Capovolgi' è stata annullata. Ti consiglio di definire prima "
                       "l'orientazione dell'immmagine e specchiarla o capovolgerla solo in seguito.")

        state_dict.update({"adjust_specifics": adjust_specifics})
        img_display = self.adjust_img(bright=bright, contrast=contrast, proj_idx=proj_idx, state_dict=state_dict)
        return img_display, state_dict

    def flip_vert_img(self, bright, contrast, proj_idx, state_dict):
        img_display, state_dict = self.flip_img(direction_ind=1, bright=bright, contrast=contrast, state_dict=state_dict,
                                                proj_idx=proj_idx)
        return img_display, state_dict

    def flip_horiz_img(self, bright, contrast, proj_idx, state_dict):
        img_display, state_dict = self.flip_img(direction_ind=2, bright=bright, contrast=contrast, state_dict=state_dict,
                                                proj_idx=proj_idx)
        return img_display, state_dict

    def flip_img(self, direction_ind, bright, contrast, proj_idx, state_dict):
        adjust_specifics = state_dict["adjust_specifics"]
        adjust_specifics[proj_idx, direction_ind] = 1 - adjust_specifics[proj_idx, direction_ind]
        state_dict.update({"adjust_specifics": adjust_specifics})
        img_display = self.adjust_img(bright=bright, contrast=contrast, state_dict=state_dict, proj_idx=proj_idx)
        return img_display, state_dict

    def show_evidence(self, bright, contrast, state_dict, proj_idx):
        support_click_times = state_dict["support_click_times"]
        support_click_times[proj_idx].append(datetime.datetime.now().isoformat(timespec="milliseconds"))
        adjust_specifics = state_dict["adjust_specifics"]
        adjust_specifics[proj_idx, 3] = 1 - adjust_specifics[proj_idx, 3]
        state_dict.update({"adjust_specifics": adjust_specifics, "support_click_times": support_click_times})

        img_display = self.adjust_img(bright=bright, contrast=contrast, state_dict=state_dict, proj_idx=proj_idx)
        if not adjust_specifics[proj_idx, 3]:
            show = gr.update(value="Nascondi supporto decisionale", icon="icons/lightbulb_off.png")
        else:
            show = gr.update(value="Mostra supporto decisionale", icon="icons/lightbulb_on.png")
        return img_display, state_dict, show

    def get_img(self, state_dict=None, proj_idx=None, first_display=False):
        try:
            count = state_dict["count"]
            desired_instances = state_dict["desired_instances"]
        except TypeError:
            first_display = True
            count = None
            desired_instances = None
        if first_display or (count is not None and (count < 0 or count >= self.n_instances)):
            img = np.zeros((1000, 500))
            base_path = None
        else:
            if isinstance(desired_instances, dict):
                try:
                    current_branch = state_dict["current_branch"]
                    branches_order = state_dict["branches_order"]
                except TypeError:
                    current_branch = None
                    branches_order = None
                block_idx = branches_order.index(current_branch)
                try:
                    desired_instances = list(desired_instances.values())[block_idx]
                    count_eff = count - block_idx * self.n_instances // 3
                    desired_instance = desired_instances[count_eff]
                    flag = False
                except IndexError:
                    block_idx += 1
                    desired_instances = list(state_dict["desired_instances"].values())[block_idx]
                    count_eff = count - block_idx * self.n_instances // 3
                    desired_instance = desired_instances[count_eff]
                    flag = True
            else:
                desired_instance = desired_instances[count]

            base_path = self.data_dir + desired_instance + "_proj" + str(proj_idx)
            if not os.path.isfile(base_path + "/raw_image.png"):
                raise ImageReadError("Instance does not exist: " + base_path)
            else:
                img = cv2.imread(base_path + "/raw_image.png")
            if self.debug_mode:
                print("Retrieved image for item", count, "projection", proj_idx)

        # AI-support extra operations
        try:
            current_branch = state_dict["current_branch"]
            cam_order = state_dict["cam_order"]
        except TypeError:
            current_branch = None
            cam_order = None
        if current_branch is None or base_path is None or flag:
            return img, img
        else:
            if current_branch == self.branches[0]:
                exp_name = base_path + "/predicted_boxes.png"
                cmap = None
            elif current_branch == self.branches[1]:
                exp_name = base_path + "/" + cam_order[0] + str(1) + ".png"
                cmap = "inferno"
                gist_earth_linear = None
            else:
                exp_name = base_path + "/" + cam_order[1] + str(1) + ".png"
                cmap = "gist_earth"

                # Adjust map brightness
                base_cmap = plt.get_cmap("gist_earth")
                x = np.linspace(0, 1, 1024)
                rgb = base_cmap(x)[:, :3]
                rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = 0.2126 * rgb_linear[:, 0] + 0.7152 * rgb_linear[:, 1] + 0.0722 * rgb_linear[:, 2]
                luminance = np.maximum.accumulate(luminance)
                target_luminance = np.linspace(luminance[0], luminance[-1], 256)
                new_x = np.interp(target_luminance, luminance, x)
                gist_earth_linear = LinearSegmentedColormap.from_list("gist_earth_linear", base_cmap(new_x))
            if cmap is None:
                exp = cv2.imread(exp_name, cv2.IMREAD_COLOR)
                exp = cv2.cvtColor(exp, cv2.COLOR_BGR2RGB)
            else:
                exp = cv2.imread(exp_name, cv2.IMREAD_GRAYSCALE)
            if exp is None:
                print("Issue with explanation '" + exp_name + "'.")
                exp = img
            else:
                if cmap is not None:
                    exp = exp / np.max(exp)
                    selected_cmap = gist_earth_linear if cmap == "gist_earth" else plt.get_cmap(cmap)
                    exp = selected_cmap(exp)[:, :, :3]
                    exp = (exp * 255).astype(np.uint8)

            # Outdated case
            if current_branch == "Supporto testuale":
                exp = {}
                for directory in os.listdir(self.data_dir):
                    with open(self.data_dir + directory + "/model_predictions.txt", "r", encoding="utf-8") as f:
                        for line in f:
                            class_id, probability = line.strip().split(":", 1)
                            if class_id.strip() == "1":
                                prob_class_1 = float(probability.strip())
                                break
                    if prob_class_1 > 0.0:
                        with open(self.data_dir + directory + "text_explanations.txt", "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        for line_idx, line in enumerate(lines):
                            if line.strip() == exp_name and line_idx + 1 < len(lines):
                                exp.update({directory: lines[line_idx + 1].strip()})
            return img, exp

    def login(self, state_dict, name):
        # Check for name correctness
        preference_msg = gr.update()
        preference = gr.update()
        preference_txt = gr.update()
        sub_descr = gr.update()
        for char in ["^", "~", "\"", "#", "%", "&", "*", ":", "<", ">", "?", "/", "\\", "{", "}", "|"]:
            if char in name:
                gr.Warning("ATTENZIONE! Il tuo username non può contenere i seguenti caratteri:\n"
                           "^ ~ \" # % & * : < > ? / \\ { } |")
                state_dict.update({"count": -10})
                return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), state_dict,
                        preference, preference_txt, sub_descr, sub_descr)

        # Create folder
        if name not in os.listdir(self.survey_dir):
            os.mkdir(self.survey_dir + name)
        else:
            gr.Success("Lo username '" + name + "' è già presente nel nostro database. I tuoi progressi precedenti "
                                                "verranno caricati e potrai continuare la compilazione del survey da "
                                                "dove l'hai interrotto.")

        # Check for survey result file
        state_dict.update({"name": name})
        user_folder = self.survey_dir + name + "/"
        if self.results_file_name in os.listdir(user_folder):
            results_file = pd.read_csv(user_folder + self.results_file_name)
            try:
                count = results_file["index"].iloc[-1] + 1
            except IndexError:
                count = 0
            try:
                branches_order = results_file["group"].unique().tolist()
                if len(branches_order) == 0:
                    branches_order = None
                elif len(branches_order) == 1:
                    user_seed = hash(name) % 1_000_000
                    rng = random.Random(user_seed)
                    branches_order += rng.sample([b for b in self.branches if b not in branches_order], k=2)
                else:
                    branches_order += [b for b in self.branches if b not in branches_order]
            except IndexError:
                branches_order = None
            try:
                cam_order = results_file["cam"].dropna().unique().tolist()
                if len(cam_order) == 0:
                    cam_order = None
                else:
                    cam_order += [c for c in self.cam_names if c not in cam_order]
            except IndexError:
                cam_order = None
        else:
            results_file = pd.DataFrame(columns=["index", "instance", "annotator", "group", "cam", "diagnosis", "location",
                                                 "confidence", "complexity", "usefulness", "time"])
            results_file.to_csv(user_folder + self.results_file_name, index=False)
            count = 0
            branches_order = None
            cam_order = None
        if self.debug_mode:
            print("==============================")
            print("Initial count:", count)

        # Read previously-evaluated instances
        if not isinstance(self.desired_instances, dict):
            try:
                evaluated_instances = results_file["instance"].unique().tolist()
                remaining_instances = [ins for ins in self.desired_instances if ins not in evaluated_instances]
            except KeyError:
                evaluated_instances = []
                remaining_instances = self.desired_instances
        else:
            # Reorder blocks
            user_seed = hash(name) % 1_000_000
            rng = random.Random(user_seed)
            items = list(self.desired_instances.items())
            rng.shuffle(items)
            ordered_desired_instances = dict(items)

            # Read ordered blocks
            evaluated_all = results_file["instance"].dropna().tolist()
            evaluated_instances = {k: [instance for instance in v if instance in evaluated_all]
                                   for k, v in ordered_desired_instances.items()}
            remaining_instances = {k: [instance for instance in v if instance not in evaluated_instances[k]]
                                   for k, v in ordered_desired_instances.items()}
            remaining_instances = dict(sorted(remaining_instances.items(), key=lambda x: len(x[1]), reverse=False))

        # Shuffle remaining instances
        user_seed = hash(name) % 1_000_000
        rng = random.Random(user_seed)
        if not isinstance(self.desired_instances, dict):
            rng.shuffle(remaining_instances)
            desired_instances = evaluated_instances + remaining_instances
        else:
            desired_instances = {}
            for k, v in remaining_instances.items():
                v_tmp = v.copy()
                rng.shuffle(v_tmp)
                desired_instances.update({k: evaluated_instances[k] + v_tmp})
        state_dict.update({"desired_instances": desired_instances})
        if self.debug_mode:
            print("Instances for '" + name + "':")
            if not isinstance(self.desired_instances, dict):
                for i, instance in enumerate(desired_instances):
                    print(" " + str(i + 1) + ") " + instance)
            else:
                for k, v in desired_instances.items():
                    print(k.upper() + ":")
                    for i, instance in enumerate(v):
                        print(" " + str(i + 1) + ") " + instance)

        # Define user branch
        if branches_order is None:
            user_seed = hash(name) % 1_000_000
            rng = random.Random(user_seed)
            branches_order = rng.sample(self.branches, k=len(self.branches))
        if cam_order is None:
            user_seed = hash(name) % 1_000_000
            rng = random.Random(user_seed)
            cam_order = rng.sample(self.cam_names, k=len(self.cam_names))
        state_dict.update({"branches_order": branches_order, "cam_order": cam_order})

        if self.debug_mode:
            print("User '" + name + "' is assigned to the following group order:", branches_order, "with CAM order",
                  cam_order)

        tab1 = self.avoid_interaction
        tab2 = gr.update()
        tab3 = gr.update()
        tab4 = gr.update()
        tab5 = gr.update()
        performance_flag = False
        descr_flag = False
        state_dict.update({"avoid_subsequent_next": True})
        if "final_questionnaire.csv" in os.listdir(user_folder) and pd.read_csv(user_folder + "final_questionnaire.csv").shape[0] == 3:
            # Finish
            tab5 = self.allow_interaction
            selected = 5
        elif count == self.n_instances:
            # Go to final questionnaire C
            tab4 = self.allow_interaction
            selected = 4
            state_dict.update({"current_branch": state_dict["branches_order"][2], "preliminary_flag": True})
            performance_flag = True
        elif "final_questionnaire.csv" in os.listdir(user_folder) and pd.read_csv(user_folder + "final_questionnaire.csv").shape[0] == 2:
            # Go to branch C
            tab3 = self.allow_interaction
            selected = 3
            state_dict.update({"current_branch": state_dict["branches_order"][2]})
            performance_flag = True
            descr_flag = True
        elif count == 2 * self.n_instances // 3:
            # Go to final questionnaire B
            tab4 = self.allow_interaction
            selected = 4
            state_dict.update({"current_branch": state_dict["branches_order"][1], "preliminary_flag": True})
        elif "final_questionnaire.csv" in os.listdir(user_folder) and pd.read_csv(user_folder + "final_questionnaire.csv").shape[0] == 1:
            # Go to branch B
            tab3 = self.allow_interaction
            selected = 3
            state_dict.update({"current_branch": state_dict["branches_order"][1]})
            descr_flag = True
        elif count == self.n_instances // 3:
            # Go to final questionnaire A
            tab4 = self.allow_interaction
            selected = 4
            state_dict.update({"current_branch": state_dict["branches_order"][0], "preliminary_flag": True})
        elif "preliminary_questionnaire.csv" in os.listdir(user_folder):
            # Go to branch A
            tab3 = self.allow_interaction
            selected = 3
            state_dict.update({"current_branch": state_dict["branches_order"][0]})
            '''if count == 0:
                state_dict.update({"preliminary_flag": True})'''
            descr_flag = True
        else:
            # Go to preliminary questionnaire
            tab2 = self.allow_interaction
            selected = 2
            state_dict.update({"preliminary_flag": True, "current_branch": state_dict["branches_order"][0]})
            descr_flag = True
        tabs = gr.update(selected=selected)

        if performance_flag:
            preference_msg = gr.update(visible=True)
            preference = gr.update(choices=branches_order + ["Non ho preferenze"],
                                   label="Quale modalità di support hai preferito?", visible=True)
            preference_txt = gr.update(visible=True)
        if descr_flag:
            sub_descr = (f"## Sistema {branches_order.index(state_dict['current_branch']) + 1}/{len(self.branches)} - " +
                         self.sub_descriptions[state_dict["current_branch"]])

        state_dict.update({"count": count})
        if self.debug_mode:
            print("Count at 'login' end:", count)
            print("------------------------------")

        return (state_dict, tabs, tab1, tab2, tab3, tab4, tab5, preference_msg, preference, preference_txt, sub_descr,
                sub_descr)

    def start(self, state_dict, age, sex, career, expertise, q1, q2, q3, q4, q5):
        # Store preliminary questionnaire results
        name = state_dict["name"]
        count = state_dict["count"]
        user_folder = self.survey_dir + name + "/"
        titles = ["annotator", "age", "sex", "career", "expertise", "knowledge", "worked", "performance", "productivity",
                  "effectiveness", "time"]
        if (age == 0 or sex is None or expertise is None or q1 is None or q2 is None or q3 is None or q4 is None or q5 is None):
            gr.Warning("ATTENZIONE! Completa tutti i campi del questionario prima di procedere.")
            q1i = self.binary_answers[q1] if q1 is not None else None
            q2i = self.binary_answers[q2] if q2 is not None else None
            q3i = self.binary_answers[q3] if q3 is not None else None
            q4i = self.binary_answers[q4] if q4 is not None else None
            q5i = self.binary_answers[q5] if q5 is not None else None
            return (state_dict, gr.update(), gr.update(), gr.update(), gr.update(value=age), gr.update(value=sex),
                    gr.update(value=career), gr.update(value=expertise), gr.update(value=q1i), gr.update(value=q2i),
                    gr.update(value=q3i), gr.update(value=q4i), gr.update(value=q5i))
        values = [state_dict["name"], age, sex, career, expertise, q1, q2, q3, q4, q5, datetime.datetime.now()]
        df = pd.DataFrame([values], columns=titles)
        df.to_csv(user_folder + "preliminary_questionnaire.csv", index=False)

        tabs = gr.update(selected=3)
        tab2 = gr.update(interactive=False)
        tab3 = gr.update(interactive=True)
        if self.debug_mode:
            print("------------------------------")
            print("Count at 'start' end:", count)
            print("------------------------------")
        return (state_dict, tabs, tab2, tab3, gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None))

    def get_image_blocks(self, state_dict):
        img_blocks = []
        for i in range(self.max_projection_number):
            try:
                img, exp = self.get_img(state_dict=state_dict, proj_idx=i)
                interactive = True
            except ImageReadError:
                img, exp = self.get_img(first_display=True)
                interactive = False
            img_tmp = gr.update(value=self.overlap_input(img, exp, state_dict)) if img is not None else gr.update()
            img_blocks += (5 * [gr.update(interactive=interactive)] +
                           [gr.update(icon="icons/lightbulb_off.png", interactive=interactive), img_tmp])
        return img_blocks

    def next(self, state_dict, diagnosis, location, confidence, complexity, usefulness):
        current_time = datetime.datetime.now().isoformat(timespec="milliseconds")
        name = state_dict["name"]
        count = state_dict["count"]
        preliminary_flag = state_dict["preliminary_flag"]
        avoid_subsequent_next = state_dict["avoid_subsequent_next"]
        desired_instances = state_dict["desired_instances"]

        # Save hide/show click times
        support_click_times = state_dict["support_click_times"]
        support_clicks = ""
        for i, sct in enumerate(support_click_times):
            sct.append(current_time)
            addon = "\n" if i != 0 else ""
            support_clicks += addon + "proj" + str(i) + ": " + "; ".join(sct)

        if isinstance(self.desired_instances, dict):
            current_branch = state_dict["current_branch"]
            branches_order = state_dict["branches_order"]
            block_idx = branches_order.index(current_branch)
            desired_instances = list(desired_instances.values())[block_idx]
        else:
            block_idx = 0

        if self.debug_mode and not preliminary_flag and not avoid_subsequent_next:
            print("------------------------------")
            print("Count at 'next' start:", count)
            print("------------------------------")

        if count == -10 or count > self.n_instances or avoid_subsequent_next:
            img_blocks = self.get_image_blocks(state_dict)
            if avoid_subsequent_next:
                state_dict.update({"avoid_subsequent_next": False})
            return (state_dict, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), *img_blocks)

        # Save diagnosis
        user_folder = self.survey_dir + name + "/"
        file_path = user_folder + self.results_file_name
        results_file = pd.read_csv(file_path)
        if not preliminary_flag:
            # Store results
            if diagnosis is None or confidence is None or complexity is None or usefulness is None or (diagnosis and location == ""):
                gr.Warning("ATTENZIONE! Non hai completato la valutazione del paziente. Per favore, completa tutti"
                           " i campi prima di procedere.")
                diag = self.diagnoses[diagnosis] if diagnosis is not None else None
                conf = self.confidence_levels[confidence] if confidence is not None else None
                comp = self.confidence_levels[complexity] if complexity is not None else None
                usef = self.confidence_levels[usefulness] if usefulness is not None else None
                img_blocks = self.get_image_blocks(state_dict)
                state_dict.update({"support_click_times": support_click_times})
                return (state_dict, gr.update(value=diag), location, gr.update(value=conf), gr.update(value=comp),
                        gr.update(value=usef), gr.update(), gr.update(), gr.update(), *img_blocks)
            if self.debug_mode:
                print("Storing image results for image", count, "\n")
            if isinstance(self.desired_instances, dict):
                count_eff = count - block_idx * self.n_instances // 3
            else:
                count_eff = count
            current_branch = state_dict["current_branch"]

            # Set CAM name
            if current_branch == self.branches[0]:
                cam = np.nan
            elif current_branch == self.branches[1]:
                cam = state_dict["cam_order"][0]
            else:
                cam = state_dict["cam_order"][1]

            new_row = {"index": count, "instance": desired_instances[count_eff], "annotator": name,
                       "group": current_branch, "cam": cam, "diagnosis": int(diagnosis), "location": location,
                       "confidence": int(confidence), "complexity": int(complexity), "usefulness": int(usefulness),
                       "time": datetime.datetime.now(), "support_clicks": support_clicks}
            results_file = pd.concat([results_file, pd.DataFrame([new_row])], ignore_index=True)
            results_file.to_csv(file_path, index=False)

        tab3 = gr.update()
        tab4 = gr.update()
        tabs = gr.update()
        if not preliminary_flag:
            count += 1
        if count in [self.n_instances // 3, 2 * self.n_instances // 3, self.n_instances] and not preliminary_flag:
            tab3 = self.avoid_interaction
            tab4 = self.allow_interaction
            tabs = gr.update(selected=4)
        state_dict.update({"count": count})
        img_blocks = self.get_image_blocks(state_dict)

        state_dict.update({"preliminary_flag": False,
                          "adjust_specifics": np.zeros((self.max_projection_number, 4))})
        if self.debug_mode and not preliminary_flag and not avoid_subsequent_next:
            print("------------------------------")
            print("Count at 'next' end:", count)
            print("------------------------------")

        current_time = datetime.datetime.now().isoformat(timespec="milliseconds")
        state_dict.update({"support_click_times": [[current_time] for _ in range(self.max_projection_number)]})
        return (state_dict, gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), tabs, tab3, tab4, *img_blocks)
                
    def conclude(self, state_dict, q11, q14, q22, q23, q31, q32, q41, q51, preference, preference_txt):
        name = state_dict["name"]
        current_branch = state_dict["current_branch"]
        branches_order = state_dict["branches_order"]
        ask_preference = current_branch == branches_order[-1]
        preference_msg = gr.update()

        # Store final questionnaire results
        user_folder = self.survey_dir + name + "/"
        titles = ["annotator", "competence1", "competence4", "autonomy2", "autonomy3", "relatedness1", "relatedness2",
                  "enjoyment1", "demand1", "group", "preference", "preference_comment", "time"]
        values = [name, q11, q14, q22, q23, q31, q32, q41, q51, current_branch, preference, preference_txt, datetime.datetime.now()]

        if (q11 is None or q14 is None or q22 is None or q23 is None or q31 is None or q32 is None or q41 is None
                or q51 is None or (ask_preference and (preference is None or preference_txt == ""))):
            gr.Warning("ATTENZIONE! Completa tutti i campi del questionario prima di procedere.")
            q11 = self.likert_choices[q11] if q11 is not None else None
            q14 = self.likert_choices[q14] if q14 is not None else None
            q22 = self.likert_choices[q22] if q22 is not None else None
            q23 = self.likert_choices[q23] if q23 is not None else None
            q31 = self.likert_choices[q31] if q31 is not None else None
            q32 = self.likert_choices[q32] if q32 is not None else None
            q41 = self.likert_choices[q41] if q41 is not None else None
            q51 = self.likert_choices[q51] if q51 is not None else None
            return (state_dict, gr.update(), gr.update(), gr.update(), gr.update(), q11, q14, q22, q23, q31, q32, q41,
                    q51, preference_msg, preference, preference_txt, gr.update(), gr.update())
        df = pd.DataFrame([values], columns=titles)
        if current_branch != branches_order[0]:
            df_old = pd.read_csv(user_folder + "final_questionnaire.csv")
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(user_folder + "final_questionnaire.csv", index=False)

        tab3 = gr.update()
        tab4 = gr.update(interactive=False)
        tab5 = gr.update()
        if ask_preference:
            tabs = gr.update(selected=5)
            tab5 = gr.update(interactive=True)
        else:
            tabs = gr.update(selected=3)
            tab3 = gr.update(interactive=True)
            if current_branch == branches_order[1]:
                state_dict.update({"current_branch": branches_order[2]})
                preference_msg = gr.update(visible=True)
                preference = gr.update(choices=branches_order + ["Non ho preferenze"], visible=True)
                preference_txt = gr.update(visible=True)
            else:
                state_dict.update({"current_branch": branches_order[1]})

        sub_descr = (f"## Sistema {branches_order.index(state_dict['current_branch']) + 1}/{len(self.branches)} - " +
                     self.sub_descriptions[state_dict["current_branch"]])
        state_dict.update({"preliminary_flag": True})
        if self.debug_mode:
            print("------------------------------")
            print("Count at 'conclude' end:", state_dict["count"])
            print("------------------------------")

        current_time = datetime.datetime.now().isoformat(timespec="milliseconds")
        state_dict.update({"support_click_times": [[current_time] for _ in range(self.max_projection_number)]})
        return (state_dict, tabs, tab3, tab4, tab5, gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), preference_msg, preference, preference_txt, sub_descr,
                sub_descr)

    def display_tabs(self, block):
        state_dict = gr.State({"count": 0, "branches_order": None, "current_branch": self.branches[0],
                               "adjust_specifics": np.zeros((self.max_projection_number, 4)), "preliminary_flag": False,
                               "avoid_subsequent_next": False, "desired_instances": self.desired_instances,
                               "support_click_times": [[] for _ in range(self.max_projection_number)], "cam_order": None})
        with gr.Tabs(selected=1) as tabs:
            with gr.Tab(id=1, label="Autenticazione", interactive=True) as tab1:
                gr.Markdown("### Accedi con il tuo cognome...")
                name = gr.Textbox(placeholder="Inserisci qui il tuo cognome...", show_label=False, max_lines=1)
                login_btn = gr.Button(value="Accedi", icon="icons/next.png", variant="primary")

            with gr.Tab(id=2, label="Questionario di Profilazione", interactive=False) as tab2:
                gr.Markdown("### Compila il seguente questionario")
                with gr.Column(min_width=500):
                    gr.Markdown("#### Informazioni Personali")
                    with gr.Row():
                        age = gr.Number(label="Quanti anni hai?", step=1, precision=0)
                    with gr.Row():
                        sex = gr.Radio(choices=self.sex_levels, label="Con quale genere ti identifichi?")
                    gr.Markdown("#### Background Clinico")
                    with gr.Row():
                        career = gr.Radio(choices=self.career_levels, label="A che anno di specializzazione sei?")
                    with gr.Row():
                        expertise = gr.Number(label="Approssimativamente, quante diagnosi di fratture vertebrali con "
                                                    "immagini RX hai effettato nella tua cariera?",
                                              step=1, precision=0)
                    gr.Markdown("#### Familiarità con l'IA")
                    with gr.Row():
                        q1 = gr.Radio(choices=self.binary_answers, label="Ho una buona conoscenza sull'IA.", type="index")
                    with gr.Row():
                        q2 = gr.Radio(choices=self.binary_answers, label="Ho lavorato e/o utilizzato sistemi basati "
                                                                         "sull'IA nel mio lavoro", type="index")
                    gr.Markdown("#### Fiducia nell'IA")
                    with gr.Row():
                        q3 = gr.Radio(choices=self.binary_answers, label="Credo che l'IA possa aiutarmi a rispondere più"
                                                                         " correttamente e velocemente a domande di cui "
                                                                         "non conosco la risposta con precisione.", type="index")
                    with gr.Row():
                        q4 = gr.Radio(choices=self.binary_answers, label="Credo che usare l'IA per aiutarmi nel mio "
                                                                         "lavoro o studio possa aumentare la mia "
                                                                         "produttività", type="index")
                    with gr.Row():
                        q5 = gr.Radio(choices=self.binary_answers, label="Credo che con l'aiuto dell'IA io "
                                                                         "possa migliorare l'efficacia del mio lavoro.",
                                      type="index")
                start_btn = gr.Button(value="Inizia", icon="icons/next.png", variant="primary")

            with gr.Tab(id=3, label="Esercizio di Diagnosi", interactive=False, elem_id="tab3") as tab3:
                with gr.Row():
                    sub_descr = gr.Markdown("")
                with gr.Row():
                    gr.Markdown("### Suggerisci una diagnosi per ogni paziente.\n"
                                "Se il supporto decisionale non ti consente di visualizzare correttamente "
                                "l'immagine, puoi <span style='color:#f97316;'>nasconderlo (e farlo riapparire)</span>"
                                " con l'apposito tasto sopra di essa. Ricorda che, oltre ai bottoni presenti sopra ogni "
                                "immagine, puoi utilizzare il tourchpad del tuo laptop per "
                                "<span style='color:#f97316;'>ingrandire o rimpicciolire</span> l'immagine. "
                                "ATTENZIONE: tutti i campi sono obbligatori, ad eccezione del campo testuale in caso di"
                                " assenza di frattura.")
                with gr.Row():
                    img_blocks = []
                    for i in range(self.max_projection_number):
                        with gr.Column(min_width=200):
                            # Get image
                            gr.HTML("<h3 style='text-align:center;'>PROIEZIONE " + str(i + 1) + "</h3>")
                            img, _ = self.get_img(first_display=True)
                            interactive = False
                            proj_id = gr.State(i)

                            # Add adjust brightness and contrast sliders
                            bright = gr.Slider(minimum=0, maximum=2 * self.normal_scale_value,
                                               value=self.normal_scale_value,
                                               step=self.normal_scale_value / 100, label="Correggi luminosità",
                                               show_label=True, container=True, interactive=interactive,
                                               visible=True)
                            img_blocks.append(bright)
                            contrast = gr.Slider(minimum=0, maximum=10 * self.normal_scale_value,
                                                 value=self.normal_scale_value,
                                                 step=self.normal_scale_value / 100, label="Correggi contrasto",
                                                 show_label=True, container=True, interactive=interactive,
                                                 visible=True)
                            img_blocks.append(contrast)

                            # Add buttons
                            with gr.Row():
                                rotate = gr.Button(value="Ruota (90°)", icon="icons/rotate.png", interactive=interactive,
                                                   min_width=66)
                                img_blocks.append(rotate)
                                flip_vert = gr.Button(value="Specchia", icon="icons/flip_vert.png",
                                                      interactive=interactive,
                                                      min_width=66)
                                img_blocks.append(flip_vert)
                                flip_horiz = gr.Button(value="Capovolgi", icon="icons/flip_horiz.png",
                                                       interactive=interactive,
                                                       min_width=66)
                                img_blocks.append(flip_horiz)

                            # Add show evidence button
                            show = gr.Button(value="Nascondi supporto decisionale", icon="icons/lightbulb_off.png",
                                             interactive=interactive)
                            img_blocks.append(show)

                            # Add image display
                            img_display = gr.Image(value=img, image_mode="L", interactive=False, buttons=[])
                            img_blocks.append(img_display)

                            # Add action listeners
                            if img is not None:
                                bright.release(fn=self.adjust_img, inputs=[bright, contrast, proj_id, state_dict],
                                               outputs=[img_display])
                                contrast.release(fn=self.adjust_img, inputs=[bright, contrast, proj_id, state_dict],
                                                 outputs=[img_display])
                                rotate.click(fn=self.rotate_img, inputs=[bright, contrast, proj_id, state_dict],
                                             outputs=[img_display, state_dict])
                                flip_vert.click(fn=self.flip_vert_img, inputs=[bright, contrast, proj_id, state_dict],
                                                outputs=[img_display, state_dict])
                                flip_horiz.click(fn=self.flip_horiz_img, inputs=[bright, contrast, proj_id, state_dict],
                                                 outputs=[img_display, state_dict])
                                img_display.clear(fn=self.avoid_clear_action, inputs=[proj_id, state_dict],
                                                  outputs=[img_display, bright, contrast, state_dict])
                                show.click(fn=self.show_evidence, inputs=[bright, contrast, state_dict, proj_id],
                                           outputs=[img_display, state_dict, show])
                with gr.Row():
                    with gr.Column(min_width=500):
                        with gr.Row():
                            diagnosis = gr.Radio(choices=self.diagnoses, label="Seleziona una diagnosi.", type="index")
                        with gr.Row():
                            location = gr.Textbox(label="In caso affermativo, quali vertebre pensi siano affette da "
                                                        "frattura e dove?", placeholder="ES.  'L3 nella parte superiore'  OPPURE"
                                                                                        "  'prima vertebra in alto nello "
                                                                                        "spigolo sinistro'",
                                                  interactive=False)
                        with gr.Row():
                            confidence = gr.Radio(choices=self.confidence_levels, label="Qual è il tuo livello di "
                                                                                        "confidenza del selezionare "
                                                                                        "la diagnosi?", type="index")
                        with gr.Row():
                            complexity = gr.Radio(choices=self.confidence_levels, label="Come stimeresti la "
                                                                                        "complessità di questo caso "
                                                                                        "clinico?", type="index")
                        with gr.Row():
                            usefulness = gr.Radio(choices=self.confidence_levels, label="Come stimeresti l'utilità "
                                                                                        "del supporto fornito?",
                                                  type="index")
                next = gr.Button(value="Prossimo caso", icon="icons/next.png", variant="primary")

            with gr.Tab(id=4, label="Feedback sul Sistema Utilizzato", interactive=False, elem_id="tab4") as tab4:
                with gr.Row():
                    sub_descr1 = gr.Markdown("")
                with gr.Row():
                    gr.Markdown("### Compila il seguente questionario")
                with gr.Column(min_width=500):
                    gr.Markdown("#### Competenza Percepita")
                    with gr.Row():
                        q11 = gr.Radio(choices=self.likert_choices, label="Penso di aver svolto bene il compito di "
                                                                          "formulare diagnosi durante questa attività.",
                                       type="index")
                    with gr.Row():
                        q14 = gr.Radio(choices=self.likert_choices, label="Dopo aver svolto questo compito per un po', "
                                                                          "mi sono sentito/a piuttosto competente in questo"
                                                                          " compito diagnostico.",
                                       type="index")
                    gr.Markdown("#### Autonomia Percepita")
                    with gr.Row():
                        q22 = gr.Radio(choices=self.likert_choices, label="Mi sono sentito libero/a di scegliere la diagnosi che "
                                                                          "ritenevo più appropriata.", type="index")
                    with gr.Row():
                        q23 = gr.Radio(choices=self.likert_choices, label="Mi sono sentito/a fortemente influenzato/a "
                                                                          "dall'IA nel modo in cui raccomandavo le "
                                                                          "diagnosi.", type="index")
                    gr.Markdown("#### Sintonia con l'IA")
                    with gr.Row():
                        q31 = gr.Radio(choices=self.likert_choices, label="Ho avuto la sensazione di potermi fidare di "
                                                                          "questo sistema basato sull'IA.", type="index")
                    with gr.Row():
                        q32 = gr.Radio(choices=self.likert_choices, label="Ho avuto la sensazione che il mio ragionamento"
                                                                          " in questo compito fosse distante da quello "
                                                                          "del sistema basato su IA fornitomi.",
                                       type="index")
                    gr.Markdown("#### Coinvolgimento")
                    with gr.Row():
                        q41 = gr.Radio(choices=self.likert_choices, label="Mi è piaciuto questo compito diagnostico.",
                                       type="index")
                    gr.Markdown("#### Carico Mentale")
                    with gr.Row():
                        q51 = gr.Radio(choices=self.likert_choices, label="Ho trovato questo compito mentalmente "
                                                                          "impegnativo.", type="index")
                    preference_msg = gr.Markdown("#### Preferenza sul Sistema", visible=False)
                    with gr.Row():
                        preference = gr.Radio(choices=self.branches + ["Non ho preferenze"],
                                              label=("Quale sistema hai preferito? NON BASARTI SUI COLORI UTILIZZATI: i due sistemi "
                                                     "con mappa di calore sono generati da algoritmi completamente diversi, il "
                                                     "colore serve unicamente a distinguerli visivamente e verrà modificato nella "
                                                     "versione definitiva."),
                                              visible=False)
                    with gr.Row():
                        preference_txt = gr.Textbox(label="Spiega brevemente la tua scelta",
                                                    placeholder="Es. sistema meno complesso, spiegazioni più chiare...",
                                                    visible=False, lines=3)
                conclude = gr.Button(value="Concludi sezione", icon="icons/next.png", variant="primary")

            with gr.Tab(id=5, label="Conclusione", interactive=False) as tab5:
                gr.Markdown("# Grazie per aver completato il questionario, ora puoi chiudere la pagina.")

        login_btn.click(fn=self.login, inputs=[state_dict, name],
                        outputs=[state_dict, tabs, tab1, tab2, tab3, tab4, tab5, preference_msg, preference,
                                 preference_txt, sub_descr, sub_descr1],
                        concurrency_id="login", concurrency_limit=1).then(
            fn=self.next, inputs=[state_dict, diagnosis, location, confidence, complexity, usefulness],
            outputs=[state_dict, diagnosis, location, confidence, complexity, usefulness, tabs, tab3, tab4, *img_blocks],
            concurrency_id="login", concurrency_limit=1)
        start_btn.click(fn=self.start, inputs=[state_dict, age, sex, career, expertise, q1, q2, q3, q4, q5],
                        outputs=[state_dict, tabs, tab2, tab3, age, sex, career, expertise, q1, q2, q3, q4, q5],
                        concurrency_id="start", concurrency_limit=1).then(
            fn=self.next, inputs=[state_dict, diagnosis, location, confidence, complexity, usefulness],
            outputs=[state_dict, diagnosis, location, confidence, complexity, usefulness, tabs, tab3, tab4, *img_blocks],
            concurrency_id="login", concurrency_limit=1)
        diagnosis.change(fn=self.show_location, inputs=diagnosis, outputs=location)
        next.click(fn=self.next, inputs=[state_dict, diagnosis, location, confidence, complexity, usefulness],
                   outputs=[state_dict, diagnosis, location, confidence, complexity, usefulness, tabs, tab3, tab4,
                            *img_blocks],
                   concurrency_id="next", concurrency_limit=1).then(fn=None, js=self.scroll_js)
        conclude.click(fn=self.conclude, concurrency_id="conclude", concurrency_limit=1,
                       inputs=[state_dict, q11, q14, q22, q23, q31, q32, q41, q51, preference, preference_txt],
                       outputs=[state_dict, tabs, tab3, tab4, tab5, q11, q14, q22, q23, q31, q32, q41, q51,
                                preference_msg, preference, preference_txt, sub_descr, sub_descr1]).then(
                       fn=self.next, inputs=[state_dict, diagnosis, location, confidence, complexity, usefulness],
                       outputs=[state_dict, diagnosis, location, confidence, complexity, usefulness, tabs, tab3, tab4,
                                *img_blocks],
                       concurrency_id="conclude", concurrency_limit=1)

    def build_app(self, share=False):
        # Set up the application
        with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)) as block:
            img_path = Path(__file__).resolve().parent / "icons" / "rx.png"
            gr.HTML(
                f"""
                <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 18px;">
                    <img src=data:image/png;base64,{base64.b64encode(img_path.read_bytes()).decode("utf-8")} alt="Brain CT icon"
                        style="width: 52px; height: 52px; object-fit: contain;">
                    <h1 style="margin: 0;">VertebrAI</h1>
                </div>
                """
            )
            gr.Markdown(self.intro_msg)
            self.display_tabs(block)

        # Launch the application0
        block.launch(share=share)

    @staticmethod
    def show_location(diagnosis):
        return gr.update(interactive=diagnosis == 1)

    def overlap_input(self, img, exp, state_dict):
        current_branch = state_dict["current_branch"]
        alpha = 0.3 if current_branch != self.branches[0] else 0.5
        if not np.all(img == 0):
            exp = cv2.resize(exp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            img = np.clip((1.0 - alpha) * img + alpha * exp, 0, 255).astype(np.uint8)
        return img


# Main
if __name__ == "__main__":
    # Set seed
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    '''desired_instances1 = ["032d", "039c", "032l", "040d", "446l", "100c"]
    desired_instances1 = {"block 1": ["032d", "039c"],
                          "block 2": ["032l", "040d"],
                          "block 3": ["446l", "100c"]}'''
    desired_instances1 = {"block 1": ["474l", "378d", "405l", "281c", "413l", "297l", "170l", "093l"],
                          "block 2": ["433d", "308c", "312l", "152l", "150l", "330l", "459d", "413d"],
                          "block 3": ["338l", "229c", "386l", "123l", "226l", "354d", "174l", "113l"]}
    debug_mode1 = False
    share1 = True

    # Launch app
    survey = SurveyCreator(working_dir=working_dir1, desired_instances=desired_instances1, debug_mode=debug_mode1)
    print("Add '?__theme=dark' at the end of the link")
    survey.build_app(share=share1)
