# Import packages
import random

import cv2
import gradio as gr
import numpy as np
import os
import pandas as pd
import datetime

import warnings

from joblib.externals.cloudpickle import instance
from triton.language.core import base_type

warnings.simplefilter(action="ignore", category=FutureWarning)


# Class
class SurveyCreator:
    # Define class attributes
    results_fold = "results/"
    survey_fold = "survey_results/"
    data_fold = "jai_results/cropped_projection_resnext50_simpler_transpose_equalize/test_single_projection_yolo_cropping/"
    xai_fold = results_fold + "cXAI/"

    cam_name_start = "results_Grad-CAM_feature_extractor_features_7_2_conv3_class"

    allow_interaction = gr.update(interactive=True)
    avoid_interaction = gr.update(interactive=False)
    empty_radio = gr.update(value=None)
    simple_update = gr.update()
    normal_scale_value = 1

    intro_msg = """## VertebrAI
                    #### Istruzioni:
                    - Dovrai <u>effettuare l'accesso scegliendo uno username</u> (ad esempio, il tuo cognome). Questo ti consentirà di interrompere il questionario e riprenderlo successivamente, se necessario.
                    - Ti verrà somministrato un breve <u>questionario di profilazione</u>.
                    - Visualizzerai i dati RX di 15 pazienti, che includono almeno due proiezioni per ognuno. 
                         * <u>Durante la valutazione potresti venire supportato dall'AI</u>: in ordine casuale valuterai (a) 5 casi senza supporto, (b) 5 casi con un supporto visivo (sistema visuale), (c) 5 casi con un supporto testuale (sistema testuale).
                         * <u>Per ciascun RX dovrai fornire UNA sola diagnosi</u> - elencando quali vertebre presentano frattura - e valutare la tua confidenza diagnostica, la complessità del caso e, solo per b e c, l'utilità del supporto.
                    - Dopo l'utilizzo di ciascun sistema, compilerai un breve <u>questionario per valutare la tua esperienza complessiva</u>.
                """
    results_file_name = "survey_results.csv"
    binary_answers = ["No", "Sì"]
    sex_levels = ["Maschio", "Femmina", "Non binario", "Altro", "Preferisco non rispondere"]
    career_levels = ["Non ho ancora iniziato", "I anno", "II anno", "III anno", "IV anno", "V anno", "Ho terminato la specializzazione"]
    likert_choices = ["Fortemente in disaccordo", "In disaccordo", "Né d'accordo né in disaccordo", "D'accordo",
                      "Fortemente d'accordo"]
    confidence_levels = ["Molto bassa", "Bassa", "Neutrale", "Alta", "Molto alta"]
    branches = ["Senza supporto AI", "Supporto visuale (mappe di calore)", "Support testuale"]

    def __init__(self, working_dir, desired_instances, debug_mode=False):
        # Define attributes
        self.working_dir = working_dir
        self.results_dir = working_dir + self.results_fold
        self.data_dir = self.results_dir + self.data_fold
        self.survey_dir = self.results_dir + self.survey_fold
        if self.survey_fold[:-1] not in os.listdir(self.results_dir):
            os.mkdir(self.survey_dir)

        self.n_instances = len(desired_instances)
        self.desired_instances = desired_instances.copy()
        random.shuffle(self.desired_instances)
        print("Presented RX:")
        for i, name in enumerate(self.desired_instances):
            print(" " + str(i + 1) + ") " + name)

        self.debug_mode = debug_mode

    def get_img(self, count=None, proj_idx=None, is_cam=False, is_txt=False, first_display=False):
        if first_display or (count is not None and (count < 0 or count >= self.n_instances)):
            img = np.zeros((1000, 500))
        else:
            if self.debug_mode:
                print("Getting image for item", count)
            base_path = self.data_dir + self.desired_instances[count] + str(proj_idx)
            img = cv2.imread(base_path + "/raw_img.png")

        if not is_cam and not is_txt:
            return img
        else:
            name0 = base_path + "/" + self.cam_name_start + str(0) + ".png"
            name1 = base_path + "/" + self.cam_name_start + str(1) + ".png"
            if is_cam:
                exp0 = cv2.imread(name0)
                exp1 = cv2.imread(name1)
            elif is_txt:
                exp0 = {}
                exp1 = {}
                for directory in os.listdir(self.data_dir):
                    with open(self.data_dir + directory + "model_predictions.txt", "r", encoding="utf-8") as f:
                        for line in f:
                            class_id, probability = line.strip().split(":", 1)
                            if class_id.strip() == "1":
                                prob_class_1 = float(probability.strip())
                                break
                    if prob_class_1 > 0.0:
                        with open(self.data_dir + directory + "text_explanations.txt", "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        for line_idx, line in enumerate(lines):
                            if line.strip() == name0 and line_idx + 1 < len(lines):
                                exp0.update({directory: lines[line_idx + 1].strip()})
                            if line.strip() == name1 and line_idx + 1 < len(lines):
                                exp1.update({directory: lines[line_idx + 1].strip()})

            return img, exp0, exp1

    def login(self, name):
        # Check for name correctness
        preference_msg = self.simple_update
        preference = self.simple_update
        preference_txt = self.simple_update
        second_conclude_flag = False

        for char in ["^", "~", "\"", "#", "%", "&", "*", ":", "<", ">", "?", "/", "\\", "{", "}", "|"]:
            if char in name:
                gr.Warning("WARNING! Your surname must not contain any of the following characters:\n"
                           "^ ~ \" # % & * : < > ? / \\ { } |")
                return (name, -10, self.simple_update, self.simple_update, self.simple_update, self.simple_update,
                        self.simple_update, self.simple_update, self.simple_update, self.simple_update,
                        self.simple_update, preference_msg, preference, preference_txt, second_conclude_flag)

        # Create folder
        if name not in os.listdir(self.survey_dir):
            os.mkdir(self.survey_dir + name)
        else:
            gr.Success("The username '" + name + "' is already present in our database. Your previous progress will be "
                       "loaded, and you will be able to continue the survey from where you left it.")

        # Check for survey result file
        user_folder = self.survey_dir + name + "/"
        if self.results_file_name in os.listdir(user_folder):
            results_file = pd.read_csv(user_folder + self.results_file_name)
            try:
                count = results_file["index"].iloc[-1] + 1
            except IndexError:
                count = 0
            try:
                branches_order = list(np.unique(results_file["group"]))
                if len(branches_order) == 0:
                    branches_order = None
                elif len(branches_order) == 1:
                    user_seed = hash(name) % 1_000_000
                    rng = random.Random(user_seed)
                    branches_order += rng.sample(list(set(self.branches) - set(branches_order)), k=2)
                else:
                    branches_order += list(set(self.branches) - set(branches_order))
            except IndexError:
                first_group = None
        else:
            results_file = pd.DataFrame(columns=["index", "instance", "annotator", "group", "diagnosis", "confidence",
                                                 "complexity", "usefulness", "time"])
            results_file.to_csv(user_folder + self.results_file_name, index=False)
            count = 0
            first_group = None
        if self.debug_mode:
            print("==============================")
            print("Initial count:", count)

        # Define user branch
        if branches_order is None:
            user_seed = hash(name) % 1_000_000
            rng = random.Random(user_seed)
            branches_order = rng.sample(self.branches, k=len(self.branches))

        if self.debug_mode:
            addon = " first" if self.matched_groups else ""
            print("User '" + name + "' is assigned to the following group order:", branches_order)

        tab1 = self.avoid_interaction
        tab2 = self.simple_update
        tab3 = self.simple_update
        tab4 = self.simple_update
        tab5 = self.simple_update
        tab6 = self.simple_update
        starting_flag = False
        performance_flag = False
        if "final_questionnaire.csv" in os.listdir(user_folder) and pd.read_csv(user_folder + "final_questionnaire.csv").shape[0] == 3:
            # Finish
            tab9 = self.allow_interaction
            selected = 9
        elif count == self.n_instances:
            # Go to final questionnaire C
            tab8 = self.allow_interaction
            selected = 8
            performance_flag = True
        elif "final_questionnaire.csv" in os.listdir(user_folder) and pd.read_csv(user_folder + "final_questionnaire.csv").shape[0] == 2:
            # Go to branch C
            tab7 = self.allow_interaction
            selected = 7
            starting_flag = True
        elif count == 2 * self.n_instances // 3:
            # Go to final questionnaire B
            tab6 = self.allow_interaction
            selected = 6
        elif "final_questionnaire.csv" in os.listdir(user_folder) and pd.read_csv(user_folder + "final_questionnaire.csv").shape[0] == 1:
            # Go to branch B
            tab5 = self.allow_interaction
            selected = 5
            starting_flag = True
        elif count == self.n_instances // 3:
            # Go to final questionnaire A
            tab4 = self.allow_interaction
            selected = 4
            starting_flag = True
        elif "preliminary_questionnaire.csv" in os.listdir(user_folder):
            # Go to branch A
            tab3 = self.allow_interaction
            selected = 3
            starting_flag = True
        else:
            # Go to preliminary questionnaire
            tab2 = self.allow_interaction
            selected = 2
        tabs = gr.update(selected=selected)

        if performance_flag:
            preference = gr.update(choices=branches_order + ["Non ho preferenze"],
                                   label="Quale modelità di support hai preferito? Ricorda che hai lavorato in ordine "
                                         "con: " + branches_order[0] + ", " + branches_order[1] + ", " + branches_order[2]
                                         + ".", visible=True)
            preference_msg = gr.update(visible=True)
            preference_txt = gr.update(visible=True)
            second_conclude_flag = True

        if self.debug_mode:
            print("Count at 'login' end:", count)
            print("------------------------------")
        return (name, count, tabs, tab1, tab2, tab3, tab4, tab5, tab6, branches_order, starting_flag, preference_msg, preference,
                preference_txt, second_conclude_flag)

    def next(self, name, count, img_display, diagnosis, confidence, complexity, preliminary_flag=False):
        if count == -10 or count > self.n_instances:
            if count > self.n_instances:
                count += 1
            return (count, self.simple_update, self.simple_update, self.simple_update, self.simple_update,
                    self.simple_update, self.simple_update, self.simple_update)

        # Save diagnosis
        user_folder = self.survey_dir + name + "/"
        file_path = user_folder + self.results_file_name
        results_file = pd.read_csv(file_path)
        if not preliminary_flag:
            # Store results
            if np.mean(img_display) != 0.0 and (diagnosis is None or confidence is None or complexity is None):
                gr.Warning("WARNING! You have not fully evaluated the displayed ECG. Please complete all fields before proceeding.")
                diag = self.classes[diagnosis] if diagnosis is not None else None
                conf = self.confidence_levels[confidence] if confidence is not None else None
                comp = self.confidence_levels[complexity] if complexity is not None else None
                return (count, gr.update(), gr.update(value=diag), gr.update(value=conf), gr.update(value=comp),
                        gr.update(), gr.update(), gr.update())
            if self.debug_mode:
                print("Storing image results for image", count-1)
            new_row = {"index": count-1, "instance": self.desired_instances[count-1], "annotator": name,
                       "group": "preliminary", "diagnosis": int(diagnosis), "confidence": int(confidence),
                       "complexity": int(complexity), "usefulness": np.nan, "time": datetime.datetime.now()}
            results_file = pd.concat([results_file, pd.DataFrame([new_row])], ignore_index=True)
            results_file.to_csv(file_path, index=False)

        tab2 = self.simple_update
        tab3 = self.simple_update
        tabs = self.simple_update
        if count == self.n_instances and not preliminary_flag:
            img = None
            tab2 = self.avoid_interaction
            tab3 = self.allow_interaction
            tabs = gr.update(selected=3)
        else:
            img = self.get_signal(count=count)
        count += 1

        if img is not None:
            img_display = gr.update(value=img)
        else:
            img_display = self.simple_update
        return count, img_display, gr.update(value=None), gr.update(value=None), gr.update(value=None), tabs, tab2, tab3

    def next_ai(self, name, count, ai_display, diagnosis, confidence, usefulness, is_cxai, preliminary_flag=False,
                starting_flag=False):
        if count == -10:
            return (count, self.simple_update, self.simple_update, self.simple_update, self.simple_update,
                    self.simple_update, self.simple_update, self.simple_update, self.simple_update, self.simple_update,
                    self.simple_update, self.simple_update)

        # Eventually swap groups for second half of AI-aided cases and rearrange data
        group = "cXAI" if is_cxai else "JAI"
        desired_instances_ai = self.desired_instances_ai.copy()
        if self.matched_groups:
            if count > self.n_instances + self.n_instances // 2:
                is_cxai = not is_cxai
                if count > self.n_instances + self.n_instances // 2 + 1:
                    group = "cXAI" if is_cxai else "JAI"

            user_seed = hash(name) % 1_000_000
            rng = random.Random(user_seed)
            rng.shuffle(desired_instances_ai)

        # Save diagnosis
        user_folder = self.survey_dir + name + "/"
        file_path = user_folder + self.results_file_name
        results_file = pd.read_csv(file_path)
        idx = count - self.n_instances - 1
        if not preliminary_flag:
            # Store results
            if np.mean(ai_display) != 0.0 and (diagnosis is None or confidence is None or usefulness is None):
                gr.Warning("WARNING! You have not fully evaluated the displayed ECG. Please complete all fields before proceeding.")
                diag = self.classes[diagnosis] if diagnosis is not None else None
                conf = self.confidence_levels[confidence] if confidence is not None else None
                useful = self.confidence_levels[usefulness] if usefulness is not None else None
                return (count, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=diag),
                        gr.update(value=conf), gr.update(value=useful), gr.update(), gr.update(), gr.update())
            if self.debug_mode:
                print("Storing results for explanation(s)", idx-1, "at count", count-2)
            new_row = {"index": count-2, "instance": desired_instances_ai[idx-1], "annotator": name,
                       "group": group, "diagnosis": int(diagnosis), "confidence": int(confidence),
                       "complexity": np.nan, "usefulness": int(usefulness), "time": datetime.datetime.now()}
            if new_row["index"] is not None:
                results_file = pd.concat([results_file, pd.DataFrame([new_row])], ignore_index=True)
                results_file.to_csv(file_path, index=False)

        descr_c = self.simple_update
        descr_j = self.simple_update
        tabs = self.simple_update
        tab4 = self.simple_update
        tab5 = self.simple_update
        if not preliminary_flag and count == 2 * self.n_instances + 1:
            xai = None
            tab4 = self.avoid_interaction
            tab5 = self.allow_interaction
            tabs = gr.update(selected=5)
            ai_display_down = self.simple_update
        else:
            if self.matched_groups and not preliminary_flag and count == self.n_instances + self.n_instances // 2 + 1:
                tab4 = self.avoid_interaction
                tab5 = self.allow_interaction
                tabs = gr.update(selected=5)

            if is_cxai:
                descr_j = gr.update(visible=False)
                descr_c = gr.update(visible=True)
                xai = self.get_xai(count=idx, name=name)
                ai_display_down = gr.update(visible=False)
            else:
                descr_j = gr.update(visible=True)
                descr_c = gr.update(visible=False)
                xai, xai_d = self.get_xai(idx, is_cxai=False)
                ai_display_down = gr.update(visible=True, value=xai_d)
        if not preliminary_flag or starting_flag:
            count += 1

        if xai is not None:
            ai_display = gr.update(value=xai)
        else:
            ai_display = self.simple_update
        return (count, descr_c, descr_j, ai_display, ai_display_down, gr.update(value=None), gr.update(value=None),
                gr.update(value=None), tabs, tab4, tab5)

    def start(self, name, count, age, sex, country, hospital, career, expertise, q1, q2, q3, q4, q5):
        # Store preliminary questionnaire results
        user_folder = self.survey_dir + name + "/"
        titles = ["annotator", "age", "sex", "country", "hospital", "career", "expertise", "knowledge", "worked",
                  "performance", "productivity", "effectiveness", "time"]
        if (age == 0 or sex is None or expertise is None or not country or not hospital or q1 is None or q2 is None or
                q3 is None or q4 is None or q5 is None):
            gr.Warning("WARNING! Please complete all fields in the questionnaire before proceeding.")
            q1i = self.binary_answers[q1] if q1 is not None else None
            q2i = self.binary_answers[q2] if q2 is not None else None
            q3i = self.binary_answers[q3] if q3 is not None else None
            q4i = self.binary_answers[q4] if q4 is not None else None
            q5i = self.binary_answers[q5] if q5 is not None else None
            return (count-1, gr.update(), gr.update(), gr.update(), gr.update(value=age), gr.update(value=sex),
                    gr.update(value=country), gr.update(value=hospital), gr.update(value=career),
                    gr.update(value=expertise), gr.update(value=q1i), gr.update(value=q2i), gr.update(value=q3i),
                    gr.update(value=q4i), gr.update(value=q5i), False)
        values = [name, age, sex, country, hospital, career, expertise, q1, q2, q3, q4, q5, datetime.datetime.now()]
        df = pd.DataFrame([values], columns=titles)
        df.to_csv(user_folder + "preliminary_questionnaire.csv", index=False)

        tabs = gr.update(selected=4)
        tab3 = gr.update(interactive=False)
        tab4 = gr.update(interactive=True)
        if self.debug_mode:
            print("------------------------------")
            print("Count at 'start' end:", count)
            print("------------------------------")
        return (count, tabs, tab3, tab4, gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), True)

    def conclude(self, name, q11, q12, q13, q14, q21, q22, q23, q24, q31, q32, q33, q41, q42, q43, q44, q51, is_cxai,
                 second_conclude_flag, preference, preference_txt):
        # Store final questionnaire results
        user_folder = self.survey_dir + name + "/"
        if self.matched_groups:
            titles = ["annotator", "competence1", "competence4", "autonomy2", "autonomy3", "relatedness1", "relatedness2",
                      "enjoyment1", "demand1", "group", "preference", "preference_comment", "time"]
            if second_conclude_flag:
                is_cxai = not is_cxai
            group = "cXAI" if is_cxai else "JAI"
            values = [name, q11, q14, q22, q23, q31, q32, q41, q51, group, preference, preference_txt, datetime.datetime.now()]
        else:
            titles = ["annotator", "competence1", "competence2", "competence3", "competence4", "autonomy1", "autonomy2",
                      "autonomy3", "autonomy4", "relatedness1", "relatedness2", "relatedness3", "enjoyment1",
                      "enjoyment2",
                      "enjoyment3", "enjoyment4", "demand1", "time"]
            values = [name, q11, q12, q13, q14, q21, q22, q23, q24, q31, q32, q33, q41, q42, q43, q44, q51,
                      datetime.datetime.now()]

        preference_msg = self.simple_update
        preference_txt = self.simple_update
        if not self.matched_groups:
            if (q11 is None or q12 is None or q13 is None or q14 is None or q21 is None or q22 is None or q23 is None
                    or q24 is None or q31 is None or q32 is None or q33 is None or q41 is None or q42 is None or q43 is None
                    or q44 is None or q51 is None):
                gr.Warning("WARNING! Please complete all fields in the questionnaire before proceeding.")
                return (gr.update(), gr.update(), gr.update(), gr.update(), second_conclude_flag, None, None, None, None,
                        None, None, None, None, preference_msg, self.simple_update, preference_txt)
        else:
            if (q11 is None or q14 is None or q22 is None or q23 is None or q31 is None or q32 is None or q41 is None
                    or q51 is None or second_conclude_flag and preference is None):
                gr.Warning("WARNING! Please complete all fields in the questionnaire before proceeding.")
                return (gr.update(), gr.update(), gr.update(), gr.update(), second_conclude_flag, None, None, None, None,
                        None, None, None, None, preference_msg, self.simple_update, preference_txt)
        preference = self.simple_update
        df = pd.DataFrame([values], columns=titles)
        if second_conclude_flag:
            df_old = pd.read_csv(user_folder + "final_questionnaire.csv")
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(user_folder + "final_questionnaire.csv", index=False)

        tab4 = self.simple_update
        tab5 = gr.update(interactive=False)
        tab6 = self.simple_update
        if not self.matched_groups or second_conclude_flag:
            tabs = gr.update(selected=6)
            tab6 = gr.update(interactive=True)
        else:
            tabs = gr.update(selected=4)
            tab4 = gr.update(interactive=True)

            if is_cxai:
                old_group = self.groups[0]
                new_group = self.groups[1]
                choices = self.groups
            else:
                old_group = self.groups[1]
                new_group = self.groups[0]
                choices = [self.groups[1], self.groups[0], self.groups[2]]
            preference = gr.update(choices=choices, label="Which decision support system do you prefer overall? "
                                                          "Remember you were first shown with the " + old_group +
                                                          " system and then with the " + new_group + " one.",
                                   visible=True)
            preference_msg = gr.update(visible=True)
            preference_txt = gr.update(visible=True)

        second_conclude_flag = not second_conclude_flag
        return (tabs, tab4, tab5, tab6, second_conclude_flag, gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value=None), gr.update(value=None), preference_msg, preference, preference_txt)

    def display_tabs(self, block):
        count = gr.State({"count": 0, "preliminary_flag": True, "starting_flag": True, "branches_groups": None,
                          "second_conclude_flag": False})

        with gr.Tabs(selected=1) as tabs:
            with gr.Tab(id=1, label="Autenticazione", interactive=True) as tab1:
                gr.Markdown("### Accedi con il tuo cognome...")
                name = gr.Textbox(placeholder="Inserisci qui il tuo cognome...", show_label=False, max_lines=1)
                login_btn = gr.Button(value="Accedi", icon="next.png")

            with gr.Tab(id=3, label="Questionario di Profilazione", interactive=False) as tab2:
                gr.Markdown("### Compila il seguente questionario...")
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
                        expertise = gr.Number(label="Approssimativamente, quante diagnosi di fratture vertebrali con immagini RX hai effettato nella tua cariera?",
                                              step=1, precision=0)
                    gr.Markdown("#### Familiarità con l'AI")
                    with gr.Row():
                        q1 = gr.Radio(choices=self.binary_answers, label="Ho una buona conoscenza sull'AI.", type="index")
                    with gr.Row():
                        q2 = gr.Radio(choices=self.binary_answers, label="Ho lavorato e/o utilizzato sistemi basati sull'AI nel mio lavoro", type="index")
                    gr.Markdown("#### Fiducia nell'AI")
                    with gr.Row():
                        q3 = gr.Radio(choices=self.binary_answers, label="Credo che l'AI possa aiutarmi a rispondere più"
                                                                         " correttamente e velocemente a domande di cui "
                                                                         "non conosco la risposta con precisione.", type="index")
                    with gr.Row():
                        q4 = gr.Radio(choices=self.binary_answers, label="Credo che usare l'AI per aiutarmi nel mio "
                                                                         "lavoro o studio possa aumentare la mia "
                                                                         "produttività", type="index")
                    with gr.Row():
                        q5 = gr.Radio(choices=self.binary_answers, label="Credo che con l'aiuto dell'AI I believe io "
                                                                         "possa migliorare l'efficacia del mio lavoro.",
                                      type="index")
                start_btn = gr.Button(value="Inizia", icon="next.png")

            with gr.Tab(id=2, label="Esercizio di Diagnosi", interactive=False) as tab2:
                gr.Markdown("### Suggest a diagnosis for the following cases WITHOUT the help of AI.")
                if self.filter_signals:
                    gr.Markdown("You may use your laptop's touchpad to <u>zoom in and out</u> or you can use the "
                                "<u>download button</u> (top right corner) to open the image on your device")
                with gr.Row():
                    img = self.get_signal(first_display=True)
                    img_display = gr.Image(value=img, image_mode="L", interactive=False, buttons=["download"],
                                           label="Input ECG Data")
                with gr.Row():
                    with gr.Column(min_width=500):
                        with gr.Row():
                            diagnosis = gr.Radio(choices=self.classes, label="What is the diagnosis?",
                                                 type="index")
                        with gr.Row():
                            confidence = gr.Radio(choices=self.confidence_levels, label="What is your level of confidence in "
                                                                                        "suggesting this diagnosis?", type="index")
                        with gr.Row():
                            complexity = gr.Radio(choices=self.confidence_levels, label="What is, in your opinion, the complexity "
                                                                                        "of this case?", type="index")
                next_btn = gr.Button(value="Next", icon="next.png")



            with gr.Tab(id=4, label="AI-aided diagnostic exercise", interactive=False) as tab4:
                gr.Markdown("### Suggest a diagnosis for the following cases WITH the help of AI.")
                if self.filter_signals:
                    gr.Markdown("You may use your laptop's touchpad to <u>zoom in and out</u> or you can use the "
                                "<u>download button</u> (top right corner) to open the image on your device")
                with gr.Row():
                    with gr.Column(min_width=500):
                        if not self.filter_signals:
                            descr_c = gr.Markdown("#### Contrastive explanation\n"
                                                  "You are presented with AI-derived evidence answering the question <i>Why P, rather than Q?</i>, where:\n"
                                                  "- **P**: the class predicted by the AI model\n"
                                                  "- **Q**: the class you suggested in the preliminary evaluation, or, if it matches the AI prediction, the second most likely class\n"
                                                  "Evidence is shown as a **temporal heatmap**, highlighting the most relevant time points of the input (bright yellow) that support the proposed question.\n\n"
                                                  "**e.g.**, a heatmap titled <i>Why 'Hypertension' rather than 'Myocardial Infarction'?</i>  will show lighter tones on the ECG at certain time points. These highlighted points represent the passages the model considers most important to differentiate Hypertension from Myocardial Infarction across the leads.</small>")
                            descr_j = gr.Markdown("#### Judicial explanation\n"
                                                  "You are presented with **two pieces of AI-derived evidence**, each supporting one of the two most likely diagnoses according to the AI model.\n"
                                                  "The order in which the two heatmaps appear is random, so you are not informed which class corresponds to the model's prediction.\n"
                                                  "Each piece of evidence is shown as a **temporal heatmap**, highlighting the most relevant time points of the input (bright yellow) that support the class indicated in the title.\n\n"
                                                  "**e.g.**, a heatmap titled <i>Why 'Hypertension'?</i>  will show lighter tones on the ECG at certain time points. These points represent passages the model considers most important to diagnose Hypertension across the leads. It will be paired with another heatmap, such as <i>Why 'Myocardial Infarction'?</i>, similarly highlighting the time steps the model considers important for that diagnosis.")
                        else:
                            descr_c = gr.Markdown("### Contrastive explanation\n"
                                                  "You are presented with AI-derived evidence answering the question \"Why P, rather than Q?\", where:\n"
                                                  "- P: the class predicted by the AI model\n"
                                                  "- Q: the class you suggested in the preliminary evaluation, or, if it matches the AI prediction, the second most likely class\n"
                                                  "Evidence is shown as a temporal heatmap, highlighting the most relevant time points of the input (bright yellow) that support the proposed question.\n\n"
                                                  "e.g., a heatmap titled \"Why 'Hypertension' rather than 'Myocardial Infarction'?\"  will show lighter tones on the ECG at certain time points. These highlighted points represent the passages the model considers most important to differentiate Hypertension from Myocardial Infarction across the leads.</small> \n\n"
                                                  "## SUMMARY: \"Why P, rather than Q?\" - P is the model prediction for this ECG, Q is your initial diagnosis or, if your initial diagnosis matches P, the model's second most likely prediction.")
                            descr_j = gr.Markdown("### Judicial explanation\n"
                                                  "You are presented with two pieces of AI-derived evidence, each supporting one of the two most likely diagnoses according to the AI model.\n"
                                                  "The order in which the two heatmaps appear is random, so you are not informed which class corresponds to the model's prediction.\n"
                                                  "Each piece of evidence is shown as a temporal heatmap, highlighting the most relevant time points of the input (bright yellow) that support the class indicated in the title.\n\n"
                                                  "e.g., a heatmap titled \"Why 'Hypertension'?\"  will show lighter tones on the ECG at certain time points. These points represent passages the model considers most important to diagnose Hypertension across the leads. It will be paired with another heatmap, such as \"Why 'Myocardial Infarction'?\", similarly highlighting the time steps the model considers important for that diagnosis. \n\n"
                                                  "## SUMMARY: \"Why P?\" and \"Why Q?\" - P and Q are the model's two most likely predictions for this ECG, presented in random order")
                with gr.Row():
                    with gr.Column(min_width=500):
                        xai_img = self.get_signal(first_display=True)
                        ai_display = gr.Image(value=xai_img, image_mode="L", interactive=False, buttons=["download"],
                                              label="AI-Derived Evidence")
                        ai_display_down = gr.Image(value=xai_img, image_mode="L", interactive=False,
                                                   buttons=["download"], label="AI-Derived Evidence")
                with gr.Row():
                    with gr.Column(min_width=500):
                        with gr.Row():
                            diagnosis_ai = gr.Radio(choices=self.classes, label="What is the diagnosis?", type="index")
                        with gr.Row():
                            confidence_ai = gr.Radio(choices=self.confidence_levels,
                                                     label="What is your level of confidence in suggesting this diagnosis?",
                                                     type="index")
                        with gr.Row():
                            usefulness_ai = gr.Radio(choices=self.confidence_levels,
                                                     label="Do you consider the AI support useful for this example, or "
                                                           "did the evidence appear random? Please rate the usefulness of the heatmap.",
                                                     type="index")
                next_btn_ai = gr.Button(value="Next", icon="next.png")

            with gr.Tab(id=5, label="Final feedback", interactive=False) as tab5:
                gr.HTML(self.js)
                gr.Markdown("### Fill up the following questionnaire commenting on your experience.")
                if not self.filter_signals:
                    gr.Markdown("#### Please indicate your level of agreement with each of the following statements. In "
                                "doing so, refer to the AI-aided diagnosis exercise completed in the previous "
                                "section, not to the preliminary un-aided one.")
                if self.matched_groups:
                    if not self.filter_signals:
                        gr.Markdown("### Since this study involves two decision support systems, please refer to the one you"
                                    " were just shown when considering the previous " + str(self.n_instances // 2) +
                                    " signals.")
                    else:
                        gr.Markdown("Since <u>this study involves two decision support systems, please refer to the one you"
                                    " were just shown</u> when considering the previous " + str(self.n_instances // 2) +
                                    " signals.")

                with gr.Column(min_width=500):
                    preference_msg = gr.Markdown("#### <br>System Preference", visible=False)
                    with gr.Row():
                        preference = gr.Radio(choices=self.groups, label="", visible=False)
                    with gr.Row():
                        preference_txt = gr.Textbox(label="Please briefly explain your choice (optional)",
                                                    placeholder="E.g., clearer explanations, more clinically relevant "
                                                                "evidence, easier to interpret...", visible=False,
                                                    lines=3)
                    gr.Markdown("#### Perceived Competence")
                    with gr.Row():
                        q11 = gr.Radio(choices=self.likert_choices, label="I think I performed well in making diagnoses during "
                                                                          "this task.", type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q12 = gr.Radio(choices=self.likert_choices, label="I felt that I did not perform very well in this "
                                                                          "task.", type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q13 = gr.Radio(choices=self.likert_choices, label="I believe I am skilled at suggesting suitable "
                                                                          "diagnoses.", type="index")
                    with gr.Row():
                        q14 = gr.Radio(choices=self.likert_choices, label="After working at this task for a while, I felt "
                                                                          "pretty competent.", type="index")
                    gr.Markdown("#### Perceived Autonomy")
                    with gr.Row(visible=not self.matched_groups):
                        q21 = gr.Radio(choices=self.likert_choices, label="I felt like I had a lot of choice in selecting the "
                                                                          "diagnosis.", type="index")
                    with gr.Row():
                        q22 = gr.Radio(choices=self.likert_choices, label="I was free to choose the diagnosis I thought was "
                                                                          "best suited (between the two possible ones) for"
                                                                          " the ECGs shown.", type="index")
                    with gr.Row():
                        q23 = gr.Radio(choices=self.likert_choices, label="I felt strongly influenced by the AI in how I "
                                                                          "recommended diagnoses.", type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q24 = gr.Radio(choices=self.likert_choices, label="I recommended diagnoses in the way I wanted to.",
                                       type="index")
                    gr.Markdown("#### Relatedness to AI")
                    with gr.Row():
                        q31 = gr.Radio(choices=self.likert_choices, label="I felt I could trust this AI (meaning, the "
                                                                          "AI model and decision support provided).",
                                       type="index")
                    with gr.Row():
                        q32 = gr.Radio(choices=self.likert_choices, label="I felt my reasoning on this task was distant from "
                                                                          "this AI’s.", type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q33 = gr.Radio(choices=self.likert_choices, label="I would like a chance to interact with this AI in "
                                                                          "the future.", type="index")
                    gr.Markdown("#### Interest/Enjoyment")
                    with gr.Row():
                        q41 = gr.Radio(choices=self.likert_choices, label="I enjoyed this diagnosis task.", type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q42 = gr.Radio(choices=self.likert_choices, label="This task did not hold my attention at all.",
                                       type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q43 = gr.Radio(choices=self.likert_choices, label="While I was doing this task, I was thinking "
                                                                          "about how much I enjoyed it.", type="index")
                    with gr.Row(visible=not self.matched_groups):
                        q44 = gr.Radio(choices=self.likert_choices, label="I thought this diagnosis task boring.",
                                       type="index")
                    gr.Markdown("#### Mental Demand")
                    with gr.Row():
                        q51 = gr.Radio(choices=self.likert_choices, label="I found this task mentally demanding.",
                                       type="index")
                conclude = gr.Button(value="Conclude", icon="next.png")

            with gr.Tab(id=6, label="Conclusion", interactive=False) as tab6:
                gr.Markdown("# Thank you for your contribution! The survey is now complete, and you may close this "
                            "page.")

        (login_btn.click(fn=self.login, inputs=[name], outputs=[name, count, tabs, tab1, tab2, tab3, tab4, tab5, tab6,
                                                                is_cxai, starting_flag, preference_msg, preference,
                                                                preference_txt, second_conclude_flag],
                         concurrency_id="login", concurrency_limit=1
                         ).then(fn=self.next, inputs=[name, count, img_display, diagnosis, confidence, complexity,
                                                      preliminary_flag],
                                outputs=[count, img_display, diagnosis, confidence, complexity, tabs, tab2, tab3],
                                concurrency_id="login", concurrency_limit=1
                                ).then(fn=self.next_ai, inputs=[name, count, ai_display, diagnosis_ai, confidence_ai,
                                                                usefulness_ai, is_cxai, preliminary_flag, starting_flag],
                                       outputs=[count, descr_c, descr_j, ai_display, ai_display_down, diagnosis_ai,
                                                confidence_ai, usefulness_ai, tabs, tab4, tab5], concurrency_id="login",
                                       concurrency_limit=1))
        next_btn.click(fn=self.next, inputs=[name, count, img_display, diagnosis, confidence, complexity],
                       outputs=[count, img_display, diagnosis, confidence, complexity, tabs, tab2, tab3],
                       concurrency_id="next_btn", concurrency_limit=1)
        start_btn.click(fn=self.start, inputs=[name, count, age, sex, country, hospital, career, expertise, q1, q2, q3,
                                               q4, q5],
                        outputs=[count, tabs, tab3, tab4, age, sex, country, hospital, career, expertise, q1, q2, q3, q4,
                                 q5, starting_flag],
                        concurrency_id="start", concurrency_limit=1
                        ).then(fn=self.next_ai, inputs=[name, count, ai_display, diagnosis_ai, confidence_ai,
                                                        usefulness_ai, is_cxai, preliminary_flag, starting_flag],
                               outputs=[count, descr_c, descr_j, ai_display, ai_display_down, diagnosis_ai,
                                        confidence_ai, usefulness_ai, tabs, tab4, tab5], concurrency_id="start",
                               concurrency_limit=1)
        next_btn_ai.click(fn=self.next_ai, inputs=[name, count, ai_display, diagnosis_ai, confidence_ai, usefulness_ai,
                                                   is_cxai],
                          outputs=[count, descr_c, descr_j, ai_display, ai_display_down, diagnosis_ai,
                                   confidence_ai, usefulness_ai, tabs, tab4, tab5],
                          concurrency_id="next_btn_ai", concurrency_limit=1)
        conclude.click(fn=self.conclude, concurrency_id="conclude", concurrency_limit=1,
                       inputs=[name, q11, q12, q13, q14, q21, q22, q23, q24, q31, q32, q33, q41, q42, q43, q44, q51,
                               is_cxai, second_conclude_flag, preference, preference_txt],
                       outputs=[tabs, tab4, tab5, tab6, second_conclude_flag, q11, q14, q22, q23, q31, q32, q41, q51,
                                preference_msg, preference, preference_txt])

    def build_app(self, share=False):
        # Set up the application
        with gr.Blocks(gr.themes.Soft()) as block:
            gr.Markdown(self.intro_msg)
            self.display_tabs(block)

        # Launch the application
        block.launch(share=share)


# Main
if __name__ == "__main__":
    # Set seed
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    desired_instances1 = ["308c", "370s", "093l", "354d"]
    debug_mode1 = False
    share1 = True

    # Launch app
    survey = SurveyCreator(working_dir=working_dir1, desired_instances=desired_instances1, debug_mode=debug_mode1)
    print("Add '?__theme=dark' at the end of the link")
    survey.build_app(share=share1)
