# Import packages
import os
import math

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.exceptions import TrialPruned
import numpy as np
import pandas as pd
import json
from datetime import datetime

from TrainUtils.OptunaParamFinder import OptunaParamFinder
from DataUtils.XrayDataset import XrayDataset
from TrainUtils.ViTMAETrainer import ViTMAETrainer
from Enumerators.NetType import NetType
from Enumerators.ProjectionType import ProjectionType


# Class
class PretrainOptunaParamFinder(OptunaParamFinder):

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, epochs, val_epochs, use_cuda,
                 n_trials, projection_dataset=False, double_output=False, search_for_untracked_models=False,
                 preprocess_inputs=False, enhance_images=False):
        super().__init__(model_name, working_dir, train_data, val_data, test_data, net_type=None, epochs=epochs,
                         val_epochs=val_epochs, use_cuda=use_cuda, n_trials=n_trials, s3=None, n_parallel_gpu=0,
                         projection_dataset=projection_dataset, output_metric="loss", double_output=double_output,
                         search_for_untracked_models=search_for_untracked_models, preprocess_inputs=preprocess_inputs,
                         enhance_images=enhance_images, full_size=False, direction="minimize", is_pretrain=True)

    def retrieve_untracked_models(self, distributions):
        if self.search_for_untracked_models:
            models_present = [name.strip(".pt") for name in os.listdir(self.results_dir) if ".pt" in name]
            if len(models_present) > 0:
                max_model_id = np.max([int(name[5:]) for name in models_present])
                n_untracked_models = max_model_id - self.counter
                if n_untracked_models > -1:
                    for i in range(n_untracked_models):
                        print()
                        trainer = ViTMAETrainer.load_model(working_dir=self.working_dir, model_name=self.model_name,
                                                           trial_n=self.counter, use_cuda=self.use_cuda,
                                                           train_data=self.train_data, val_data=self.val_data,
                                                           test_data=self.test_data,
                                                           projection_dataset=self.projection_dataset)

                        train_loss, val_loss = trainer.summarize_performance_pretrain(show_test=False, show_process=False,
                                                                                      show_cm=False, trial_n=self.counter)

                        row = {"number": self.counter,
                               "datetime_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                               "datetime_complete": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}
                        if not self.double_output:
                            row.update({"value": val_loss})
                        else:
                            row.update({"values_0": val_loss, "values_1": train_loss})
                        params = trainer.net_params
                        params.update({"batch_size": int(math.log2(params["batch_size"])),
                                       "base_lr": int(-math.log10(params["base_lr"])),
                                       "weight_decay": int(-math.log10(params["weight_decay"]))})
                        self.insert_trial(row, params, distributions)
                        print("Untracked model", self.counter, "inserted!")
                        self.counter += 1

    def objective(self, trial):
        # Store previous results
        if self.counter >= 2:
            self.analyze_study(show_best=False)

        # Sample parameters
        params = {
            "base_lr": np.round(10 ** (-1 * trial.suggest_int("base_lr", 2, 5, step=1)), decimals=5),
            "beta1": np.round(trial.suggest_float("beta1", 0.8, 0.95, step=0.05), decimals=2),
            "beta2": np.round(trial.suggest_float("beta2", 0.9, 0.999, step=0.011), decimals=3),
            "weight_decay": np.round(10 ** (-1 * trial.suggest_int("weight_decay", 1, 5, step=1)), decimals=5),
            "layer_decay": np.round(trial.suggest_float("layer_decay", 0, 0.75, step=0.75), decimals=2),
            "eps": 1e-9,
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "reduce_lr_on_plateau"]),
            "min_lr": 1e-8,
            "batch_size": int(2 ** (trial.suggest_int("batch_size", 6, 7, step=1))),
        }

        # Define seeds
        ViTMAETrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        print("Trial ID:", self.counter)
        print("Parameters:", params)
        self.counter += 1
        try:
            trainer = ViTMAETrainer(model_name=self.model_name, working_dir=self.working_dir, train_data=self.train_data,
                                    val_data=self.val_data, test_data=self.test_data, decoder_net_type=None,
                                    epochs=self.epochs, val_epochs=self.val_epochs, decoder_net_params=None,
                                    use_cuda=self.use_cuda, projection_dataset=self.projection_dataset,
                                    preprocess_inputs=self.preprocess_inputs, enhance_images=self.enhance_images,
                                    train_parameters=params)
            val_loss = trainer.pretrain(show_epochs=False, trial_n=self.counter-1, trial=trial,
                                        double_output=self.double_output, continue_training=False)
            print("Value:", val_loss)
            if self.double_output:
                val_loss, train_loss = val_loss

            self.current_trainer = trainer

        except TrialPruned:
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            val_loss = 1e10
            train_loss = 1e10

        if not self.double_output:
            return val_loss
        else:
            return val_loss, train_loss


if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "vitmae_extended_dataset_optuna"
    selected_segments1 = None
    selected_projection1 = None
    epochs1 = 300
    val_epochs1 = 10
    use_cuda1 = True
    projection_dataset1 = True
    preprocess_inputs1 = False
    enhance_images1 = False

    # Load data
    '''train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training",
                                           selected_segments=selected_segments1,
                                           selected_projection=selected_projection1)'''
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="extended_xray_dataset_training",
                                           selected_segments=selected_segments1, selected_projection=selected_projection1,
                                           correct_mistakes=False)
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation",
                                         selected_segments=selected_segments1, selected_projection=selected_projection1)
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test",
                                          selected_segments=selected_segments1,
                                          selected_projection=selected_projection1)

    # Define Optuna model
    n_trials1 = 10
    double_output1 = False
    search_for_untracked_models1 = False
    optuna1 = PretrainOptunaParamFinder(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                                        val_data=val_data1, test_data=test_data1, epochs=epochs1,
                                        val_epochs=val_epochs1, use_cuda=use_cuda1, n_trials=n_trials1,
                                        projection_dataset=projection_dataset1, double_output=double_output1,
                                        search_for_untracked_models=search_for_untracked_models1,
                                        preprocess_inputs=preprocess_inputs1, enhance_images=enhance_images1)
    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
