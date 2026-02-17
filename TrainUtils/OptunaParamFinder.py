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

from DataUtils.XrayDataset import XrayDataset
from TrainUtils.NetworkTrainer import NetworkTrainer
from Enumerators.NetType import NetType
from Enumerators.ProjectionType import ProjectionType


# Class
class OptunaParamFinder:

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, net_type, epochs, val_epochs, use_cuda,
                 n_trials, s3=None, n_parallel_gpu=0, projection_dataset=False, output_metric="f1", double_output=False,
                 search_for_untracked_models=False, preprocess_inputs=True, enhance_images=True, full_size=True, direction="maximize",
                 is_pretrain=False, is_cropped=False, weight_loss=False):
        self.model_name = model_name
        self.working_dir = working_dir
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.net_type = net_type
        self.epochs = epochs
        self.val_epochs = val_epochs
        self.use_cuda = use_cuda
        self.n_parallel_gpu = n_parallel_gpu
        self.projection_dataset = projection_dataset
        self.preprocess_inputs = preprocess_inputs
        self.enhance_images = enhance_images
        self.full_size = full_size
        self.weight_loss = weight_loss

        self.output_metric = output_metric
        self.double_output = double_output

        self.best_trials = []
        self.current_trainer = None

        self.results_dir = working_dir + XrayDataset.results_fold + XrayDataset.models_fold
        if s3 is None:
            if model_name not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + model_name)
        else:
            if not s3.exists(self.results_dir + model_name):
                s3.touch(self.results_dir + model_name + "/empty.txt")
        self.results_dir += model_name + "/"
        self.s3 = s3

        self.n_trials = n_trials
        if not self.double_output:
            self.study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(),
                                             pruner=optuna.pruners.MedianPruner())
        else:
            self.study = optuna.create_study(directions=[direction, direction], sampler=optuna.samplers.TPESampler(),
                                             pruner=optuna.pruners.MedianPruner())
        self.counter = 0

        # Retrieve previous results
        self.search_for_untracked_models = search_for_untracked_models
        self.is_pretrain = is_pretrain
        self.retrieve_previous_results()

        self.is_cropped = is_cropped

    def retrieve_previous_results(self):
        addon = "" if not self.is_pretrain else "pretrain_"
        file_list = os.listdir(self.results_dir) if self.s3 is None else [x.split("/")[-1] for x in
                                                                          self.s3.ls(self.results_dir)]
        if addon + "optuna_study_results.csv" in file_list:
            filepath = self.results_dir + addon + "distributions.json"
            f = open(filepath, "r") if self.s3 is None else self.s3.open(filepath, "r")
            distributions = json.load(f)
            distributions = {k: eval(v["name"])(**{key: value for key, value in v.items() if key != "name"})
                             for k, v in distributions.items()}

            # CSV-stored models
            try:
                df = pd.read_csv(self.results_dir + addon + "optuna_study_results.csv")
                for _, row in df.iterrows():
                    if row["state"] == "RUNNING":
                        continue
                    params = row.filter(like="params_").to_dict()
                    params = {k.replace("params_", ""): v for k, v in params.items()}
                    for k, v in params.items():
                        if (k != "optimizer" and k != "use_batch_norm" and k != "beta1" and k != "beta2"
                                and k != "layer_decay" and k != "scheduler"):
                            params.update({k: int(v)})
                    self.insert_trial(row, params, distributions)
                    self.counter += 1
                print("CSV-stored models inserted!")
            except pd.errors.EmptyDataError:
                print("CSV file empty!")

            # Untracked models
            self.retrieve_untracked_models(distributions)

    def retrieve_untracked_models(self, distributions):
        if self.search_for_untracked_models:
            models_present = [name.strip(".pt") for name in os.listdir(self.results_dir) if ".pt" in name]
            if len(models_present) > 0:
                max_model_id = np.max([int(name[5:]) for name in models_present])
                n_untracked_models = max_model_id - self.counter
                if n_untracked_models > -1:
                    for i in range(n_untracked_models):
                        print()
                        trainer = NetworkTrainer.load_model(self.working_dir, self.model_name,
                                                            trial_n=self.counter, use_cuda=self.use_cuda,
                                                            train_data=self.train_data, val_data=self.val_data,
                                                            test_data=self.test_data, s3=self.s3,
                                                            projection_dataset=self.projection_dataset,
                                                            is_cropped=self.is_cropped)
                        train_stats, val_stats = trainer.summarize_performance(show_test=False, show_process=False,
                                                                               show_cm=False, trial_n=self.counter)
                        val_output = getattr(val_stats, self.output_metric)
                        if self.double_output:
                            train_output = getattr(train_stats, self.output_metric)

                        row = {"number": self.counter,
                               "datetime_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                               "datetime_complete": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}
                        if not self.double_output:
                            row.update({"value": val_output})
                        else:
                            row.update({"values_0": val_output, "values_1": train_output})
                        params = trainer.net_params
                        params.update({"batch_size": int(math.log2(params["batch_size"])),
                                       "lr_last": int(-math.log10(params["lr_last"])),
                                       "n_conv_view_neurons": int(math.log2(params["n_conv_view_neurons"])),
                                       "n_conv_segment_neurons": int(math.log2(params["n_conv_segment_neurons"])),
                                       "p_drop": int(params["p_dropout"] * 10)})
                        del params["p_dropout"]
                        self.insert_trial(row, params, distributions)
                        print("Untracked model", self.counter, "inserted!")
                        self.counter += 1

    def initialize_study(self):
        self.study.optimize(lambda trial: self.objective(trial), self.n_trials, callbacks=[self.store_best_candidates])

    def objective(self, trial):
        # Store previous results
        if self.counter >= 2:
            self.analyze_study(show_best=False)

        # Sample parameters
        params = {
            "n_conv_segment_neurons": np.round(2 ** (trial.suggest_int("n_conv_segment_neurons", 7, 9, step=1))),
            "n_conv_view_neurons": np.round(2 ** (trial.suggest_int("n_conv_view_neurons", 7, 9, step=1))),
            "n_conv_segment_layers": int(trial.suggest_int("n_conv_segment_layers", 1, 2, step=1)),
            "n_conv_view_layers": int(trial.suggest_int("n_conv_view_layers", 1, 2, step=1)),
            "kernel_size": int(trial.suggest_int("kernel_size", 3, 5, step=2)),
            "n_fc_layers": int(trial.suggest_int("n_fc_layers", 1, 2, step=1)),
            "optimizer": trial.suggest_categorical("optimizer", ["RMSprop", "Adam"]),
            "lr_last": np.round(10 ** (-1 * trial.suggest_int("lr_last", 3, 5, step=1)), decimals=6),
            "lr_second_last_factor": trial.suggest_int("lr_second_last_factor", 1, 51, step=10),
            "batch_size": int(2 ** (trial.suggest_int("batch_size", 5, 6, step=1))),
            "p_dropout": np.round(0.1 * trial.suggest_int("p_drop", 3, 7, step=2), decimals=1),
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [False, True]),
        }

        # Define seeds
        NetworkTrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        print("Trial ID:", self.counter)
        print("Parameters:", params)
        self.counter += 1
        try:
            trainer = NetworkTrainer(model_name=self.model_name, working_dir=self.working_dir,
                                     train_data=self.train_data, val_data=self.val_data, test_data=self.test_data,
                                     net_type=self.net_type, epochs=self.epochs, val_epochs=self.val_epochs,
                                     net_params=params, use_cuda=self.use_cuda, s3=self.s3,
                                     n_parallel_gpu=self.n_parallel_gpu, projection_dataset=self.projection_dataset,
                                     preprocess_inputs=self.preprocess_inputs,
                                     enhance_images=self.enhance_images, full_size=self.full_size,
                                     is_cropped=self.is_cropped, weight_loss=self.weight_loss)
            val_metric = trainer.train(show_epochs=False, trial_n=self.counter-1, trial=trial,
                                       output_metric=self.output_metric, double_output=self.double_output)
            print("Value:", val_metric)
            if self.double_output:
                val_metric, train_metric = val_metric

            self.current_trainer = trainer

        except TrialPruned:
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            val_metric = 0
            train_metric = 0

        if not self.double_output:
            return val_metric
        else:
            return val_metric, train_metric

    def analyze_study(self, show_best=True):
        # Store study results
        addon = "" if not self.is_pretrain else "pretrain_"
        print("-------------------------------------------------------------------------------------------------------")
        print("Storing study...")
        df = self.study.trials_dataframe()
        df.to_csv(self.results_dir + addon + "optuna_study_results.csv", index=False)

        # Store parameters distributions
        distr_file = addon + "distributions.json"
        if distr_file not in os.listdir(self.results_dir):
            distributions = {k: {"name": type(v).__name__, **v._asdict()} for k, v in
                             self.study.trials[-1].distributions.items()}
            filepath = self.results_dir + distr_file
            f = open(filepath, "w") if self.s3 is None else self.s3.open(filepath, "w")
            _ = json.dump(distributions, f, indent=4)
            print("Study stored!")

        if not self.double_output:
            if show_best:
                print("Best study:")
                best_trial = self.study.best_trial
                for key, value in best_trial.params.items():
                    print("{}: {}".format(key, value))

            fig = optuna.visualization.plot_intermediate_values(self.study)
            imgpath = self.results_dir + addon + "plot_intermediate_values.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

            targets = [None]

        else:
            if show_best:
                print("Best study:")
                best_trials = self.study.best_trials
                for trial in best_trials:
                    print("Trial ID", trial.number, "- outputs:", trial.values)

            fig = optuna.visualization.plot_pareto_front(self.study,
                                                         target_names=["Validation metric", "Training metric"])
            imgpath = self.results_dir + addon + "plot_pareto_front.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

            targets = [lambda t: t.values[0], lambda t: t.values[1]]

        for i in range(len(targets)):
            target = targets[i]
            addon = str(i) if target is not None else ""
            target_name = "Objective" + addon + " Value"

            fig = optuna.visualization.plot_optimization_history(self.study, target=target, target_name=target_name)
            imgpath = self.results_dir + addon + "plot_optimization_history" + addon + ".jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

            fig = optuna.visualization.plot_parallel_coordinate(self.study, target=target, target_name=target_name)
            imgpath = self.results_dir + addon + "plot_parallel_coordinate" + addon + ".jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")
        
        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            imgpath = self.results_dir + addon + "plot_param_importance.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")
        except RuntimeError as e:
            print("Unable to plot parameter importance!", e)

    def insert_trial(self, row, params, distributions):
        if not self.double_output:
            trial = FrozenTrial(
                number=int(row["number"]),
                state=TrialState.COMPLETE,
                value=row["value"],
                datetime_start=datetime.strptime(row["datetime_start"], "%Y-%m-%d %H:%M:%S.%f"),
                datetime_complete=datetime.strptime(row["datetime_complete"], "%Y-%m-%d %H:%M:%S.%f"),
                params=params,
                distributions=distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=int(row["number"]),
            )
        else:
            trial = FrozenTrial(
                number=int(row["number"]),
                state=TrialState.COMPLETE,
                value=None,
                values=[row["values_0"], row["values_1"]],
                datetime_start=datetime.strptime(row["datetime_start"], "%Y-%m-%d %H:%M:%S.%f"),
                datetime_complete=datetime.strptime(row["datetime_complete"], "%Y-%m-%d %H:%M:%S.%f"),
                params=params,
                distributions=distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=int(row["number"]),
            )
        self.study.add_trial(trial)
        self.store_best_candidates(None, trial)

    def store_best_candidates(self, study, trial):
        is_best_candidate = True
        for past_trial in self.best_trials:
            if self.double_output:
                if ((self.double_output and past_trial.values[0] >= trial.values[0]
                     and past_trial.values[1] >= trial.values[1]) or (not self.double_output
                                                                      and past_trial.value >= trial.value)):
                    is_best_candidate = False
                    break

        if is_best_candidate:
            if self.double_output:
                self.best_trials.append(trial)
            else:
                self.best_trials = [trial]
            if self.current_trainer is not None:
                self.current_trainer.save_model(trial.number)


if __name__ == "__main__":
    # Define variables
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "cropped_projection_resnext101_optuna"
    selected_segments1 = None
    selected_projection1 = None
    net_type1 = NetType.BASE_RES_NEXT101
    epochs1 = 200
    val_epochs1 = 10
    use_cuda1 = True
    projection_dataset1 = True
    enhance_images1 = False
    full_size1 = False
    is_cropped1 = True

    # Load data
    addon = "" if not is_cropped1 else "cropped_"
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon + "xray_dataset_training",
                                           selected_segments=selected_segments1,
                                           selected_projection=selected_projection1)
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon + "xray_dataset_validation",
                                         selected_segments=selected_segments1, selected_projection=selected_projection1)
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon + "xray_dataset_test",
                                          selected_segments=selected_segments1,
                                          selected_projection=selected_projection1)

    # Define Optuna model
    n_trials1 = 50
    output_metric1 = "mcc"
    double_output1 = True
    search_for_untracked_models1 = False
    weight_loss1 = True
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                                val_data=val_data1, test_data=test_data1, net_type=net_type1, epochs=epochs1,
                                val_epochs=val_epochs1, use_cuda=use_cuda1, n_trials=n_trials1,
                                projection_dataset=projection_dataset1, output_metric=output_metric1,
                                double_output=double_output1, search_for_untracked_models=search_for_untracked_models1,
                                enhance_images=enhance_images1, full_size=full_size1, is_cropped=is_cropped1,
                                weight_loss=weight_loss1)
    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
