# Import packages
import os
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


# Class
class OptunaParamFinder:

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, net_type, epochs, val_epochs, use_cuda,
                 n_trials, s3=None, n_parallel_gpu=0, projection_dataset=False, output_metric="f1",
                 double_output=False):
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

        self.output_metric = output_metric
        self.double_output = double_output

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
            self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                             pruner=optuna.pruners.MedianPruner())
        else:
            self.study = optuna.create_study(directions=["maximize", "maximize"], sampler=optuna.samplers.TPESampler(),
                                             pruner=optuna.pruners.MedianPruner())
        self.counter = 0

        # Retrieve previous results
        file_list = os.listdir(self.results_dir) if self.s3 is None else [x.split("/")[-1] for x in self.s3.ls(self.results_dir)]
        if "optuna_study_results.csv" in file_list:
            filepath = self.results_dir + "distributions.json"
            f = open(filepath, "r") if self.s3 is None else self.s3.open(filepath, "r")
            distributions = json.load(f)
            distributions = {k: eval(v["name"])(**{key: value for key, value in v.items() if key != "name"})
                             for k, v in distributions.items()}

            df = pd.read_csv(self.results_dir + "optuna_study_results.csv")
            for _, row in df.iterrows():
                params = row.filter(like="params_").to_dict()
                params = {k.replace("params_", ""): v for k, v in params.items()}
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
                    trial_id=self.counter,
                )
                self.study.add_trial(trial)
                self.counter += 1

    def initialize_study(self):
        self.study.optimize(lambda trial: self.objective(trial), self.n_trials)

    def objective(self, trial):
        params = {
            "n_conv_segment_neurons": np.round(2 ** (trial.suggest_int("n_conv_segment_neurons", 8, 10, step=1))),
            "n_conv_view_neurons": np.round(2 ** (trial.suggest_int("n_conv_view_neurons", 8, 10, step=1))),
            "n_conv_segment_layers": int(trial.suggest_int("n_conv_segment_layers", 1, 2, step=1)),
            "n_conv_view_layers": int(trial.suggest_int("n_conv_view_layers", 1, 2, step=1)),
            "kernel_size": int(trial.suggest_int("kernel_size", 3, 7, step=2)),
            "n_fc_layers": int(trial.suggest_int("n_fc_layers", 1, 3, step=1)),
            "optimizer": trial.suggest_categorical("optimizer", ["SGD", "RMSprop", "Adam"]),
            "lr_last": np.round(10 ** (-1 * trial.suggest_int("lr_last", 4, 6, step=1)), decimals=6),
            "lr_second_last_factor": trial.suggest_int("lr_second_last_factor", 1, 101, step=10),
            "batch_size": int(2 ** (trial.suggest_int("batch_size", 5, 6, step=1))),
            "p_dropout": np.round(0.1 * trial.suggest_int("p_drop", 4, 8, step=2), decimals=1),
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [False, True]),
        }

        # Define seeds
        NetworkTrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        print("Parameters:", params)
        self.counter += 1
        try:
            trainer = NetworkTrainer(model_name=self.model_name, working_dir=self.working_dir,
                                     train_data=self.train_data, val_data=self.val_data, test_data=self.test_data,
                                     net_type=self.net_type, epochs=self.epochs, val_epochs=self.val_epochs,
                                     net_params=params, use_cuda=self.use_cuda, s3=self.s3,
                                     n_parallel_gpu=self.n_parallel_gpu, projection_dataset=self.projection_dataset)
            val_metric = trainer.train(show_epochs=False, trial_n=self.counter, trial=trial,
                                       output_metric=self.output_metric, double_output=self.double_output)
            if self.double_output:
                val_metric, train_metric = val_metric
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

    def analyze_study(self):
        # Store study results
        print("Storing study...")
        df = self.study.trials_dataframe()
        df.to_csv(self.results_dir + "optuna_study_results.csv", index=False)

        # Store parameters distributions
        distributions = {k: {"name": type(v).__name__, **v._asdict()} for k, v in
                         self.study.best_trial.distributions.items()}
        filepath = self.results_dir + "distributions.json"
        f = open(filepath, "w") if self.s3 is None else self.s3.open(filepath, "w")
        json_file = json.dump(distributions, f, indent=4)
        print("Study stored!")

        print("Best study:")
        best_trial = self.study.best_trial
        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        fig = optuna.visualization.plot_intermediate_values(self.study)
        imgpath = self.results_dir + "plot_intermediate_values.jpg"
        if self.s3 is not None:
            imgpath = self.s3.open(imgpath, "wb")
        fig.write_image(imgpath, format="jpg")
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        imgpath = self.results_dir + "plot_optimization_history.jpg"
        if self.s3 is not None:
            imgpath = self.s3.open(imgpath, "wb")
        fig.write_image(imgpath, format="jpg")
        
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        imgpath = self.results_dir + "plot_parallel_coordinate.jpg"
        if self.s3 is not None:
            imgpath = self.s3.open(imgpath, "wb")
        fig.write_image(imgpath, format="jpg")
        
        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            imgpath = self.results_dir + "plot_param_importance.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")
        except RuntimeError as e:
            print("Unable to plot parameter importance!", e)

        if self.double_output:
            fig = optuna.visualization.plot_pareto_front(self.study,
                                                         target_names=["Validation metric", "Training metric"])
            imgpath = self.results_dir + "plot_pareto_front.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")


if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"
    model_name1 = "projection_resnet101_optuna"
    selected_segments1 = None
    net_type1 = NetType.BASE_RES_NEXT101
    epochs1 = 2#200
    val_epochs1 = 2#10
    use_cuda1 = True

    # Load data
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training",
                                           selected_segments=selected_segments1)
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation",
                                         selected_segments=selected_segments1)
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test",
                                          selected_segments=selected_segments1)

    # Define Optuna model
    n_trials1 = 2
    output_metric1 = "mcc"
    double_output1 = True
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                                val_data=val_data1, test_data=test_data1, net_type=net_type1, epochs=epochs1,
                                val_epochs=val_epochs1, use_cuda=use_cuda1, n_trials=n_trials1, projection_dataset=True,
                                output_metric=output_metric1, double_output=double_output1)
    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
