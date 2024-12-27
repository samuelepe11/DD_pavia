# Import packages
import matplotlib.pyplot as plt
import optuna
import numpy as np
import torch
import kaleido

from DataUtils.XrayDataset import XrayDataset
from TrainUtils.NetworkTrainer import NetworkTrainer
from Enumerators.NetType import NetType


# Class
class OptunaParamFinder:

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, net_type, epochs, val_epochs, use_cuda,
                 n_trials, s3=None):
        self.model_name = model_name
        self.working_dir = working_dir
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.net_type = net_type
        self.epochs = epochs
        self.val_epochs = val_epochs
        self.use_cuda = use_cuda

        self.counter = 0
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                         pruner=optuna.pruners.MedianPruner())
        self.n_trials = n_trials

        self.results_dir = working_dir + XrayDataset.results_fold + XrayDataset.models_fold + model_name + "/"
        self.s3 = s3

    def initialize_study(self):
        self.study.optimize(lambda trial: self.objective(trial), self.n_trials)

    def objective(self, trial):
        params = {
            "n_conv_segment_neurons": 1024,
            "n_conv_view_neurons": 1024,
            "n_conv_segment_layers": int(trial.suggest_int("n_conv_segment_layers", 1, 3, step=1)),
            "n_conv_view_layers": int(trial.suggest_int("n_conv_view_layers", 1, 3, step=1)),
            "kernel_size": int(trial.suggest_int("kernel_size", 3, 5, step=2)),
            "n_fc_layers": int(trial.suggest_int("n_fc_layers", 1, 3, step=1)),
            "optimizer": trial.suggest_categorical("optimizer", ["RMSprop", "Adam"]),
            "lr_last": np.round(10 ** (-1 * trial.suggest_int("lr_last", 3, 5, step=1)), decimals=3),
            "lr_second_last_factor": trial.suggest_int("lr_second_last_factor", 1, 101, step=10),
            "batch_size": int(2 ** (trial.suggest_int("batch_size", 5, 7, step=1))),
            "p_dropout": np.round(0.1 * trial.suggest_int("p_drop", 0, 6, step=2), decimals=1),
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
                                     net_params=params, use_cuda=self.use_cuda, s3=self.s3)
            val_f1 = trainer.train(show_epochs=False, trial_n=self.counter, trial=trial)
        except Exception as e:
            print(f"An error occurred: {e}")
            val_f1 = 0
        return val_f1

    def analyze_study(self):
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


if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"
    model_name1 = "resnext50_optuna"
    net_type1 = NetType.RES_NEXT50
    epochs1 = 1
    val_epochs1 = 1
    use_cuda1 = False

    # Load data
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training")
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation")
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test")

    # Define Optuna model
    n_trials1 = 1
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                                val_data=val_data1, test_data=test_data1, net_type=net_type1, epochs=epochs1,
                                val_epochs=val_epochs1, use_cuda=use_cuda1, n_trials=n_trials1)
    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
