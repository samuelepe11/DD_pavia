# Import packages
import optuna
import numpy as np
import torch


# Class
class OptunaParamFinder:

    def __init__(self, model_name, working_dir, task_type, net_type, epochs, batch_size, val_epochs, n_trials,
                 separated_inputs=True):
        self.model_name = model_name
        self.working_dir = working_dir
        self.task_type = task_type
        self.net_type = net_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_epochs = val_epochs
        self.separated_inputs = separated_inputs

        self.counter = 0
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                         pruner=optuna.pruners.MedianPruner())
        self.n_trials = n_trials

    def initialize_study(self):
        self.study.optimize(lambda trial: self.objective(trial), self.n_trials)

    def objective(self, trial):
        params = {
            "n_conv_neurons": int(2 ** (trial.suggest_int("n_conv_neurons", 5, 10, step=1))),
            "n_conv_layers": int(trial.suggest_int("n_conv_layers", 1, 3, step=1)),
            "kernel_size": 3,
            "hidden_dim": int(2 ** (trial.suggest_int("hidden_dim", 5, 8, step=1))),
            "p_drop": np.round(trial.suggest_float("p_drop", 0, 0.4, step=0.2), 1),
            "n_extra_fc_after_conv": int(trial.suggest_int("n_extra_fc_after_conv", 0, 3, step=1)),
            "n_extra_fc_final": int(trial.suggest_int("n_extra_fc_final", 0, 3, step=1)),
            "optimizer": trial.suggest_categorical("optimizer", ["RMSprop", "Adam"]),
            "lr": np.round(10 ** (-1 * trial.suggest_int("lr", 1, 3, step=1)), 3),
            "batch_size": trial.suggest_int("batch_size", 32, 64, step=32)
        }

        # Define seeds
        NetworkTrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        print("Parameters:", params)
        self.counter += 1
        try:
            trainer = NetworkTrainer(model_name=self.model_name, working_dir=self.working_dir, task_type=self.task_type,
                                     net_type=self.net_type, epochs=self.epochs, val_epochs=self.val_epochs,
                                     params=params, separated_inputs=self.separated_inputs)
            val_f1 = trainer.train(show_epochs=False, trial_n=self.counter, trial=trial)
        except torch.cuda.OutOfMemoryError:
            print("CUDA ran out of memory...")
            val_f1 = 0
        return val_f1

    def analyze_study(self):
        print("Best study:")
        best_trial = self.study.best_trial
        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        fig = optuna.visualization.plot_intermediate_values(self.study)
        fig.show()
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.show()
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()


if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"
    model_name1 = "age_conv2d_optuna"
    net_type1 = NetType.CONV2D
    task_type1 = TaskType.AGE
    epochs1 = 200
    batch_size1 = None
    val_epochs1 = 10
    separated_inputs1 = True

    # Define Optuna model
    n_trials1 = 50
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, task_type=task_type1,
                                net_type=net_type1, epochs=epochs1, batch_size=batch_size1,
                                val_epochs=val_epochs1, n_trials=n_trials1, separated_inputs=separated_inputs1)

    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
