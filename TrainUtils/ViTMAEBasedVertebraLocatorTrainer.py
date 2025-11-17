# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import optuna
from PIL import Image
from sympy.categories import Object
from tensorflow.python.ops.gen_nn_ops import l2_loss
from transformers import ViTImageProcessor, ViTMAEForPreTraining, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DataUtils.XrayDataset import XrayDataset
from Enumerators.NetType import NetType
from Enumerators.SetType import SetType
from TrainUtils.NetworkTrainer import NetworkTrainer
from TrainUtils.ViTMAETrainer import ViTMAETrainer
from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class VitMAEBasedVertebraLocatorTrainer(NetworkTrainer):

    def __init__(self, model_name, working_dir, pre_trainer, epochs, val_epochs, net_type=NetType.LOCATOR_DEFAULT,
                 convergence_patience=5, convergence_thresh=1e-3, net_params=None, use_cuda=True):
        super().__init__(model_name=model_name, working_dir=working_dir, train_data=pre_trainer.train_data,
                         val_data=pre_trainer.val_data, test_data=pre_trainer.test_data, net_type=net_type, epochs=epochs,
                         val_epochs=val_epochs, convergence_patience=convergence_patience,
                         convergence_thresh=convergence_thresh, net_params=net_params, use_cuda=use_cuda)
        self.crop_refs = {st.value: pd.read_csv(working_dir + XrayDataset.data_fold + st.value + "cropping_ref.csv") for st in SetType}

        self.img_preprocessor = pre_trainer.img_preprocessor
        self.inference_data_transforms = pre_trainer.inference_data_transforms
        self.net_pretrain = pre_trainer.net_pretrain

        # Training parameters
        lr = self.net_params["base_lr"]
        param_groups = [{"params": self.net_pretrain.parameters(), "lr": lr}]
        self.optimizer = torch.optim.AdamW(param_groups, betas=(self.net_params["beta1"], self.net_params["beta2"]),
                                           eps=self.net_params["eps"])
        self.batch_size = self.net_params["batch_size"]

    def apply_network(self, net, instance, set_type, is_vit_mae=False):
        # IoUC???
        item, extra = instance
        projection_type_batch, projection_batch, _ = item
        input = self.preprocess_fn(projection_batch, projection_type_batch, extra, set_type)

        y = (item[-1] != "").astype(int)
        y = torch.tensor(np.array(y)).to(torch.float32)
        y = y.unsqueeze(1)
        y = y.to(self.device)

        # Loss evaluation
        if not is_vit_mae:
            output = net(input, extra[1], projection_type_batch, self.n_parallel_gpu)
        else:
            output = net(input)
        loss = self.criterion(output, y)

        # Accuracy evaluation
        pred = (output > 0.5).to(torch.int)
        acc = torch.sum(pred == y) / y.shape[0]

        return loss, output, y, acc


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    # working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    pretrained_model_name1 = "vitmae"
    model_name1 = "vitmae_vertebra_locator"
    epochs1 = 500
    trial_n1 = 22
    val_epochs1 = 10
    use_cuda1 = False
    assess_calibration1 = False
    show_test1 = False
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None
    preprocess_inputs1 = False
    enhance_images1 = False

    # Load data
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training",
                                           selected_segments=selected_segments1,
                                           selected_projection=selected_projection1)
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation",
                                         selected_segments=selected_segments1, selected_projection=selected_projection1)
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test",
                                          selected_segments=selected_segments1,
                                          selected_projection=selected_projection1)

    pre_trainer1 = ViTMAETrainer.load_model(working_dir=working_dir1, model_name=pretrained_model_name1, trial_n=trial_n1,
                                            use_cuda=use_cuda1, train_data=train_data1, val_data=val_data1,
                                            test_data=test_data1, projection_dataset=projection_dataset1)
    NetworkTrainer.set_seed(111099)
    #pre_trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)

    # Specialize model
    net_params1 = {"n_fc_layers": 2, "n_fc_neurons": 256, "base_lr": 1e-5, "beta1": 0.85, "beta2": 0.9,
                   "layer_decay": 0.75, "eps": 1e-9, "batch_size": 64, "lr": 1e-4, "optimizer": "Adam"}
    trainer1 = VitMAEBasedVertebraLocatorTrainer(model_name=model_name1, working_dir=working_dir1,
                                                 pre_trainer=pre_trainer1, epochs=epochs1, val_epochs=val_epochs1,
                                                 net_params=net_params1, use_cuda=use_cuda1)

    NetworkTrainer.set_seed(111099)
    trainer1.summarize_performance(show_test=show_test1, show_process=True)
