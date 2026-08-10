# Import packages
import os
import cv2
import torch
import torch.nn as nn
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import optuna
import copy
import io
import gc
import json
import pandas as pd
from pyarrow.dataset import dataset
from sqlalchemy.testing import is_not_
from tensorflow.python.ops.linalg.linalg_impl import transpose
from torch.utils.data import DataLoader, Subset
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from pandas import DataFrame
from functools import partial
from torchvision.transforms.v2.functional import equalize

from Networks.BaseResNeXt50AllFreeze import BaseResNeXt50AllFreeze
from calfram.calibrationframework import select_probability, reliabilityplot, calibrationdiagnosis, classwise_calibration
from contextlib import redirect_stdout
from cleanlab import Datalab
from sklearn.model_selection import StratifiedGroupKFold
from enum import Enum

import textwrap
from matplotlib.backends.backend_pdf import PdfPages

from DataUtils.XrayDataset import XrayDataset
from DataUtils.XrayProjectionDataset import XrayProjectionDataset
from DataUtils.Preprocessor import Preprocessor
from Enumerators.NetType import NetType
from Enumerators.SetType import SetType
from Enumerators.ProjectionType import ProjectionType
from Networks.BaseResNet18 import BaseResNet18
from Networks.BaseResNeXt50 import BaseResNeXt50
from Networks.BaseResNeXt101 import BaseResNeXt101
from Networks.BaseViT import BaseViT
from Networks.AttentionViT import AttentionViT
from Networks.MLPLocator import MLPLocator
from Networks.BaseResNeXt50Bicocca import BaseResNeXt50Bicocca
from TrainUtils.StatsHolder import StatsHolder


# Class
class NetworkTrainer:
    default_net_params = None
    default_net_type = None

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, net_type, epochs, val_epochs,
                 convergence_patience=5, convergence_thresh=1e-3, preprocess_inputs=False, net_params=None,
                 use_cuda=True, s3=None, n_parallel_gpu=0, projection_dataset=False, enhance_images=True, full_size=False,
                 is_cropped=False, weight_loss=False, dynamic_under_sampling=False, transpose=False, equalize_images=False):
        # Initialize attributes
        self.model_name = model_name
        self.working_dir = working_dir
        self.results_dir = working_dir + XrayDataset.results_fold + XrayDataset.models_fold
        self.s3 = s3
        self.n_parallel_gpu = n_parallel_gpu
        if s3 is None:
            if model_name not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + model_name)
        else:
            if not s3.exists(self.results_dir + model_name):
                s3.touch(self.results_dir + model_name + "/empty.txt")
        self.results_dir += model_name + "/"

        self.use_cuda = torch.cuda.is_available() and use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.net_type = net_type if net_type is not None else self.default_net_type
        self.net_params = net_params if net_params is not None else self.default_net_params
        if self.net_type == NetType.BASE_RES_NEXT50:
            self.net = BaseResNeXt50(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling, transpose=transpose)
        elif self.net_type == NetType.BASE_RES_NEXT50_ALLFREEZE:
            self.net = BaseResNeXt50AllFreeze(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling, transpose=transpose)
        elif self.net_type == NetType.BASE_RES_NET18:
            self.net = BaseResNet18(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling, transpose=transpose)
        elif self.net_type == NetType.BASE_RES_NEXT101:
            self.net = BaseResNeXt101(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling, transpose=transpose)
        elif self.net_type == NetType.BASE_VIT:
            self.net = BaseViT(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling, transpose=transpose)
        elif self.net_type == NetType.ATTENTION_VIT:
            self.net = AttentionViT(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling)
        elif self.net_type == NetType.LOCATOR_DEFAULT:
            self.net = MLPLocator(params=self.net_params, device=self.device)
        elif self.net_type == NetType.BASE_BICOCCA:
            self.net = BaseResNeXt50Bicocca(params=self.net_params, device=self.device, weight_loss=weight_loss and not dynamic_under_sampling, transpose=transpose)
        else:
            self.net = None
            print("The selected network is not available.")

        # Define initial state for cleanlab
        self._cleanlab_initial_state = copy.deepcopy(self.net.state_dict())

        # Define training parameters
        self.classes = XrayDataset.classes
        self.epochs = epochs
        self.val_epochs = val_epochs
        try:
            param_groups = [{"params": [param for name, param in self.net.named_parameters() if "resnet.7" in name],
                             "lr": self.net_params["lr_last"] / self.net_params["lr_second_last_factor"]},
                            {"params":  [param for name, param in self.net.named_parameters() if "resnet.7" not in name],
                             "lr": self.net_params["lr_last"]}]
        except KeyError:
            param_groups = [{"params": self.net.parameters(), "lr": self.net_params["lr"]}]
        self.optimizer = getattr(torch.optim, self.net_params["optimizer"])(param_groups)
        self.convergence_thresh = convergence_thresh
        self.convergence_patience = convergence_patience
        self.preprocess_inputs = preprocess_inputs
        self.enhance_images = enhance_images
        self.full_size = full_size
        self.batch_size = self.net_params["batch_size"]

        self.start_time = None
        self.end_time = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_eval_epochs = []
        self.optuna_study = None

        self.projection_dataset = projection_dataset
        if projection_dataset:
            train_data = XrayProjectionDataset(working_dir=working_dir, dataset=train_data)
            val_data = XrayProjectionDataset(working_dir=working_dir, dataset=val_data)
            test_data = XrayProjectionDataset(working_dir=working_dir, dataset=test_data)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        if train_data is not None:
            # Load datasets
            self.train_loader, self.train_dim = self.load_data(train_data, shuffle=True)
            self.val_loader, self.val_dim = self.load_data(val_data)
            self.test_loader, self.test_dim = self.load_data(test_data)

            # Initialize preprocessors
            self.preprocessor = Preprocessor(dataset=train_data, only_apply=True, s3=s3)

        self.is_cropped = is_cropped

        self.dynamic_under_sampling = dynamic_under_sampling
        self.weights_loss = weight_loss
        self.transpose = transpose
        self.equalize_images = equalize_images
        if not weight_loss or dynamic_under_sampling:
            self.criterion = nn.BCELoss()
        else:
            num_pos = 0
            for item, _ in self.train_loader:
                num_pos += np.sum([label != "" for label in item[2]])
            w_pos = self.train_data.len / num_pos - 1
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_pos]))

    def load_data(self, data, shuffle=False):
        if self.use_cuda:
            num_workers = 8
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False
        img_dim = self.net.input_dim if not self.full_size else None
        dataloader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory,
                                collate_fn=partial(NetworkTrainer.custom_collate_fn, img_dim=img_dim))
        dim = len(data)

        return dataloader, dim

    def train(self, show_epochs=False, trial_n=None, trial=None, output_metric="f1", double_output=False):
        if show_epochs:
            self.start_time = time.time()
        
        net = self.net
        if not self.n_parallel_gpu:
            net.set_cuda(cuda=self.use_cuda)
        else:
            if not isinstance(net, nn.DataParallel):
                net = nn.DataParallel(net)
            net = net.cuda()
            
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
    
        if len(self.train_losses) == 0 and trial is None:
            print("\nPerforming initial evaluation...")
            #_, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False)

        if show_epochs:
            print("\nStarting the training phase...")

        if self.dynamic_under_sampling:
            train_dataset_labels = self.get_dataset_binary_labels(self.train_data)
            '''labels = []
            for item, _ in self.train_loader:
                labels += [int(label != "") for label in item[2]]
            labels = np.array(labels)
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]'''

        for epoch in range(self.epochs):
            if not self.n_parallel_gpu:
                net.set_training(True)
            else:
                net.train()

            # loader = self.train_loader if not self.dynamic_under_sampling else self.create_balanced_loader(pos_idx, neg_idx)
            loader = (self.train_loader if not self.dynamic_under_sampling else self.create_balanced_loader(dataset=self.train_data,
                                                                                                            labels=train_dataset_labels,
                                                                                                            neg_proportion=2))
            train_loss = 0
            train_acc = 0
            for batch in loader:
                self.optimizer.zero_grad()
                loss, _, _, acc = self.apply_network(net, batch, set_type=SetType.TRAIN)
                train_loss += loss.item()
                train_acc += acc.item()

                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(loader)
            self.train_losses.append(train_loss)

            train_acc = train_acc / len(loader)
            self.train_accuracies.append(train_acc)

            if show_epochs and (epoch % 10 == 0 or epoch % self.val_epochs == 0):
                print()
                print("Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " completed...")
                print(" > train loss = " + str(np.round(train_loss, 5)))
                print(" > train acc = " + str(np.round(train_acc * 100, 2)) + "%")

            if epoch % self.val_epochs == 0:
                val_stats = self.test(set_type=SetType.VAL)
                self.val_losses.append(val_stats.loss)
                self.val_accuracies.append(val_stats.acc)
                self.val_eval_epochs.append(epoch)
                if show_epochs:
                    print(" > val loss = " + str(np.round(val_stats.loss, 5)))
                    print(" > val acc = " + str(np.round(val_stats.acc * 100, 2)) + "%")

                    # Update and store training curves
                    if epoch != 0:
                        plt.close()
                        self.draw_training_curves()
                        if trial_n is None:
                            filepath = self.results_dir + "training_curves.jpg"
                        else:
                            filepath = self.results_dir + "trial_" + str(trial_n - 1) + "_curves.jpg"
                        if self.s3 is not None:
                            filepath = self.s3.open(filepath, "wb")
                        plt.savefig(filepath)

                        plt.close()
                        self.draw_training_curves()
                        plt.show()

                    # Store intermediate result
                    if trial_n is None:
                        self.save_model()

                if trial is not None and not double_output:
                    trial.report(getattr(val_stats, output_metric), epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if epoch % self.val_epochs == 0 and len(self.val_losses) > self.convergence_patience:
                # Check for convergence
                val_mean = np.mean(self.val_losses[-self.convergence_patience:])
                val_std = np.std(self.val_losses[-self.convergence_patience:])
                val_cv = val_std / val_mean
                if val_cv < self.convergence_thresh:
                    print("Validation convergence has been reached sooner...")
                    break

        self.net = net
        if show_epochs:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print("Execution time:", round(duration / 60, 4), "min")

        if trial_n is None:
            self.save_model()
        else:
            train_stats, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False,
                                                                trial_n=trial_n)
            val_output = getattr(val_stats, output_metric)
            if double_output:
                train_output = getattr(train_stats, output_metric)
                return val_output, train_output
            else:
                return val_output

    def apply_network(self, net, instance, set_type, is_vit_mae=False, zero_shot=False):
        item, extra = instance
        projection_type_batch, projection_batch, _ = item
        input = self.preprocess_fn(projection_batch, projection_type_batch, extra, set_type)
        input = input.to(self.device)

        y = (item[-1] != "").astype(int)
        y = torch.tensor(np.array(y)).to(torch.float32)
        y = y.unsqueeze(1)
        y = y.to(self.device)

        # Loss evaluation
        if not is_vit_mae and not zero_shot:
            output = net(input, extra[1], projection_type_batch, self.n_parallel_gpu)
        else:
            if zero_shot:
                input = input.repeat(1, 3, 1, 1)
                input = input.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                input = [self.net.inference_data_transforms(img) for img in input]
                input = torch.stack(input)
                net.to(self.device)
                net.eval()
            output = net(input)
        probs = output if ((not self.weights_loss or self.dynamic_under_sampling) and not zero_shot) else torch.sigmoid(
            output)
        if not zero_shot:
            loss = self.criterion(probs, y)
        else:
            loss = self.criterion(probs[:, 1], y[:, 0])

        # Accuracy evaluation
        if not zero_shot:
            pred = (probs > 0.5).to(torch.int)
        else:
            pred = (probs[:, 1] > 0.5).to(torch.int)
        acc = torch.sum(pred == y) / y.shape[0]

        return loss, output, y, acc

    def select_dataset(self, set_type):
        if self.train_data is None:
            # Read datasets
            self.train_data = XrayDataset.load_dataset(working_dir=self.working_dir, dataset_name="xray_dataset_training", s3=self.s3)
            self.val_data = XrayDataset.load_dataset(working_dir=self.working_dir, dataset_name="xray_dataset_validation", s3=self.s3)
            self.test_data = XrayDataset.load_dataset(working_dir=self.working_dir, dataset_name="xray_dataset_test", s3=self.s3)

            # Load datasets
            self.train_loader, self.train_dim = self.load_data(self.train_data, shuffle=True)
            self.val_loader, self.val_dim = self.load_data(self.val_data)
            self.test_loader, self.test_dim = self.load_data(self.test_data)

            # Initialize preprocessors
            self.preprocessor = Preprocessor(dataset=self.train_data, only_apply=True, s3=self.s3)

        if set_type == SetType.TRAIN:
            dataset = self.train_data
            loader = self.train_loader
            dim = self.train_dim
        elif set_type == SetType.VAL:
            dataset = self.val_data
            loader = self.val_loader
            dim = self.val_dim
        else:
            # SetType.TEST
            dataset = self.test_data
            loader = self.test_loader
            dim = self.test_dim

        return dataset, loader, dim

    def compute_stats(self, y_true, y_pred, loss, acc, y_prob=None, y_output=None):
        n_vals = len(y_true)
        if loss is None:
            if y_prob is not None:
                tmp = y_prob if (not self.weights_loss or self.dynamic_under_sampling) else y_output
                loss = self.criterion(tmp, y_true)
                loss = loss.item()
            else:
                loss = None

        if acc is None:
            acc = torch.sum(y_true == y_pred) / n_vals
            acc = acc.item()

        # Get binary confusion matrix
        desired_classes = range(len(self.classes)) if len(self.classes) > 2 else [1]
        cm = NetworkTrainer.compute_binary_confusion_matrix(y_true, y_pred, desired_classes)
        tp = cm[0]
        tn = cm[1]
        fp = cm[2]
        fn = cm[3]
        auc = NetworkTrainer.compute_binary_auc(y_true, y_pred, desired_classes)

        stats = StatsHolder(loss, acc, tp, tn, fp, fn, auc)
        return stats

    def test(self, set_type=SetType.TRAIN, show_cm=False, assess_calibration=False, threshold=0.5, zero_shot=False):
        net = self.net if not zero_shot else self.net.feature_extractor_model
        if not zero_shot:
            if not self.n_parallel_gpu:
                net.set_cuda(cuda=self.use_cuda)
                net.set_training(False)
            else:
                net = nn.DataParallel(net)
                net = net.cuda()
                net.eval()
        else:
            net.eval()
            self.net.training = False
            
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        data, loader, _ = self.select_dataset(set_type)

        # Store class labels
        y_prob = []
        y_true = []
        y_pred = []
        y_output = []
        loss = 0
        with torch.no_grad():
            for batch in loader:
                temp_loss, output, y, _ = self.apply_network(net, batch, set_type=set_type, zero_shot=zero_shot)
                loss += temp_loss.item()

                # Accuracy evaluation
                if self.weights_loss and not self.dynamic_under_sampling:
                    y_output.append(output.cpu())

                if output.shape[-1] == 1:
                    if self.weights_loss and not self.dynamic_under_sampling:
                        output = torch.sigmoid(output)
                    prediction = (output > threshold)[:, 0].to(torch.int)
                    output = torch.cat((1 - output, output), dim=-1)
                else:
                    prediction = torch.argmax(output, dim=1)

                y_prob.append(output.cpu())
                y_true.append(y.cpu())
                y_pred.append(prediction.cpu())

            y_prob = torch.cat(y_prob, dim=0)
            y_true = torch.cat(y_true).squeeze(1).to(int)
            y_pred = torch.cat(y_pred).to(int)
            y_output = torch.cat(y_output, dim=0) if len(y_output) > 0 else None

            loss /= len(loader)
            acc = torch.sum(y_true == y_pred) / len(y_true)
            acc = acc.item()
        stats_holder = self.compute_stats(y_true, y_pred, loss, acc, y_output=y_output)

        # Compute multiclass confusion matrix
        cm_name = set_type.value + "_cm"
        if show_cm:
            imgpath = self.results_dir + cm_name + ".jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
        else:
            imgpath = None
        self.__dict__[cm_name] = NetworkTrainer.compute_multiclass_confusion_matrix(y_true, y_pred, self.classes,
                                                                                    imgpath)

        if assess_calibration:
            if len(y_prob.shape) == 1:
                y_prob = np.concatenate([y_prob, 1 - y_prob], axis=1)

            descr = data.dicom_instances if not self.projection_dataset else data.dicom_projection_instances
            stats_holder.calibration_results = self.assess_calibration(y_true, y_prob, y_pred, set_type, descr=descr)

        return stats_holder

    def assess_calibration(self, y_true, y_prob, y_pred, set_type, descr=None):
        y_true = y_true.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        class_scores = select_probability(y_true, y_prob, y_pred)

        # Store results file
        data = np.concatenate((y_true[:, np.newaxis], y_pred[:, np.newaxis], y_prob), axis=1)
        titles = ["y_true", "y_pred"] + ["y_prob" + str(i) for i in range(y_prob.shape[1])]
        if descr is not None:
            descr = np.asarray(descr)
            data = np.concatenate((descr[:, np.newaxis], data), axis=1)
            titles = ["descr"] + titles
        df = DataFrame(data, columns=titles)
        filepath = self.results_dir + set_type.value + "_classification_results.csv"
        if self.s3 is not None:
            filepath = self.s3.open(filepath, "wb")
        df.to_csv(filepath, index=False)

        # Draw reliability plot
        reliabilityplot(class_scores, strategy=10, split=False)
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        filepath = self.results_dir + set_type.value + "_calibration.png"
        if self.s3 is not None:
            filepath = self.s3.open(filepath, "wb")
        plt.savefig(filepath)
        plt.close()

        # Compute local metrics
        results, _ = calibrationdiagnosis(class_scores, strategy=10)

        # Compute global metrics
        results_cw = classwise_calibration(results)
        return results_cw

    def save_model(self, trial_n=None):
        if trial_n is None:
            addon = self.model_name
        else:
            addon = "trial_" + str(trial_n)
            
        filepath = self.results_dir + addon + ".pt"
        if self.s3 is not None:
            filepath = self.s3.open(filepath, "wb")
        net_params = self.net_params if not hasattr(self, "train_parameters") else self.train_parameters
        optimizer = self.optimizer if not hasattr(self, "optimizer_pretrain") else self.optimizer_pretrain
        weight_loss = False if not hasattr(self, "weight_loss") else self.weights_loss
        dynamic_under_sampling = False if hasattr(self, "dynamic_under_sampling") else self.dynamic_under_sampling
        transpose = False if not hasattr(self, "transpose") else self.transpose
        equalize_images = False if not hasattr(self, "equalize_images") else self.equalize_images
        torch.save({"net_type": self.net_type, "epochs": self.epochs, "val_epochs": self.val_epochs,
                    "preprocess_inputs": self.preprocess_inputs, "net_params": net_params,
                    "model_state_dict": self.net.state_dict(), "train_losses": self.train_losses,
                    "val_losses": self.val_losses, "val_eval_epochs": self.val_eval_epochs,
                    "enhance_images": self.enhance_images,
                    "optimizer_pretrain_state_dict": optimizer.state_dict(), "weight_loss": weight_loss,
                    "dynamic_under_sampling": dynamic_under_sampling, "transpose": transpose,
                    "equalize_images": equalize_images}, filepath)

        print("'" + self.model_name + "' has been successfully saved!... train loss: " +
              str(np.round(self.train_losses[0], 4)) + " -> " + str(np.round(self.train_losses[-1], 4)))
        print()

    def summarize_performance(self, show_test=False, show_process=False, show_cm=False, trial_n=None,
                              assess_calibration=False, threshold=0.5, zero_shot=False):
        # Show final losses
        train_stats = self.test(set_type=SetType.TRAIN, show_cm=show_cm, assess_calibration=assess_calibration,
                                threshold=threshold, zero_shot=zero_shot)
        print("Training loss = " + str(np.round(train_stats.loss, 5)) + " - Training accuracy = " +
              str(np.round(train_stats.acc * 100, 7)) + "% - Training F1-score = " +
              str(np.round(train_stats.f1 * 100, 7)) + "%")

        NetworkTrainer.show_performance_table(train_stats, "training")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(train_stats, "training")

        print()
        val_stats = self.test(set_type=SetType.VAL, show_cm=show_cm, assess_calibration=assess_calibration,
                              threshold=threshold, zero_shot=zero_shot)
        print("Validation loss = " + str(np.round(val_stats.loss, 5)) + " - Validation accuracy = " +
              str(np.round(val_stats.acc * 100, 7)) + "% - Validation F1-score = " +
              str(np.round(val_stats.f1 * 100, 7)))

        NetworkTrainer.show_performance_table(val_stats, "validation")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(val_stats, "validation")

        if show_test:
            print()
            test_stats = self.test(set_type=SetType.TEST, show_cm=show_cm, assess_calibration=assess_calibration,
                                   threshold=threshold, zero_shot=zero_shot)
            print("Test loss = " + str(np.round(test_stats.loss, 5)) + " - Test accuracy = " +
                  str(np.round(test_stats.acc * 100, 7)) + "% - Test F1-score = " +
                  str(np.round(test_stats.f1 * 100, 7)))

            NetworkTrainer.show_performance_table(test_stats, "test")
            if assess_calibration:
                NetworkTrainer.show_calibration_table(test_stats, "test")

        if (show_process or trial_n is not None) and len(self.train_losses) != 0:
            self.draw_training_curves()
            if show_process:
                filepath = self.results_dir + "training_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()
            if trial_n is not None:
                filepath = self.results_dir + "trial_" + str(trial_n) + "_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()

        return train_stats, val_stats

    def draw_training_curves(self, is_pretrain=False):
        plt.close()
        fig_size = (5, 7) if not is_pretrain else (7, 5)
        plt.figure(figsize=fig_size)
        plt.suptitle("Training curves")

        # Losses
        if not is_pretrain:
            plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, "b", label="Training set")
        plt.plot(self.val_eval_epochs, self.val_losses, "g", label="Validation set")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

        if not is_pretrain:
            # Accuracies
            plt.subplot(2, 1, 2)
            plt.plot(self.train_accuracies, "b", label="Training set")
            plt.plot(self.val_eval_epochs, self.val_accuracies, "g", label="Validation set")
            plt.legend()
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")

    def show_model(self):
        print("MODEL:")
        attributes = self.net.__dict__
        for attr in attributes.keys():
            val = attributes[attr]
            if issubclass(type(val), nn.Module):
                print(" > " + attr, "-" * (20 - len(attr)), val)

    def preprocess_fn(self, projection_batch, projection_type_batch, extra, set_type, max_proj_num=None):
        adjusted_projection = []
        for i in range(projection_type_batch.shape[0]):
            projection_list = np.split(projection_batch[i], projection_batch.shape[1], axis=0)
            projection_list = [x[0] for x in projection_list]
            try:
                projection_id = extra[2][i]
            except IndexError:
                projection_id = None

            if not self.is_cropped:
                projections = self.preprocessor.mask_projection(projection_list, (extra[0][i], extra[1][i]),
                                                                set_type=set_type, preprocess=self.preprocess_inputs,
                                                                projection_id=projection_id)
            else:
                projections = projection_list

            if self.enhance_images:
                temp = []
                for j in range(len(projections)):
                    projection = projections[j]
                    projection = self.preprocessor.preprocess(img=projection, segm=extra[1], downsampling_iterates=0,
                                                              show=False)
                    temp.append(projection)
                projections = temp
            if self.equalize_images:
                temp = []
                for projection in projections:
                    projection = projection.astype(np.float32)
                    projection /= 255
                    low, hi = np.percentile(projection, [1, 99])
                    projection = np.clip(projection, low, hi)
                    projection = (projection - low) / (hi - low + 1e-6)
                    projection *= 255
                    projection = projection.astype(np.uint8)
                    temp.append(projection)
                projections = temp
            temp = [torch.tensor(projection, dtype=torch.float32) for projection in projections]
            adjusted_projection.append(temp)

        if max_proj_num is not None and max_proj_num > len(adjusted_projection[0]):
            h, w = adjusted_projection[0][-1].shape
            adjusted_projection[0] += [torch.zeros(h, w)] * (max_proj_num - len(adjusted_projection[0]))

        input = np.stack(adjusted_projection)
        input = torch.from_numpy(input)
        return input

    '''def create_balanced_loader(self, pos_idx, neg_idx, neg_proportion=2):
        sampled_neg_idx = np.random.choice(neg_idx, size=neg_proportion*len(pos_idx), replace=False)
        epoch_indices = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(epoch_indices)
        epoch_subset = Subset(self.train_data, epoch_indices)
        loader, _ = self.load_data(epoch_subset, shuffle=True)
        return loader'''

    def create_balanced_loader(self, dataset, labels, neg_proportion=2):
        labels = np.asarray(labels, dtype=np.int64)
        positive_indices = np.flatnonzero(labels == 1)
        negative_indices = np.flatnonzero(labels == 0)
        requested_negatives = int(np.ceil(neg_proportion * len(positive_indices)))
        number_to_sample = min(requested_negatives, len(negative_indices))
        if number_to_sample == len(negative_indices):
            sampled_negative_indices = negative_indices.copy()
        else:
            sampled_negative_indices = np.random.choice(negative_indices, size=number_to_sample, replace=False)
        epoch_indices = np.concatenate([positive_indices, sampled_negative_indices])
        np.random.shuffle(epoch_indices)
        epoch_subset = Subset(dataset, epoch_indices.tolist())
        loader, _ = self.load_data(epoch_subset, shuffle=False)
        return loader

    @staticmethod
    def custom_collate_fn(batch, img_dim=None):
        segment_datas = []
        pt_ids = []
        segment_ids = []
        proj_ids = []
        for segment_data, extra_info in batch:
            try:
                pt_id, segment_id = extra_info
            except ValueError:
                pt_id, segment_id, proj_id = extra_info
                proj_ids.append(proj_id)
            segment_datas.append(segment_data)
            pt_ids.append(pt_id)
            segment_ids.append(segment_id)

        # Resize images to the same dimension and patch channels in the batch
        max_proj_num = np.max([len(x) for x in segment_datas])
        segment_datas_patched = []
        if img_dim is not None:
            img_h = img_dim
            img_w = img_dim
        else:
            shapes = [[proj[1].shape for proj in segment_data] for segment_data in segment_datas]
            img_h = int(np.mean([shape[0][0] for shape in shapes]))
            img_w = int(np.mean([shape[0][1] for shape in shapes]))
        for segment_data in segment_datas:
            temp = []
            for proj in segment_data:
                proj_id = proj[0]
                img = proj[1]
                descr = proj[2]
                temp.append((proj_id, cv2.resize(img, (img_w, img_h)), descr))
            extra_proj = max_proj_num - len(segment_data)
            if extra_proj > 0:
                for _ in range(extra_proj):
                    temp.append((proj_id, np.zeros((img_w, img_h)), descr))
            segment_datas_patched.append(temp)

        batch_proj_id = []
        batch_img = []
        batch_descr = []
        for segment_data in segment_datas_patched:
            proj_id_list = []
            img_list = []
            for proj_data in segment_data:
                proj_id_list.append(proj_data[0])
                img_list.append(proj_data[1])
            batch_proj_id.append(proj_id_list)
            batch_img.append(np.stack(img_list))
            batch_descr.append(segment_data[0][2])
        segment_datas = (np.stack(batch_proj_id, dtype=object), np.stack(batch_img), np.stack(batch_descr,
                                                                                              dtype=object))
        extra = (pt_ids, segment_ids) if len(proj_ids) == 0 else (pt_ids, segment_ids, proj_ids)
        return segment_datas, extra

    @staticmethod
    def compute_binary_confusion_matrix(y_true, y_predicted, classes=None):
        if classes is None or len(classes) == 1:
            # Classical binary computation (class 0 as negative and class 1 as positive)
            tp = torch.sum((y_predicted == 1) & (y_true == 1))
            tn = torch.sum((y_predicted == 0) & (y_true == 0))
            fp = torch.sum((y_predicted == 1) & (y_true == 0))
            fn = torch.sum((y_predicted == 0) & (y_true == 1))

            out = [tp.item(), tn.item(), fp.item(), fn.item()]
            return out
        else:
            # One VS Rest computation for Macro-Averaged F1-score and other metrics
            out = []
            for c in classes:
                y_true_i = (y_true == c).to(int)
                y_predicted_i = (y_predicted == c).to(int)
                out_i = NetworkTrainer.compute_binary_confusion_matrix(y_true_i, y_predicted_i, classes=None)
                out.append(out_i)

            out = np.asarray(out)
            out = [out[:, i] for i in range(out.shape[1])]
            return out

    @staticmethod
    def compute_binary_auc(y_true, y_predicted, classes):
        y_true = y_true.cpu()
        y_predicted = y_predicted.cpu()

        # One VS Rest computation for Macro-Averaged AUC
        out = []
        for c in classes:
            y_true_i = (y_true == c).to(int)
            y_predicted_i = (y_predicted == c).to(int)
            try:
                out_i = roc_auc_score(y_true_i, y_predicted_i)
            except ValueError:
                out_i = 0.5
            out.append(out_i)

        if len(out) == 1:
            out = out[0]
        out = np.asarray(out)
        return out

    @staticmethod
    def compute_multiclass_confusion_matrix(y_true, y_pred, classes, imgpath=None):
        # Compute confusion matrix
        cm = multiclass_confusion_matrix(y_pred, y_true, len(classes))

        # Draw heatmap
        if imgpath is not None:
            NetworkTrainer.draw_multiclass_confusion_matrix(cm, classes, imgpath)

        return cm

    @staticmethod
    def draw_multiclass_confusion_matrix(cm, labels, imgpath):
        plt.figure(figsize=(2, 2))
        try:
            cm = cm.cpu()
        except AttributeError:
            # For the use of this method in MaskSurvey
            cm = np.array(cm)
        plt.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(j, i, f"{val.item()}", ha="center", va="center", color="black", fontsize="xx-large")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel("Predicted class")
        plt.yticks(range(len(labels)), labels, rotation=45)
        plt.ylabel("True class")

        plt.savefig(imgpath, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def load_model(working_dir, model_name, trial_n=None, use_cuda=True, train_data=None, val_data=None, test_data=None,
                   projection_dataset=False, s3=None, batch_size=None, is_cropped=False):
        file_name = model_name if trial_n is None else "trial_" + str(trial_n)
        print("Loading " + file_name + "...")
        filepath = (working_dir + XrayDataset.results_fold + XrayDataset.models_fold + model_name + "/" +
                    file_name + ".pt")
        if s3 is not None:
            filepath = s3.open(filepath)
        checkpoint = torch.load(filepath, weights_only=False)

        if "enhance_images" not in checkpoint.keys():
            checkpoint.update({"enhance_images": False})
        weight_loss = False if not "weight_loss" in checkpoint.keys() else checkpoint["weight_loss"]
        dynamic_under_sampling = False if not "dynamic_under_sampling" in checkpoint.keys() else checkpoint["dynamic_under_sampling"]
        transpose = False if not "transpose" in checkpoint.keys() else checkpoint["transpose"]
        equalize_images = False if not "equalize_images" in checkpoint.keys() else checkpoint["equalize_images"]
        network_trainer = NetworkTrainer(model_name=model_name, working_dir=working_dir, train_data=train_data,
                                         val_data=val_data, test_data=test_data, net_type=checkpoint["net_type"],
                                         epochs=checkpoint["epochs"], val_epochs=checkpoint["val_epochs"],
                                         preprocess_inputs=checkpoint["preprocess_inputs"],
                                         net_params=checkpoint["net_params"], use_cuda=use_cuda, s3=s3,
                                         projection_dataset=projection_dataset, enhance_images=checkpoint["enhance_images"],
                                         is_cropped=is_cropped, weight_loss=weight_loss, dynamic_under_sampling=dynamic_under_sampling,
                                         transpose=transpose, equalize_images=equalize_images)
        network_trainer.net.load_state_dict(checkpoint["model_state_dict"])

        if batch_size is not None:
            network_trainer.batch_size = batch_size

        # Handle models created with Optuna
        if trial_n is None and network_trainer.model_name.endswith("_optuna"):
            old_model_name = network_trainer.model_name
            network_trainer.model_name = old_model_name[:-7]
            addon = XrayDataset.models_fold
            if addon in network_trainer.results_dir:
                addon = ""
            network_trainer.results_dir = (network_trainer.results_dir[:-(len(old_model_name) + 1)] +
                                           addon + network_trainer.model_name + "/")
        return network_trainer

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def show_performance_table(stats, set_name, boot_stats=None):
        print("Performance for", set_name.upper() + " set:")
        for stat in ["acc", "loss"] + StatsHolder.table_stats:
            if boot_stats is None:
                addon = ""
            else:
                if stat not in ["loss", "mcc"]:
                    s = str(np.round(boot_stats.__dict__[stat + "_s"] * 100, 2)) + "%)"
                else:
                    s = str(np.round(boot_stats.__dict__[stat + "_s"], 2)) + ")"
                addon = " (std: " + s

            if stat not in ["loss", "mcc"]:
                s = str(np.round(stats.__dict__[stat] * 100, 2)) + "%"
            else:
                s = str(np.round(stats.__dict__[stat], 2))
            name = StatsHolder.comparable_stats[stat] if stat in StatsHolder.comparable_stats else stat.upper()
            print(" - " + name + ": " + s + addon)

    @staticmethod
    def show_calibration_table(stats, set_name):
        print("Calibration information for", set_name.upper() + " set:")
        for stat in stats.calibration_results.keys():
            print(" - " + stat + ": " + str(stats.calibration_results[stat]))

    def clean_labels(self, records, dataset=None, k=5, clean_epochs=20, seed=111099, neg_proportion=2, output_dir=None,
                     show=False, feature_layer_ind=-1):
        # Check inputs
        if dataset is None:
            dataset = self.train_data
        records_df = DataFrame(records).reset_index(drop=True)
        if not self.projection_dataset:
            raise ValueError("The supplied records are projection-level records. Initialize NetworkTrainer with "
                             "projection_dataset=True.")
        if len(records_df) != len(dataset):
            raise ValueError("Record/dataset length mismatch: " + f"{len(records_df)} records versus " +
                             f"{len(dataset)} items in the dataset. The records must correspond exactly to "
                             f"XrayProjectionDataset.")
        labels = records_df["label"].to_numpy(dtype=np.int64)
        patient_ids = records_df["patient_id"].to_numpy()
        if k < 2:
            raise ValueError("k must be at least 2.")
        n_patients = len(np.unique(patient_ids))
        if k > n_patients:
            raise ValueError(f"k={k}, but the dataset contains only {n_patients} patients.")

        # Each class should occur in at least K different patients.
        for class_id in [0, 1]:
            class_patient_count = len(np.unique(patient_ids[labels == class_id]))
            if class_patient_count < k:
                raise ValueError(f"Class {class_id} occurs in only " + f"{class_patient_count} different patients, but k={k}. "
                    "Reduce k or add more patients containing this class.")

        # Create an optimizer for a new fold model
        def create_optimizer(model):
            try:
                last_parameters = [parameter for name, parameter in model.named_parameters() if "resnet.7" in name]
                other_parameters = [parameter for name, parameter in model.named_parameters() if "resnet.7" not in name]
                parameter_groups = [{"params": last_parameters, "lr": (self.net_params["lr_last"] / self.net_params["lr_second_last_factor"])},
                                    {"params": other_parameters, "lr": self.net_params["lr_last"]}]
            except KeyError:
                parameter_groups = [{"params": model.parameters(), "lr": self.net_params["lr"]}]
            optimizer_class = getattr(torch.optim, self.net_params["optimizer"])
            return optimizer_class(parameter_groups)

        # Restore a fresh model for each fold
        def create_fresh_model():
            if self.n_parallel_gpu != 0:
                raise ValueError("The code is not compatible with multi-GPU training for cleanlab. Set n_parallel_gpu=0 "
                                 "to use a single GPU or CPU.")
            model = copy.deepcopy(self.net)
            model.load_state_dict(copy.deepcopy(self._cleanlab_initial_state))
            model.set_cuda(cuda=self.use_cuda)
            return model

        # Run the same preprocessing and model forward pass
        def forward_batch(model, batch, set_type):
            item, extra = batch
            projection_type_batch, projection_batch, _ = item
            inputs = self.preprocess_fn(projection_batch, projection_type_batch, extra, set_type)
            inputs = inputs.to(self.device)
            batch_labels = (item[-1] != "").astype(np.int64)
            batch_labels = torch.as_tensor(batch_labels, dtype=torch.float32, device=self.device).unsqueeze(1)
            outputs = model(inputs, extra[1], projection_type_batch, self.n_parallel_gpu)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            return outputs, batch_labels
        uses_logits = (self.weights_loss and not self.dynamic_under_sampling)

        def evaluate_fold_model(model, loader, criterion):
            model.set_training(False)
            all_probabilities = []
            all_labels = []
            total_loss = 0.0
            total_examples = 0
            with torch.no_grad():
                for batch in loader:
                    outputs, batch_labels = forward_batch(model, batch, SetType.VAL)
                    if uses_logits:
                        probabilities = torch.sigmoid(outputs)
                        loss = criterion(outputs, batch_labels)
                    else:
                        probabilities = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
                        loss = criterion(probabilities, batch_labels)
                    batch_size = batch_labels.shape[0]
                    total_loss += loss.item() * batch_size
                    total_examples += batch_size
                    all_probabilities.append(probabilities.reshape(-1).cpu())
                    all_labels.append(batch_labels.reshape(-1).cpu())
            probabilities = torch.cat(all_probabilities).numpy()
            true_labels = torch.cat(all_labels).numpy().astype(np.int64)
            predictions = (probabilities >= 0.5).astype(np.int64)
            mean_loss = total_loss / total_examples
            mcc = float(matthews_corrcoef(true_labels, predictions))
            return mean_loss, mcc, probabilities, true_labels

        def predict_loaded_model(dataset):
            model = self.net
            model.set_cuda(cuda=self.use_cuda)
            model.set_training(False)
            loader, _ = self.load_data(dataset, shuffle=False)
            all_probabilities = []
            with torch.no_grad():
                for batch in loader:
                    outputs, _ = forward_batch(model, batch, dataset.set_type)
                    probabilities = torch.sigmoid(outputs) if uses_logits else torch.clamp(outputs, min=1e-7,
                                                                                           max=1.0 - 1e-7)
                    all_probabilities.append(probabilities.reshape(-1).cpu())
            positive_probabilities = torch.cat(all_probabilities).numpy()
            pred_probs = np.stack((1.0 - positive_probabilities, positive_probabilities), axis=1).astype(np.float32)
            return pred_probs

        if dataset.set_type == SetType.TRAIN:
            # Create patient-level folds
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            splits = splitter.split(X=np.zeros(len(labels)), y=labels, groups=patient_ids)
            oof_pred_probs = np.full(shape=(len(labels), 2), fill_value=np.nan, dtype=np.float32)

            # Train model for each fold
            for fold_idx, (train_indices, held_out_indices) in enumerate(splits, start=1):
                NetworkTrainer.set_seed(seed + fold_idx)
                train_indices = np.asarray(train_indices, dtype=np.int64)
                held_out_indices = np.asarray(held_out_indices, dtype=np.int64)
                train_labels = labels[train_indices]
                held_out_labels = labels[held_out_indices]
                if show:
                    print()
                    print(f"Cleanlab fold {fold_idx}/{k}")
                    print(f"  Train: {len(train_indices)} projections, " + f"{len(np.unique(patient_ids[train_indices]))} patients, "
                          + f"{np.sum(train_labels == 1)} positive")
                    print(f"  Held out: {len(held_out_indices)} projections, " + f"{len(np.unique(patient_ids[held_out_indices]))} patients, "
                          + f"{np.sum(held_out_labels == 1)} positive")
                train_subset = Subset(dataset, train_indices.tolist())
                held_out_subset = Subset(dataset, held_out_indices.tolist())
                model = create_fresh_model()
                optimizer = create_optimizer(model)
                if uses_logits:
                    n_positive = int(np.sum(train_labels == 1))
                    n_negative = int(np.sum(train_labels == 0))
                    if n_positive == 0:
                        raise ValueError(f"Fold {fold_idx} has no positive training examples.")
                    pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32, device=self.device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    criterion = nn.BCELoss().to(self.device)
                standard_train_loader, _ = self.load_data(train_subset, shuffle=True)
                for epoch in range(clean_epochs):
                    model.set_training(True)
                    if self.dynamic_under_sampling:
                        positive_local_indices = np.flatnonzero(train_labels == 1)
                        negative_local_indices = np.flatnonzero(train_labels == 0)
                        if len(positive_local_indices) == 0:
                            raise ValueError(f"Fold {fold_idx} contains no positive examples.")
                        requested_negatives = (neg_proportion * len(positive_local_indices))
                        n_sampled_negatives = min(requested_negatives, len(negative_local_indices))
                        sampled_negative_indices = np.random.choice(negative_local_indices, size=n_sampled_negatives, replace=False)
                        epoch_local_indices = np.concatenate([positive_local_indices, sampled_negative_indices])
                        np.random.shuffle(epoch_local_indices)
                        epoch_dataset = Subset(train_subset, epoch_local_indices.tolist())
                        epoch_loader, _ = self.load_data(epoch_dataset, shuffle=True)
                    else:
                        epoch_loader = standard_train_loader
                    running_loss = 0.0
                    for batch in epoch_loader:
                        optimizer.zero_grad(set_to_none=True)
                        outputs, batch_labels = forward_batch(model, batch, SetType.TRAIN)
                        if uses_logits:
                            loss = criterion(outputs, batch_labels)
                        else:
                            probabilities = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
                            loss = criterion(probabilities, batch_labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.detach().item()
                    if show and (epoch == 0 or epoch == clean_epochs - 1 or (epoch + 1) % 10 == 0):
                        mean_loss = running_loss / len(epoch_loader)
                        print(f"  Epoch {epoch + 1}/{clean_epochs}: " + f"loss={mean_loss:.5f}")

                train_eval_loader, _ = self.load_data(train_subset, shuffle=False)
                train_eval_loss, train_mcc, _, train_eval_labels = (evaluate_fold_model(model, train_eval_loader, criterion))
                held_out_loader, _ = self.load_data(held_out_subset, shuffle=False)
                held_out_loss, held_out_mcc, fold_positive_probabilities, evaluated_held_out_labels = evaluate_fold_model(model,
                                                                                                                          held_out_loader,
                                                                                                                          criterion)
                if not np.array_equal(evaluated_held_out_labels, held_out_labels):
                    raise RuntimeError("Held-out labels are not aligned with predictions.")
                if show:
                    print(f"  Fold {fold_idx} complete:\n" + f"    train loss    = {train_eval_loss:.5f}\n" +
                          f"    train MCC     = {train_mcc:.4f}\n" + f"    held-out loss = {held_out_loss:.5f}\n" +
                          f"    held-out MCC  = {held_out_mcc:.4f}")

                # Cleanlab probabilities
                oof_pred_probs[held_out_indices, 1] = (fold_positive_probabilities)
                oof_pred_probs[held_out_indices, 0] = (1.0 - fold_positive_probabilities)
                del model
                del optimizer
                del criterion
                if self.use_cuda:
                    torch.cuda.empty_cache()
        else:
            if show:
                print(f"\nUsing the original model to evaluate the {dataset.set_type.value} set...")
            oof_pred_probs = predict_loaded_model(dataset)

        # Validate the complete OOF probability matrix
        if not np.all(np.isfinite(oof_pred_probs)):
            missing_indices = np.flatnonzero(~np.isfinite(oof_pred_probs).all(axis=1))
            raise RuntimeError("Some examples did not receive an out-of-fold prediction: " + f"{missing_indices[:20].tolist()}")
        if not np.allclose(oof_pred_probs.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Each row of oof_pred_probs must sum to one.")

        # Run Cleanlab
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, dataset.set_type.value + "_cleanlab")
        lab, full_report, features, issue_summary = (self.run_cleanlab_full_audit(records_df=records_df, labels=labels,
                                                                                  oof_pred_probs=oof_pred_probs,
                                                                                  output_dir=output_dir,
                                                                                  feature_layer_ind=feature_layer_ind,
                                                                                  show=show, dataset=dataset))
        return lab, full_report, oof_pred_probs, features, issue_summary,

    def extract_features_from_loaded_model(self, dataset=None, feature_layer_ind=-1, show=False):
        if dataset is None:
            dataset = self.train_data
        loaded_model = self.net
        was_training = bool(getattr(loaded_model, "training", False))
        if hasattr(loaded_model, "set_cuda"):
            loaded_model.set_cuda(cuda=self.use_cuda)
        else:
            loaded_model.to(self.device)
        if hasattr(loaded_model, "set_training"):
            loaded_model.set_training(False)
        else:
            loaded_model.eval()
        named_modules = dict(loaded_model.named_modules())
        linear_layers = [(name, module) for name, module in loaded_model.named_modules() if isinstance(module, nn.Linear)]
        if show:
            print("Available Linear layers in the loaded model:")
            for name, layer in linear_layers:
                print(f"  {name}: {layer.in_features} -> {layer.out_features}")
            selected_layer = list(named_modules.values())[feature_layer_ind]
            selected_name = list(named_modules.keys())[feature_layer_ind]
        if show:
            print(f"Extracting features from the INPUT of layer: {selected_name}")
        loader, _ = self.load_data(dataset, shuffle=False)
        all_features = []
        captured_for_batch = []

        def capture_layer_input(module, inputs):
            value = inputs[0]
            if isinstance(value, (tuple, list)):
                value = value[0]
            if not torch.is_tensor(value):
                raise TypeError(f"The input of layer '{selected_name}' is not a tensor: {type(value)}")
            if value.ndim > 2:
                value = torch.flatten(value, start_dim=1)
            captured_for_batch.append(value.detach())
        hook = selected_layer.register_forward_pre_hook(capture_layer_input)
        try:
            with torch.no_grad():
                for batch in loader:
                    item, extra = batch
                    projection_type_batch, projection_batch, _ = item
                    inputs = self.preprocess_fn(projection_batch, projection_type_batch, extra, SetType.VAL).to(self.device)
                    captured_for_batch.clear()
                    _ = loaded_model(inputs, extra[1], projection_type_batch, 0)
                    if len(captured_for_batch) != 1:
                        raise RuntimeError(f"Layer '{selected_name}' was triggered " +
                                           f"{len(captured_for_batch)} times for one batch. " +
                                           "Specify a different feature_layer_name or implement an explicit "
                                           "forward_features() method.")
                    batch_features = captured_for_batch[0]
                    if batch_features.shape[0] != inputs.shape[0]:
                        raise RuntimeError("Feature batch size does not match input batch size: " +
                                           f"{batch_features.shape[0]} vs {inputs.shape[0]}.")
                    all_features.append(batch_features.cpu())
        finally:
            hook.remove()
            if hasattr(loaded_model, "set_training"):
                loaded_model.set_training(was_training)
            elif was_training:
                loaded_model.train()
            else:
                loaded_model.eval()
        features = torch.cat(all_features, dim=0).numpy().astype(np.float32, copy=False)
        if features.ndim != 2:
            raise RuntimeError(f"Expected a 2D feature matrix, got {features.shape}.")
        if features.shape[0] != len(dataset):
            raise RuntimeError(f"Extracted {features.shape[0]} feature vectors for {len(dataset)} examples.")
        if not np.isfinite(features).all():
            raise RuntimeError("Extracted features contain NaN or infinite values.")
        if show:
            print("Loaded-model feature matrix:", features.shape)
            print("Feature memory: %.2f MB" % (features.nbytes / 1024 ** 2))
        return features, selected_name

    def run_cleanlab_full_audit(self, records_df, labels, oof_pred_probs, output_dir, feature_layer_ind=-1, show=False,
                                dataset=None):
        if dataset is None:
            dataset = self.train_data

        requested_issues = ["null", "label", "outlier", "near_duplicate", "non_iid", "class_imbalance",
                            "underperforming_group", "data_valuation"]
        os.makedirs(output_dir, exist_ok=True)
        issues_dir = os.path.join(output_dir, "issues")
        os.makedirs(issues_dir, exist_ok=True)
        labels = np.asarray(labels, dtype=np.int64)
        oof_pred_probs = np.asarray(oof_pred_probs, dtype=np.float32)
        if not np.isfinite(oof_pred_probs).all():
            raise ValueError("oof_pred_probs contains NaN or infinite values.")
        if not np.allclose(oof_pred_probs.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Every row of oof_pred_probs must sum to one.")
        gc.collect()
        if self.use_cuda:
            torch.cuda.empty_cache()
        features, selected_feature_layer = self.extract_features_from_loaded_model(dataset=dataset,
                                                                                   feature_layer_ind=feature_layer_ind,
                                                                                   show=show)
        if features.shape[0] != len(labels):
            raise RuntimeError(f"Feature/label length mismatch: {features.shape[0]} vs {len(labels)}.")
        np.save(os.path.join(output_dir, "cleanlab_original_model_features.npy"), features)
        np.save(os.path.join(output_dir, "cleanlab_oof_pred_probs.npy"), oof_pred_probs)
        cleanlab_data = DataFrame({"label": labels.astype(np.int64)})
        lab = Datalab(data=cleanlab_data, label_name="label")

        # Run incrementally
        check_calls = {"null": {"features": features},
                       "label": {"pred_probs": oof_pred_probs},
                       "outlier": {"features": features},
                       "near_duplicate": {"features": features},
                       "non_iid": {"features": features},
                       "class_imbalance": {},
                       "underperforming_group": {"pred_probs": oof_pred_probs, "features": features},
                       "data_valuation": {"features": features}}
        issue_frames = {}
        summary_rows = []
        failed_checks = {}
        interpretations = {
            "label": ("Possible incorrect annotation. Verify clinically; correct the label or remove only when the case"
                      " cannot be reliably relabeled."),
            "outlier": ("Atypical in the loaded model's feature space. Inspect for bad crop, artifact, domain shift, "
                        "or a rare but valid clinical case."),
            "near_duplicate": ("Very similar to another example. Review duplicate sets and prevent patient/image "
                               "leakage across train/validation/test splits."),
            "non_iid": ("Dataset-level ordering/dependence signal. Do not delete individual rows solely from this "
                        "check; investigate ordering and patient-level splitting."),
            "class_imbalance": ("Dataset/group-level class imbalance. Do not delete minority-class rows; consider "
                                "weighting, sampling, or collecting more minority data."),
            "underperforming_group": ("A feature-space subgroup on which OOF predictions perform poorly. Investigate "
                                      "the subgroup rather than automatically deleting it."),
            "data_valuation": ("Low estimated KNN-Shapley training value. Use as a review signal only; rare valid cases"
                               " can still be important."),
            "null": ("Missing/invalid feature representation. Re-extract or fix the example; remove only if the "
                     "underlying image is unusable.")}
        for issue_name in requested_issues:
            if show:
                print("\n" + "=" * 70)
                print("Running Cleanlab issue:", issue_name)
                print("=" * 70)
            try:
                lab.find_issues(issue_types={issue_name: {}}, **check_calls[issue_name])
                issue_df = lab.get_issues(issue_name).reset_index(drop=True).copy()
                issue_frames[issue_name] = issue_df
                issue_flag_column = f"is_{issue_name}_issue"
                number_flagged = (int(issue_df[issue_flag_column].fillna(False).astype(bool).sum())
                                  if issue_flag_column in issue_df.columns else np.nan)
                issue_summary = lab.get_issue_summary(issue_name)
                overall_score = (float(issue_summary["score"].iloc[0]) if "score" in issue_summary.columns and
                                                                          len(issue_summary) > 0 else np.nan)
                summary_rows.append({"issue_type": issue_name, "status": "completed", "number_flagged": number_flagged,
                                     "overall_quality_score": overall_score, "interpretation": interpretations[issue_name],
                                     "error": ""})
                issue_df.to_csv(os.path.join(issues_dir, f"{issue_name}_all_examples.csv"), index=False)
                if issue_flag_column in issue_df.columns:
                    (issue_df.loc[issue_df[issue_flag_column].fillna(False).astype(bool)]
                     .to_csv(os.path.join(issues_dir, f"{issue_name}_flagged_examples.csv"), index=False))
            except Exception as exc:
                failed_checks[issue_name] = repr(exc)
                summary_rows.append({"issue_type": issue_name, "status": "failed", "number_flagged": np.nan,
                                     "overall_quality_score": np.nan, "interpretation": interpretations[issue_name],
                                     "error": repr(exc)})
                print(f"Cleanlab check '{issue_name}' failed: {exc}")
            gc.collect()

        def readable_value(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (list, tuple, dict, np.ndarray)):
                return json.dumps(np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
            return value

        metadata_columns = ["dicom_instance", "patient_id", "projection_idx", "projection_id"]
        metadata_columns = [column for column in metadata_columns if column in records_df.columns]
        review_table = records_df[metadata_columns].reset_index(drop=True).copy()
        review_table.insert(0, "dataset_index", np.arange(len(labels), dtype=np.int64))
        for column in review_table.columns:
            review_table[column] = review_table[column].map(readable_value)
        review_table["original_label"] = labels
        review_table["label_name"] = DataFrame({"label": labels})["label"].map({0: "no_fracture", 1: "fracture"})
        review_table["probability_no_fracture"] = oof_pred_probs[:, 0]
        review_table["probability_fracture"] = oof_pred_probs[:, 1]
        review_table["oof_predicted_label"] = np.argmax(oof_pred_probs, axis=1)
        review_table["oof_prediction_correct"] = (review_table["oof_predicted_label"].to_numpy() == labels)
        review_table["probability_assigned_to_original_label"] = oof_pred_probs[np.arange(len(labels)), labels]
        review_table["oof_prediction_confidence"] = np.max(oof_pred_probs, axis=1)
        for issue_name, issue_df in issue_frames.items():
            if len(issue_df) != len(review_table):
                raise RuntimeError(f"Issue '{issue_name}' returned {len(issue_df)} rows for {len(review_table)} examples.")
            for column in issue_df.columns:
                values = issue_df[column].map(readable_value)
                review_table[f"{issue_name}_{column}"] = values.to_numpy()

        def flag_value(row, issue_name):
            column = f"{issue_name}_is_{issue_name}_issue"
            if column not in row.index:
                return False
            value = row[column]
            return False if pd.isna(value) else bool(value)

        point_issue_names = ["null", "label", "outlier", "near_duplicate", "data_valuation"]
        point_flag_columns = [f"{name}_is_{name}_issue" for name in point_issue_names if f"{name}_is_{name}_issue" in review_table.columns]
        if point_flag_columns:
            review_table["pointwise_issue_count"] = (review_table[point_flag_columns].fillna(False).astype(bool).sum(axis=1))
        else:
            review_table["pointwise_issue_count"] = 0
        point_score_columns = [f"{name}_{name}_score" for name in point_issue_names if f"{name}_{name}_score" in review_table.columns]
        if point_score_columns:
            review_table["minimum_pointwise_quality_score"] = (review_table[point_score_columns].apply(pd.to_numeric, errors="coerce").min(axis=1))
        else:
            review_table["minimum_pointwise_quality_score"] = np.nan

        def choose_action(row):
            has_null = flag_value(row, "null")
            has_label = flag_value(row, "label")
            has_outlier = flag_value(row, "outlier")
            has_duplicate = flag_value(row, "near_duplicate")
            has_low_value = flag_value(row, "data_valuation")
            has_group = flag_value(row, "underperforming_group")
            has_imbalance = flag_value(row, "class_imbalance")
            has_non_iid = flag_value(row, "non_iid")
            strong_disagreement = row["probability_assigned_to_original_label"] < 0.20
            if has_null:
                return 5, "FIX/RE-EXTRACT: invalid or missing feature vector"
            if has_label and strong_disagreement:
                return 5, "VERIFY LABEL URGENTLY: strong OOF disagreement"
            if has_label:
                return 4, "VERIFY LABEL: correct it, or remove only if unverifiable"
            if has_duplicate:
                return 4, "CHECK DUPLICATE SET: remove redundancy or prevent split leakage"
            if has_outlier and has_low_value:
                return 4, "HIGH-PRIORITY VISUAL REVIEW: atypical and low estimated value"
            if has_outlier:
                return 3, "VISUAL REVIEW: artifact/bad crop versus rare valid case"
            if has_low_value:
                return 2, "REVIEW: low estimated value alone is not enough for deletion"
            if has_group:
                return 2, "INVESTIGATE SUBGROUP: improve coverage/model, do not auto-delete"
            if has_imbalance:
                return 1, "RETAIN: minority-class flag; use weighting/sampling"
            if has_non_iid:
                return 1, "CHECK ORDER/SPLIT: do not delete based on this row"
            return 0, "NO STRONG CLEANLAB REMOVAL SIGNAL"
        actions = review_table.apply(choose_action, axis=1)
        review_table["manual_review_priority"] = [item[0] for item in actions]
        review_table["suggested_action"] = [item[1] for item in actions]
        review_table = review_table.sort_values(by=["manual_review_priority", "pointwise_issue_count",
                                                    "minimum_pointwise_quality_score",
                                                    "probability_assigned_to_original_label"],
                                                ascending=[False, False, True, True],
                                                na_position="last",).reset_index(drop=True)
        summary_table = DataFrame(summary_rows)
        summary_table["feature_source"] = f"loaded self.net; input of layer '{selected_feature_layer}'"
        summary_table["feature_shape"] = str(tuple(features.shape))
        full_csv_path = os.path.join(output_dir, "cleanlab_full_manual_review.csv")
        summary_csv_path = os.path.join(output_dir, "cleanlab_issue_summary.csv")
        report_txt_path = os.path.join(output_dir, "cleanlab_full_report.txt")
        review_table.to_csv(full_csv_path, index=False)
        summary_table.to_csv(summary_csv_path, index=False)
        report_buffer = io.StringIO()
        with redirect_stdout(report_buffer):
            lab.report()
        cleanlab_report_text = report_buffer.getvalue()
        with open(report_txt_path, "w", encoding="utf-8") as report_file:
            report_file.write("FEATURE EXTRACTION\n")
            report_file.write("==================\n")
            report_file.write("Source: original/loaded self.net (not K-fold models)\n")
            report_file.write(f"Layer input: {selected_feature_layer}\n")
            report_file.write(f"Feature shape: {features.shape}\n\n")
            report_file.write("CHECK EXECUTION SUMMARY\n")
            report_file.write("=======================\n")
            report_file.write(summary_table.to_string(index=False))
            report_file.write("\n\nCLEANLAB REPORT\n")
            report_file.write("===============\n")
            report_file.write(cleanlab_report_text)
            if failed_checks:
                report_file.write("\n\nFAILED CHECKS\n")
                report_file.write("=============\n")
                for name, error in failed_checks.items():
                    report_file.write(f"{name}: {error}\n")
        print("\nCleanlab full audit saved in:", output_dir)
        print("Full per-image review:", full_csv_path)
        print("Issue summary:", summary_csv_path)
        print("Text report:", report_txt_path)
        print("Original-model features:", os.path.join(output_dir, "cleanlab_original_model_features.npy"))
        return lab, review_table, features, summary_table

    def create_cleanlab_review_pdf(self, manual_review_path=None, output_pdf_path=None, include_dataset_level_issues=True,
                                   max_pages=None, removable_instances_path=None):
        # Paths
        cleanlab_dir = os.path.join(self.results_dir, "cleanlab")
        if manual_review_path is None:
            manual_review_path = os.path.join(cleanlab_dir, "cleanlab_full_manual_review.csv")
        if output_pdf_path is None:
            output_pdf_path = os.path.join(cleanlab_dir, "cleanlab_manual_review.pdf")
        if not os.path.isfile(manual_review_path):
            raise FileNotFoundError(f"Manual-review CSV not found: {manual_review_path}")
        os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
        review_df = pd.read_csv(manual_review_path)

        if removable_instances_path is None:
            removable_instances_path = os.path.join(self.working_dir, "data", "instance_removable.txt")
        if not os.path.isfile(removable_instances_path):
            raise FileNotFoundError(f"Removable-instances file not found: {removable_instances_path}")
        with open(removable_instances_path, "r", encoding="utf-8") as removable_file:
            removable_instances = {line.strip().strip(",").strip("'\"") for line in removable_file if line.strip()}
        print("Instances to exclude from PDF:", len(removable_instances))

        # Cleanlab issue columns
        issue_definitions = {
            "null": {"flag": "null_is_null_issue", "name": "Invalid/null feature vector", "dataset_level": False},
            "label": {"flag": "label_is_label_issue", "name": "Possible incorrect label", "dataset_level": False},
            "outlier": {"flag": "outlier_is_outlier_issue", "name": "Feature-space outlier", "dataset_level": False},
            "near_duplicate": {"flag": "near_duplicate_is_near_duplicate_issue", "name": "Near duplicate", "dataset_level": False},
            "data_valuation": {"flag": "data_valuation_is_data_valuation_issue", "name": "Low data valuation",
                               "dataset_level": False},
            "non_iid": {"flag": "non_iid_is_non_iid_issue", "name": "Non-IID pattern", "dataset_level": True},
            "class_imbalance": {"flag": "class_imbalance_is_class_imbalance_issue", "name": "Class-imbalance flag",
                                "dataset_level": True},
            "underperforming_group": {"flag": ("underperforming_group_is_underperforming_group_issue"),
                                      "name": "Underperforming group", "dataset_level": True}}

        # CSV Boolean conversion
        def as_bool(value):
            if pd.isna(value):
                return False
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return bool(value)
            return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
        active_issues = {}
        for issue_name, specification in issue_definitions.items():
            if (specification["dataset_level"] and not include_dataset_level_issues):
                continue
            if specification["flag"] in review_df.columns:
                active_issues[issue_name] = specification
        if not active_issues:
            raise RuntimeError("No Cleanlab issue columns were found in the CSV.")

        # Keep only rows with at least one issue
        issue_mask = np.zeros(len(review_df), dtype=bool)
        for specification in active_issues.values():
            issue_mask |= (review_df[specification["flag"]].map(as_bool).to_numpy())
        flagged_df = review_df.loc[issue_mask].copy()

        def dataframe_projection_key(row):
            try:
                projection_idx = str(int(float(row["projection_idx"])))
            except (TypeError, ValueError):
                projection_idx = str(row["projection_idx"]).strip()
            return f"{str(row['dicom_instance']).strip()}_{projection_idx}"

        flagged_df["_pdf_projection_key"] = flagged_df.apply(dataframe_projection_key, axis=1)
        excluded_from_pdf = flagged_df["_pdf_projection_key"].isin(removable_instances)
        print("Flagged instances excluded from PDF:", int(excluded_from_pdf.sum()))
        flagged_df = flagged_df.loc[~excluded_from_pdf].copy()
        flagged_df = flagged_df.drop(columns=["_pdf_projection_key"])
        if flagged_df.empty:
            raise RuntimeError("No instances with Cleanlab issues were found.")

        # Highest-priority instances first
        sort_columns = []
        ascending = []
        if "manual_review_priority" in flagged_df.columns:
            sort_columns.append("manual_review_priority")
            ascending.append(False)
        if "pointwise_issue_count" in flagged_df.columns:
            sort_columns.append("pointwise_issue_count")
            ascending.append(False)
        if ("probability_assigned_to_original_label" in flagged_df.columns):
            sort_columns.append("probability_assigned_to_original_label")
            ascending.append(True)
        if sort_columns:

            flagged_df = flagged_df.sort_values(sort_columns, ascending=ascending, na_position="last")
        if max_pages is not None:
            flagged_df = flagged_df.head(int(max_pages))

        # Projection identifier lookup
        if not hasattr(self.train_data, "dicom_projection_instances"):
            raise AttributeError("self.train_data does not contain 'dicom_projection_instances'.")
        projection_instances = [str(projection_identifier) for projection_identifier in self.train_data.dicom_projection_instances]
        projection_lookup = {}
        duplicate_identifiers = []
        for dataset_index, projection_identifier in enumerate(projection_instances):
            if projection_identifier in projection_lookup:
                duplicate_identifiers.append(projection_identifier)
                continue
            projection_lookup[projection_identifier] = dataset_index
        if duplicate_identifiers:
            print("Warning: duplicate identifiers found in dicom_projection_instances. The first occurrence will be used.")
            print("First duplicates:", duplicate_identifiers[:20])

        # Helpers
        def parse_index_list(value):
            if value is None:
                return []
            if not isinstance(value, (list, tuple, np.ndarray)):
                try:
                    if pd.isna(value):
                        return []
                except (TypeError, ValueError):
                    pass
            if isinstance(value, (list, tuple, np.ndarray)):
                parsed = value
            else:
                value = str(value).strip()
                if not value or value.lower() in {"nan", "none", "[]"}:
                    return []
                try:
                    parsed = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    try:
                        parsed = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        return []
            indices = []

            def flatten_indices(current_value):
                if isinstance(current_value, (list, tuple, np.ndarray)):
                    for nested_value in current_value:
                        flatten_indices(nested_value)
                    return
                try:
                    indices.append(int(current_value))
                except (TypeError, ValueError):
                    pass
            flatten_indices(parsed)
            return list(dict.fromkeys(indices))

        def get_near_duplicate_keys(row=None, current_projection_key=None):
            column = "near_duplicate_near_duplicate_sets"
            if column not in row.index:
                return []
            duplicate_indices = parse_index_list(row[column])
            duplicate_keys = []
            for duplicate_index in duplicate_indices:
                if 0 <= duplicate_index < len(projection_instances):
                    duplicate_key = projection_instances[duplicate_index]

                    # Do not report the current projection as its own duplicate
                    if duplicate_key != current_projection_key:
                        duplicate_keys.append(duplicate_key)
                else:
                    duplicate_keys.append(f"invalid-index-{duplicate_index}")
            return list(dict.fromkeys(duplicate_keys))

        def clean_projection_index(value):
            try:
                return str(int(float(value)))
            except (TypeError, ValueError):
                return str(value).strip()

        def get_projection_key(row):
            if ("dicom_projection_instance" in row.index and pd.notna(row["dicom_projection_instance"])):
                return str(row["dicom_projection_instance"]).strip()
            if ("dicom_instance" not in row.index or "projection_idx" not in row.index):
                raise KeyError("The CSV must contain either 'dicom_projection_instance' or both 'dicom_instance' and 'projection_idx'.")
            return (f"{str(row['dicom_instance']).strip()}_{clean_projection_index(row['projection_idx'])}")

        def class_description(value):
            if pd.isna(value):
                return "unknown"
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                return str(value)
            return {0: "no fracture", 1: "fracture"}.get(value, str(value))

        def get_image(dataset_index):
            x, _ = self.train_data.__getitem__(dataset_index)
            if len(x) == 0:
                raise RuntimeError(f"Empty projection data at index {dataset_index}.")
            image = x[0][1]
            if torch.is_tensor(image):
                image = image.detach().cpu().numpy()
            else:
                image = np.asarray(image)
            image = np.squeeze(image)

            # Convert channel-first images into channel-last format
            if (image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)):
                image = np.moveaxis(image, 0, -1)
            if image.ndim not in (2, 3):
                raise RuntimeError(f"Unsupported image shape at index {dataset_index}: {image.shape}")
            return image

        def get_problem_list(row, current_projection_key):
            problems = []
            for issue_name, specification in active_issues.items():
                flag_column = specification["flag"]
                if not as_bool(row.get(flag_column, False)):
                    continue
                if issue_name == "near_duplicate":
                    duplicate_keys = get_near_duplicate_keys(row=row, current_projection_key=current_projection_key)
                    if duplicate_keys:
                        problems.append("Near duplicate of: " + ", ".join(duplicate_keys))
                    else:
                        problems.append("Near duplicate")
                else:
                    problems.append(specification["name"])
            return problems

        # Generate PDF
        skipped_instances = []
        number_of_pages = 0
        with PdfPages(output_pdf_path) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = "Cleanlab Manual Review"
            metadata["Subject"] = ("Flagged X-ray projections from Cleanlab")
            metadata["Author"] = "NetworkTrainer"
            for _, row in flagged_df.iterrows():
                try:
                    projection_key = get_projection_key(row)
                except Exception as exc:
                    skipped_instances.append(f"Unknown key: {exc}")
                    continue
                if projection_key not in projection_lookup:
                    skipped_instances.append(projection_key)
                    continue
                dataset_index = projection_lookup[projection_key]
                try:
                    image = get_image(dataset_index)
                except Exception as exc:
                    skipped_instances.append(f"{projection_key}: {exc}")
                    continue
                problems = get_problem_list(row=row, current_projection_key=projection_key)
                if not problems:
                    continue

                # Classes and prediction probabilities
                true_class = class_description(row.get("original_label", np.nan))
                predicted_class = class_description(row.get("oof_predicted_label", np.nan))
                probability_no_fracture = row.get("probability_no_fracture", np.nan)
                probability_fracture = row.get("probability_fracture", np.nan)
                probability_parts = []
                if pd.notna(probability_no_fracture):
                    probability_parts.append(f"P(no fracture)={float(probability_no_fracture):.3f}")
                if pd.notna(probability_fracture):
                    probability_parts.append(f"P(fracture)={float(probability_fracture):.3f}")
                probability_text = ", ".join(probability_parts)
                prediction_text = f"Predicted: {predicted_class}"
                if probability_text:
                    prediction_text += f" ({probability_text})"
                title_information = f"True: {true_class} | {prediction_text}"
                wrapped_title_information = textwrap.wrap(title_information, width=82, break_long_words=False, break_on_hyphens=False)
                title_lines = [projection_key, *wrapped_title_information]
                title_text = "\n".join(title_lines)
                problem_lines = []
                for problem in problems:
                    wrapped_problem = textwrap.wrap(f"- {problem}", width=100, subsequent_indent="  ", break_long_words=False, break_on_hyphens=False)
                    problem_lines.extend(wrapped_problem)
                problems_text = "\n".join(problem_lines)

                # Fixed page width and dynamic page height
                image_height_px, image_width_px = image.shape[:2]
                figure_width = 8.27
                left_margin = 0.50
                right_margin = 0.50
                top_margin = 0.25
                bottom_margin = 0.25
                gap_after_title = 0.08
                gap_after_problems = 0.12
                available_image_width = figure_width - left_margin - right_margin
                maximum_image_height = 7.0
                scale = min(available_image_width / image_width_px, maximum_image_height / image_height_px)
                display_width = image_width_px * scale
                display_height = image_height_px * scale
                title_block_height = max(0.65, 0.30 * len(title_lines) + 0.10)
                problem_block_height = max(0.35, 0.23 * len(problem_lines) + 0.10)
                figure_height = (top_margin + title_block_height + gap_after_title + problem_block_height +
                                 gap_after_problems + display_height + bottom_margin)
                figure = plt.figure(figsize=(figure_width, figure_height))

                # Title and description
                title_y = 1.0 - top_margin / figure_height
                figure.text(0.5, title_y, title_text, ha="center", va="top", fontsize=11, fontweight="bold", linespacing=1.15)
                problems_y = 1.0 - (top_margin + title_block_height + gap_after_title) / figure_height
                figure.text(left_margin / figure_width, problems_y, problems_text, ha="left", va="top", fontsize=9.5,
                            fontweight="normal", linespacing=1.20)

                # Show image
                image_left = (figure_width - display_width) / (2.0 * figure_width)
                image_bottom = (bottom_margin / figure_height)
                image_axis = figure.add_axes([image_left, image_bottom, display_width / figure_width,
                                              display_height / figure_height])
                if image.ndim == 2:
                    image_axis.imshow(image, cmap="gray")
                else:
                    image_axis.imshow(image)
                image_axis.axis("off")
                pdf.savefig(figure)
                plt.close(figure)
                number_of_pages += 1
        if number_of_pages == 0:
            if os.path.exists(output_pdf_path):
                os.remove(output_pdf_path)
            raise RuntimeError("No PDF pages were created. Check that the CSV identifiers match self.train_data.dicom_projection_instances.")

        # Save unmatched identifiers
        if skipped_instances:
            skipped_path = os.path.splitext(output_pdf_path)[0] + "_skipped.csv"
            DataFrame({"skipped_instance": skipped_instances}).to_csv(skipped_path, index=False)
            print(f"Skipped {len(skipped_instances)} instances.")
            print("Skipped-instance list:", skipped_path)
        print("Manual-review PDF created:", output_pdf_path)
        print("PDF pages:", number_of_pages)

        return number_of_pages, skipped_instances


# Main
if __name__ == "__main__":
    # Define seed
    seed1 = 111099
    NetworkTrainer.set_seed(seed1)

    # Define variables
    # working_dir1 = "./../../"
    working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "cropped_projection_resnext50_simpler"
    net_type1 = NetType.BASE_RES_NEXT50
    epochs1 = 200
    preprocess_inputs1 = False
    trial_n1 = 16
    val_epochs1 = 10
    use_cuda1 = True
    assess_calibration1 = True
    show_test1 = True
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None
    enhance_images1 = False
    full_size1 = False
    is_cropped1 = True
    weight_loss1 = False
    dynamic_under_sampling1 = True
    transpose1 = False
    equalize_images1 = False

    # Load data
    addon = "" if not is_cropped1 else "cropped_"
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon + "xray_dataset_training",
                                           selected_segments=selected_segments1,
                                           selected_projection=selected_projection1,
                                           removable_instances_txt="removable_instances_training.txt")
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon + "xray_dataset_validation",
                                         selected_segments=selected_segments1, selected_projection=selected_projection1)
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=addon + "xray_dataset_test",
                                          selected_segments=selected_segments1,
                                          selected_projection=selected_projection1)
    # bicocca_only_dataset = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="DD_bicocca")

    # Define trainer
    net_params1 = {"n_conv_segment_neurons": 0, "n_conv_view_neurons": 1024, "n_conv_segment_layers": 0,
                   "n_conv_view_layers": 0, "kernel_size": 7, "n_fc_layers": 4, "optimizer": "SGD",
                   "lr_last": 0.00001, "lr_second_last_factor": 71, "batch_size": 32, "p_dropout": 0.9,
                   "use_batch_norm": False}
    trainer1 = NetworkTrainer(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                              val_data=val_data1, test_data=test_data1, net_type=net_type1, epochs=epochs1,
                              val_epochs=val_epochs1, preprocess_inputs=preprocess_inputs1, net_params=net_params1,
                              use_cuda=use_cuda1, projection_dataset=projection_dataset1, enhance_images=enhance_images1,
                              full_size=full_size1, is_cropped=is_cropped1, weight_loss=weight_loss1,
                              dynamic_under_sampling=dynamic_under_sampling1, equalize_images=equalize_images1)

    # Test zero-shot Bicocca model
    # trainer1.summarize_performance(show_test=False, show_process=False, show_cm=False, assess_calibration=False,
    #                                zero_shot=True)

    # Train model
    # trainer1.train(show_epochs=True)
    # trainer1.summarize_performance(show_test=show_test1, show_process=True, show_cm=True,
    #                                assess_calibration=assess_calibration1)
    
    # Evaluate model
    print()
    trainer1 = NetworkTrainer.load_model(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1,
                                         use_cuda=use_cuda1, train_data=train_data1, val_data=val_data1,
                                         test_data=test_data1, projection_dataset=projection_dataset1,
                                         is_cropped=is_cropped1)
    '''trainer1.summarize_performance(show_test=show_test1, show_process=True, show_cm=True,
                                   assess_calibration=assess_calibration1)'''

    # Clean labels with Cleanlab
    dataset1 = trainer1.val_data
    records = dataset1.clean_labels_input()
    lab, review_table, oof_pred_probs, features, issue_summary = trainer1.clean_labels(records=records,
                                                                                       dataset=dataset1, k=10,
                                                                                       clean_epochs=epochs1, seed=seed1,
                                                                                       show=True)
    number_of_pages, skipped = trainer1.create_cleanlab_review_pdf(removable_instances_path=dataset1.data_dir +
                                                                                            "/removable_instances_" +
                                                                                            dataset1.set_type.value + ".txt")
