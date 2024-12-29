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
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import roc_auc_score
from pandas import DataFrame
from functools import partial

from Networks.AttentionViT import AttentionViT
from calfram.calibrationframework import select_probability, reliabilityplot, calibrationdiagnosis, classwise_calibration

from DataUtils.XrayDataset import XrayDataset
from DataUtils.Preprocessor import Preprocessor
from Enumerators.NetType import NetType
from Enumerators.SetType import SetType
from Networks.BaseResNeXt50 import BaseResNeXt50
from Networks.BaseResNeXt101 import BaseResNeXt101
from Networks.BaseViT import BaseViT
from TrainUtils.StatsHolder import StatsHolder


# Class
class NetworkTrainer:

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, net_type, epochs, val_epochs,
                 convergence_patience=3, convergence_thresh=1e-3, preprocess_inputs=False, net_params=None,
                 use_cuda=True, s3=None):
        # Initialize attributes
        self.model_name = model_name
        self.working_dir = working_dir
        self.results_dir = working_dir + XrayDataset.results_fold + XrayDataset.models_fold
        self.s3 = s3
        if s3 is None:
            if model_name not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + model_name)
        else:
            if not s3.exists(self.results_dir + model_name):
                s3.touch(self.results_dir + model_name + "/empty.txt")
        self.results_dir += model_name + "/"

        self.use_cuda = torch.cuda.is_available() and use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.net_type = net_type
        self.net_params = net_params
        if net_type == NetType.BASE_RES_NEXT50:
            self.net = BaseResNeXt50(params=net_params, device=self.device)
        elif net_type == NetType.BASE_RES_NEXT101:
            self.net = BaseResNeXt101(params=net_params, device=self.device)
        elif net_type == NetType.BASE_VIT:
            self.net = BaseViT(params=net_params, device=self.device)
        elif net_type == NetType.ATTENTION_VIT:
            self.net = AttentionViT(params=net_params, device=self.device)
        else:
            self.net = None
            print("The selected network is not available.")

        # Define training parameters
        self.classes = XrayDataset.classes
        self.epochs = epochs
        self.val_epochs = val_epochs
        self.criterion = nn.BCELoss()
        param_groups = [{"params": [param for name, param in self.net.named_parameters() if "resnet.7" in name],
                         "lr": self.net_params["lr_last"] / self.net_params["lr_second_last_factor"]},
                        {"params":  [param for name, param in self.net.named_parameters() if "resnet.7" not in name],
                         "lr": self.net_params["lr_last"]}]
        self.optimizer = getattr(torch.optim, net_params["optimizer"])(param_groups)
        self.convergence_thresh = convergence_thresh
        self.convergence_patience = convergence_patience
        self.preprocess_inputs = preprocess_inputs

        self.start_time = None
        self.end_time = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_eval_epochs = []
        self.optuna_study = None

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

    def load_data(self, data, shuffle=False):
        dataloader = DataLoader(dataset=data, batch_size=self.net_params["batch_size"], shuffle=shuffle, num_workers=0,
                                collate_fn=partial(NetworkTrainer.custom_collate_fn, img_dim=self.net.input_dim))
        dim = len(data)

        return dataloader, dim

    def train(self, show_epochs=False, trial_n=None, trial=None):
        if show_epochs:
            self.start_time = time.time()

        self.net.set_cuda(cuda=self.use_cuda)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        net = self.net

        if len(self.train_losses) == 0 and trial is None:
            print("\nPerforming initial evaluation...")
            _, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False)

        if show_epochs:
            print("\nStarting the training phase...")
        for epoch in range(self.epochs):
            net.set_training(True)
            train_loss = 0
            train_acc = 0
            for batch in self.train_loader:
                loss, _, _, acc = self.apply_network(net, batch, set_type=SetType.TRAIN)
                train_loss += loss.item()
                train_acc += acc.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(self.train_loader)
            train_acc = train_acc / len(self.train_loader)
            self.train_losses.append(train_loss)
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
                    self.save_model(trial_n)

                if trial is not None:
                    trial.report(val_stats.f1, epoch)
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

        self.save_model(trial_n)
        if trial_n is not None:
            _, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False,
                                                      trial_n=trial_n)
            return val_stats.f1

    def apply_network(self, net, instance, set_type):
        item, extra = instance
        y = (item[-1] != "").astype(int)
        y = torch.tensor(np.array(y)).to(torch.float32)
        y = y.unsqueeze(1)
        y = y.to(self.device)

        projection_type_batch, projection_batch, _ = item
        adjusted_projection = []
        for i in range(projection_type_batch.shape[0]):
            projection_list = np.split(projection_batch[i], projection_batch.shape[1], axis=0)
            projection_list = [x[0] for x in projection_list]
            projections = self.preprocessor.mask_projection(projection_list, (extra[0][i], extra[1][i]),
                                                            set_type=set_type, preprocess=self.preprocess_inputs)
            temp = []
            for j in range(len(projections)):
                projection = projections[j]
                projection = self.preprocessor.preprocess(img=projection, segm=extra[1], downsampling_iterates=0,
                                                          show=False)
                projection = torch.tensor(projection, dtype=torch.float32)
                temp.append(projection)
            adjusted_projection.append(temp)

        # Loss evaluation
        input = np.stack(adjusted_projection)
        input = torch.from_numpy(input)
        output = net(input, extra[1], projection_type_batch)
        loss = self.criterion(output, y)

        # Accuracy evaluation
        pred = (output > 0.5).to(torch.int)
        acc = torch.sum(pred == y) / y.shape[0]

        return loss, output, y, acc

    def select_dataset(self, set_type):
        if self.train_data is None:
            # Read datasets
            self.train_data = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training", s3=self.s3)
            self.val_data = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation", s3=self.s3)
            self.test_data = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test", s3=self.s3)

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

    def compute_stats(self, y_true, y_pred, loss, acc, y_prob=None):
        n_vals = len(y_true)
        if loss is None:
            if y_prob is not None:
                loss = self.criterion(y_prob, y_true)
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

    def test(self, set_type=SetType.TRAIN, show_cm=False, assess_calibration=False):
        self.net.set_cuda(cuda=self.use_cuda)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        net = self.net
        _, loader, _ = self.select_dataset(set_type)

        # Store class labels
        y_prob = []
        y_true = []
        y_pred = []
        loss = 0
        net.set_training(False)
        with torch.no_grad():
            for batch in loader:
                temp_loss, output, y, _ = self.apply_network(net, batch, set_type=set_type)
                loss += temp_loss.item()

                # Accuracy evaluation
                if output.shape[-1] == 1:
                    output = torch.cat((1 - output, output), dim=-1)
                prediction = torch.argmax(output, dim=1)

                y_prob.append(output.cpu())
                y_true.append(y.cpu())
                y_pred.append(prediction.cpu())

            y_prob = torch.cat(y_prob, dim=0)
            y_true = torch.cat(y_true).squeeze(1).to(int)
            y_pred = torch.cat(y_pred).to(int)

            loss /= len(loader)
            acc = torch.sum(y_true == y_pred) / len(y_true)
            acc = acc.item()
        stats_holder = self.compute_stats(y_true, y_pred, loss, acc)

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
            stats_holder.calibration_results = self.assess_calibration(y_true, y_prob, y_pred, set_type)

        return stats_holder

    def assess_calibration(self, y_true, y_prob, y_pred, set_type):
        y_true = y_true.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        class_scores = select_probability(y_true, y_prob, y_pred)

        # Store results file
        data = np.concatenate((y_true[:, np.newaxis], y_pred[:, np.newaxis], y_prob), axis=1)
        titles = ["y_true", "y_pred"] + ["y_prob" + str(i) for i in range(y_prob.shape[1])]
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
            addon = "trial_" + str(trial_n - 1)
            
        filepath = self.results_dir + addon + ".pt"
        if self.s3 is not None:
            filepath = self.s3.open(filepath, "wb")
        torch.save({"net_type": self.net_type, "epochs": self.epochs, "val_epochs": self.val_epochs,
                    "preprocess_inputs": self.preprocess_inputs, "net_params": self.net_params,
                    "model_state_dict": self.net.state_dict()}, filepath)

        print("'" + self.model_name + "' has been successfully saved!... train loss: " +
              str(np.round(self.train_losses[0], 4)) + " -> " + str(np.round(self.train_losses[-1], 4)))
        print()

    def summarize_performance(self, show_test=False, show_process=False, show_cm=False, trial_n=None,
                              assess_calibration=False):
        # Show final losses
        train_stats = self.test(set_type=SetType.TRAIN, show_cm=show_cm, assess_calibration=assess_calibration)
        print("Training loss = " + str(np.round(train_stats.loss, 5)) + " - Training accuracy = " +
              str(np.round(train_stats.acc * 100, 7)) + "% - Training F1-score = " +
              str(np.round(train_stats.f1 * 100, 7)) + "%")

        NetworkTrainer.show_performance_table(train_stats, "training")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(train_stats, "training")

        print()
        val_stats = self.test(set_type=SetType.VAL, show_cm=show_cm, assess_calibration=assess_calibration)
        print("Validation loss = " + str(np.round(val_stats.loss, 5)) + " - Validation accuracy = " +
              str(np.round(val_stats.acc * 100, 7)) + "% - Validation F1-score = " +
              str(np.round(val_stats.f1 * 100, 7)))

        NetworkTrainer.show_performance_table(val_stats, "validation")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(val_stats, "validation")

        if show_test:
            print()
            test_stats = self.test(set_type=SetType.TEST, show_cm=show_cm, assess_calibration=assess_calibration)
            print("Test loss = " + str(np.round(test_stats.loss, 5)) + " - Test accuracy = " +
                  str(np.round(test_stats.acc * 100, 7)) + "% - Test F1-score = " +
                  str(np.round(test_stats.f1 * 100, 7)))

            NetworkTrainer.show_performance_table(test_stats, "test")
            if assess_calibration:
                NetworkTrainer.show_calibration_table(test_stats, "test")

        if show_process or trial_n is not None:
            self.draw_training_curves()
            if show_process:
                filepath = self.results_dir + "training_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()
            if trial_n is not None:
                filepath = self.results_dir + "trial_" + str(trial_n - 1) + "_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()

        return train_stats, val_stats

    def draw_training_curves(self):
        plt.close()
        plt.figure(figsize=(5, 7))
        plt.suptitle("Training curves")

        # Losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, "b", label="Training set")
        plt.plot(self.val_eval_epochs, self.val_losses, "g", label="Validation set")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

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

    @staticmethod
    def custom_collate_fn(batch, img_dim):
        segment_datas = []
        pt_ids = []
        segment_ids = []
        for segment_data, extra_info in batch:
            pt_id, segment_id = extra_info
            segment_datas.append(segment_data)
            pt_ids.append(pt_id)
            segment_ids.append(segment_id)

        # Resize images to the same dimension and patch channels in the batch
        max_proj_num = np.max([len(x) for x in segment_datas])
        segment_datas_patched = []
        for segment_data in segment_datas:
            temp = []
            for proj in segment_data:
                proj_id = proj[0]
                img = proj[1]
                descr = proj[2]
                temp.append((proj_id, cv2.resize(img, (img_dim, img_dim)), descr))
            extra_proj = max_proj_num - len(segment_data)
            if extra_proj > 0:
                for _ in range(extra_proj):
                    temp.append((proj_id, np.zeros((img_dim, img_dim)), descr))
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
        segment_datas = (np.stack(batch_proj_id, dtype=object), np.stack(batch_img), np.stack(batch_descr, dtype=object))

        return segment_datas, (pt_ids, segment_ids)

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
    def draw_multiclass_confusion_matrix(cm, labels, imgpath, s3=None):
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
                   s3=None):
        print("Loading " + model_name + "...")
        if trial_n is None:
            file_name = model_name
        else:
            file_name = "trial_" + str(trial_n)
        filepath = (working_dir + XrayDataset.results_fold + XrayDataset.models_fold + model_name + "/" +
                    file_name + ".pt")
        if s3 is not None:
            filepath = s3.open(filepath)
        checkpoint = torch.load(filepath)
        network_trainer = NetworkTrainer(model_name=model_name, working_dir=working_dir, train_data=train_data,
                                         val_data=val_data, test_data=test_data, net_type=checkpoint["net_type"],
                                         epochs=checkpoint["epochs"], val_epochs=checkpoint["val_epochs"],
                                         preprocess_inputs=checkpoint["preprocess_inputs"],
                                         net_params=checkpoint["net_params"], use_cuda=use_cuda, s3=s3)
        network_trainer.net.load_state_dict(checkpoint["model_state_dict"])

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


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    model_name1 = "atention_vit"
    net_type1 = NetType.ATTENTION_VIT
    epochs1 = 2
    preprocess_inputs1 = True
    trial_n1 = None
    val_epochs1 = 10
    use_cuda1 = True
    assess_calibration1 = True
    show_test1 = False

    # Load data
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training")
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation")
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test")

    # Define trainer
    net_params1 = {"n_conv_segment_neurons": 1024, "n_conv_view_neurons": 1024, "n_conv_segment_layers": 1,
                   "n_conv_view_layers": 1, "kernel_size": 3, "n_fc_layers": 1, "optimizer": "Adam",
                   "lr_last": 0.00001, "lr_second_last_factor": 10, "batch_size": 4, "p_dropout": 0,
                   "use_batch_norm": False}
    trainer1 = NetworkTrainer(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                              val_data=val_data1, test_data=test_data1, net_type=net_type1, epochs=epochs1,
                              val_epochs=val_epochs1, preprocess_inputs=preprocess_inputs1, net_params=net_params1,
                              use_cuda=use_cuda1)

    # Train model
    trainer1.train(show_epochs=True)
    trainer1.summarize_performance(show_test=show_test1, show_process=True, show_cm=True,
                                   assess_calibration=assess_calibration1)
    
    # Evaluate model
    print()
    trainer1 = NetworkTrainer.load_model(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1,
                                         use_cuda=use_cuda1, train_data=train_data1, val_data=val_data1,
                                         test_data=test_data1)
    trainer1.summarize_performance(show_test=show_test1, show_process=True, show_cm=True,
                                   assess_calibration=assess_calibration1)
