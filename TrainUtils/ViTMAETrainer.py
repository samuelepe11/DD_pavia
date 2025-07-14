# Import packages
import numpy as np
import torch
import torch.nn as nn
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import optuna
from PIL import Image
from transformers import ViTImageProcessor, ViTMAEForPreTraining, Trainer, TrainingArguments

from DataUtils.XrayDataset import XrayDataset
from Enumerators.NetType import NetType
from Enumerators.SetType import SetType
from TrainUtils.NetworkTrainer import NetworkTrainer
from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class ViTMAETrainer(NetworkTrainer):
    pretrained_model_name = "facebook/vit-mae-base"

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, decoder_net_type, epochs, val_epochs,
                 convergence_patience=5, convergence_thresh=1e-3, decoder_net_params=None, use_cuda=True,
                 projection_dataset=False, enhance_images=True):
        super().__init__(model_name, working_dir, train_data, val_data, test_data, net_type=decoder_net_type,
                         epochs=epochs, val_epochs=val_epochs, convergence_patience=convergence_patience,
                         convergence_thresh=convergence_thresh, preprocess_inputs=False, net_params=decoder_net_params,
                         use_cuda=use_cuda, s3=None, n_parallel_gpu=0, projection_dataset=projection_dataset,
                         enhance_images=enhance_images)

        self.img_preprocessor = ViTImageProcessor.from_pretrained(self.pretrained_model_name)
        self.inference_data_transforms = transforms.Compose([
            transforms.Resize((self.img_preprocessor.size["height"], self.img_preprocessor.size["width"])),
        ])

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.net_pretrain = ViTMAEForPreTraining.from_pretrained(self.pretrained_model_name, attn_implementation="sdpa")

        # Training parameters
        self.criterion_pretrain = nn.MSELoss()
        self.optimizer_pretrain = torch.optim.AdamW(self.net_pretrain.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                                    weight_decay=0.05)

    def preprocess_img(self, img_batch, do_normalise=True):
        if do_normalise:
            img_batch = self.img_preprocessor(img_batch, return_tensors="pt", do_rescale=True, do_resize=True).pixel_values
        if self.use_cuda:
            img_batch = img_batch.to("cuda")
        return img_batch

    def reconstruct_img(self, img_batch, do_normalise=True, net=None):
        net = self.net_pretrain if net is None else net
        outputs = net(self.preprocess_img(img_batch=img_batch, do_normalise=do_normalise))
        return outputs

    def remove_patching(self, tensor):
        tensor = self.net_pretrain.unpatchify(tensor)
        tensor = torch.einsum("bchw->bhwc", tensor).detach().cpu()
        return tensor

    def visualise_reconstruction(self, img_batch, do_normalise_input=True):
        for img in img_batch:
            # Get reconstruction
            img = self.inference_data_transforms(img)
            img = img.unsqueeze(0)
            outputs = self.reconstruct_img(img, do_normalise=do_normalise_input)
            y = self.remove_patching(outputs.logits)

            # Get mask
            mask = outputs.mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.net_pretrain.config.patch_size ** 2 * 3)
            mask = self.remove_patching(mask)

            # Adjust input
            x = torch.einsum("bchw->bhwc", img)
            img_masked = x * (1 - mask)

            # MAE reconstruction pasted with visible patches
            img_reconstructed = x * (1 - mask) + y * mask

            # Display result
            plt.rcParams["figure.figsize"] = [24, 24]
            plt.subplot(1, 4, 1)
            self.show_img(x[0], "Original image")
            plt.subplot(1, 4, 2)
            self.show_img(img_masked[0], "Masked image")
            plt.subplot(1, 4, 3)
            self.show_img(y[0], "Reconstructed patches")
            plt.subplot(1, 4, 4)
            self.show_img(img_reconstructed[0], "Final result")
            plt.show()

    def pretrain(self, show_epochs=False, trial_n=None, trial=None, double_output=False):
        if show_epochs:
            self.start_time = time.time()

        net = self.net_pretrain
        if self.n_parallel_gpu:
            net = nn.DataParallel(net)

        if self.use_cuda:
            self.criterion = self.criterion.cuda()
            net = net.cuda()
        net.to(self.device)

        if len(self.train_losses) == 0 and trial is None:
            print("\nPerforming initial evaluation...")
            _, val_loss = self.summarize_performance_pretrain(show_test=False, show_process=False)

        if show_epochs:
            print("\nStarting the training phase...")
        for epoch in range(self.epochs):
            net.train()

            train_loss = 0
            for batch in self.train_loader:
                self.optimizer_pretrain.zero_grad()
                loss = self.apply_network_pretrain(net, batch, is_training=True)
                train_loss += loss.item()

                loss.backward()
                self.optimizer_pretrain.step()

            train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            if show_epochs and (epoch % 10 == 0 or epoch % self.val_epochs == 0):
                print()
                print("Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " completed...")
                print(" > train loss = " + str(np.round(train_loss, 5)))

            if epoch % self.val_epochs == 0:
                val_loss = self.test_pretrain(set_type=SetType.VAL)
                self.val_losses.append(val_loss)
                self.val_eval_epochs.append(epoch)
                if show_epochs:
                    print(" > val loss = " + str(np.round(val_loss, 5)))

                    # Update and store training curves
                    if epoch != 0:
                        plt.close()
                        self.draw_training_curves(is_pretrain=True)
                        if trial_n is None:
                            filepath = self.results_dir + "pretraining_curves.jpg"
                        else:
                            filepath = self.results_dir + "trial_" + str(trial_n - 1) + "_pretraining_curves.jpg"
                        if self.s3 is not None:
                            filepath = self.s3.open(filepath, "wb")
                        plt.savefig(filepath)

                        plt.close()
                        self.draw_training_curves(is_pretrain=True)
                        plt.show()

                    # Store intermediate result
                    if trial_n is None:
                        self.save_model()

                if trial is not None and not double_output:
                    trial.report(val_loss, epoch)
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
            train_loss, val_loss = self.summarize_performance_pretrain(show_test=False, show_process=False,
                                                                       trial_n=trial_n)
            if double_output:
                return val_loss, train_loss
            else:
                return val_loss

    def apply_network_pretrain(self, net, instance_batch, is_training=False):
        proj_batch = []
        item, _ = instance_batch
        _, projection_batch, _ = item
        for i in range(projection_batch.shape[0]):
            proj = projection_batch[i]
            proj = torch.tensor(proj).float()
            proj = torch.concat([proj] * 3, dim=0)
            if is_training:
                proj = ConvBaseNetwork.training_data_transforms(proj)
            proj = self.inference_data_transforms(proj)
            proj -= torch.min(proj)
            proj /= (torch.max(proj) - torch.min(proj) + 1e-8)
            proj_batch.append(proj)
        proj_batch = torch.stack(proj_batch, dim=0)
        if self.use_cuda:
            proj_batch = proj_batch.cuda()

        # Loss evaluation
        outputs = self.reconstruct_img(proj_batch, do_normalise=False, net=net)
        loss = outputs.loss

        return loss

    def test_pretrain(self, set_type=SetType.TRAIN):
        net = self.net_pretrain
        if self.n_parallel_gpu:
            net = nn.DataParallel(net)

        if self.use_cuda:
            self.criterion = self.criterion.cuda()
            net = net.cuda()
        net = net.to(self.device)
        net.eval()

        data, loader, _ = self.select_dataset(set_type)

        # Store class labels
        loss = 0
        with torch.no_grad():
            for batch in loader:
                temp_loss = self.apply_network_pretrain(net, batch)
                loss += temp_loss.item()
            loss /= len(loader)

        return loss

    def summarize_performance_pretrain(self, show_test=False, show_process=False, trial_n=None):
        # Show final losses
        train_loss = self.test_pretrain(set_type=SetType.TRAIN)
        print("Training loss = " + str(np.round(train_loss, 5)))

        val_loss = self.test_pretrain(set_type=SetType.VAL)
        print("Validation loss = " + str(np.round(val_loss, 5)))

        if show_test:
            test_loss = self.test_pretrain(set_type=SetType.TEST)
            print("Test loss = " + str(np.round(test_loss, 5)))

        if (show_process or trial_n is not None) and len(self.train_losses) != 0:
            self.draw_training_curves(is_pretrain=True)
            if show_process:
                filepath = self.results_dir + "pretraining_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()
            if trial_n is not None:
                filepath = self.results_dir + "trial_" + str(trial_n) + "_pretraining_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()

        return train_loss, val_loss

    @staticmethod
    def show_img(img, title):
        assert img.shape[2] == 3
        plt.imshow(torch.clip(img * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis("off")


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    # working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "vitmae"
    decoder_net_type1 = NetType.BASE_RES_NEXT101
    epochs1 = 100
    preprocess_inputs1 = False
    trial_n1 = None
    val_epochs1 = 10
    use_cuda1 = True
    assess_calibration1 = False
    show_test1 = False
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None
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

    # Define trainer
    decoder_net_params1 = {"n_conv_segment_neurons": 512, "n_conv_view_neurons": 512, "n_conv_segment_layers": 1,
                           "n_conv_view_layers": 1, "kernel_size": 3, "n_fc_layers": 1, "optimizer": "Adam",
                           "lr_last": 0.00001, "lr_second_last_factor": 10, "batch_size": 8, "p_dropout": 0,
                           "use_batch_norm": False}
    trainer1 = ViTMAETrainer(model_name=model_name1, working_dir=working_dir1, train_data=val_data1,
                             val_data=val_data1, test_data=val_data1, decoder_net_type=decoder_net_type1, epochs=epochs1,
                             val_epochs=val_epochs1, decoder_net_params=decoder_net_params1, use_cuda=use_cuda1,
                             projection_dataset=projection_dataset1, enhance_images=enhance_images1)

    # Apply on an example image
    url1 = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"
    image1 = Image.open(requests.get(url1, stream=True).raw)
    image1 = transforms.ToTensor()(image1)
    image_batch1 = image1.unsqueeze(0)
    # trainer1.visualise_reconstruction(image_batch1, do_normalise_input=False)

    # Apply on an X-ray image
    indices1 = list(range(3))
    proj_batch1 = []
    for i1 in indices1:
        segm, extra = train_data1[i1]
        proj1 = segm[0][1]
        proj1 = np.stack([proj1] * 3, axis=0)
        proj1 = torch.tensor(proj1).float()
        proj1 -= torch.min(proj1)
        proj1 /= (torch.max(proj1) - torch.min(proj1))
        proj_batch1.append(transforms.Resize((trainer1.img_preprocessor.size["height"],
                                             trainer1.img_preprocessor.size["width"]))(proj1))
    proj_batch1 = torch.stack(proj_batch1, 0)
    # trainer1.visualise_reconstruction(proj_batch1, do_normalise_input=False)

    # Pretrain model
    trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)
    trainer1.pretrain(show_epochs=True)
    trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)
