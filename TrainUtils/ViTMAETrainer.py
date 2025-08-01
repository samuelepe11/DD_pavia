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
from sympy.categories import Object
from tensorflow.python.ops.gen_nn_ops import l2_loss
from transformers import ViTImageProcessor, ViTMAEForPreTraining, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DataUtils.XrayDataset import XrayDataset
from Enumerators.NetType import NetType
from Enumerators.SetType import SetType
from TrainUtils.NetworkTrainer import NetworkTrainer
from Networks.ConvBaseNetwork import ConvBaseNetwork


# Class
class ViTMAETrainer(NetworkTrainer):
    pretrained_model_name = "facebook/vit-mae-base"
    default_net_params = {"n_conv_segment_neurons": 1, "n_conv_view_neurons": 1, "n_conv_segment_layers": 1,
                          "n_conv_view_layers": 1, "kernel_size": 1, "n_fc_layers": 1, "optimizer": "SGD",
                          "lr_last": 1, "lr_second_last_factor": 1, "batch_size": 1, "p_dropout": 1,
                          "use_batch_norm": False}
    default_net_type = NetType.BASE_RES_NEXT101

    def __init__(self, model_name, working_dir, train_data, val_data, test_data, decoder_net_type, epochs, val_epochs,
                 convergence_patience=10, convergence_thresh=1e-3, decoder_net_params=None, use_cuda=True,
                 projection_dataset=False, preprocess_inputs=False, enhance_images=False, train_parameters=None):
        super().__init__(model_name, working_dir, train_data, val_data, test_data, net_type=decoder_net_type,
                         epochs=epochs, val_epochs=val_epochs, convergence_patience=convergence_patience,
                         convergence_thresh=convergence_thresh, preprocess_inputs=preprocess_inputs, net_params=decoder_net_params,
                         use_cuda=use_cuda, s3=None, n_parallel_gpu=0, projection_dataset=projection_dataset,
                         enhance_images=enhance_images, full_size=False)

        self.img_preprocessor = ViTImageProcessor.from_pretrained(self.pretrained_model_name)
        self.inference_data_transforms = transforms.Compose([
            transforms.Normalize(mean=self.img_preprocessor.image_mean, std=self.img_preprocessor.image_std),
            transforms.Resize((self.img_preprocessor.size["height"], self.img_preprocessor.size["width"])),
        ])

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.net_pretrain = ViTMAEForPreTraining.from_pretrained(self.pretrained_model_name, attn_implementation="sdpa")

        # Training parameters
        self.train_parameters = train_parameters
        lr = train_parameters["base_lr"]
        l_decay = train_parameters["layer_decay"]
        if l_decay != 0:
            param_groups = self.get_layerwise_lr_decay_params(base_lr=lr, layer_decay=l_decay)
        else:
            param_groups = [{"params": self.net_pretrain.parameters(), "lr": lr}]
        self.optimizer_pretrain = torch.optim.AdamW(param_groups, betas=(train_parameters["beta1"],
                                                                         train_parameters["beta2"]),
                                                    eps=train_parameters["eps"])
        num_steps = len(self.train_loader) * self.epochs
        if train_parameters["scheduler"] == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer_pretrain,
                                                             num_warmup_steps=int(0.05 * num_steps),
                                                             num_training_steps=num_steps)
        elif train_parameters["scheduler"] == "reduce_lr_on_plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer_pretrain, min_lr=train_parameters["min_lr"])
        else:
            self.scheduler = None
        self.batch_size = train_parameters["batch_size"]
        self.starting_epoch = 0

    def preprocess_img(self, img_batch, do_normalise=True):
        if do_normalise:
            img_batch = self.img_preprocessor(img_batch, return_tensors="pt", do_rescale=True, do_resize=True).pixel_values
        if self.use_cuda:
            img_batch = img_batch.to("cuda")
        return img_batch

    def reconstruct_img(self, img_batch, do_normalise=True, net=None):
        net = self.net_pretrain if net is None else net
        if self.use_cuda:
            net = net.cuda()
        net = net.to(self.device)
        net.eval()

        outputs = net(self.preprocess_img(img_batch=img_batch, do_normalise=do_normalise))
        return outputs

    def remove_patching(self, tensor):
        tensor = self.net_pretrain.unpatchify(tensor)
        tensor = torch.einsum("bchw->bhwc", tensor).detach().cpu()
        return tensor

    def visualise_reconstruction(self, img_batch, do_normalise_input=True, title=None, store_img=False):
        for i, img in enumerate(img_batch):
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
            self.show_img(self.denormalize(x)[0], "Original image")
            plt.subplot(1, 4, 2)
            self.show_img(self.denormalize(img_masked)[0], "Masked image")
            plt.subplot(1, 4, 3)
            self.show_img(self.denormalize(y)[0], "Reconstructed patches")
            plt.subplot(1, 4, 4)
            self.show_img(self.denormalize(img_reconstructed)[0], "Final result")
            if not store_img:
                plt.show()
            else:
                addon = "_" + title.lower().replace(" ", "_") if title is not None else ""
                plt.savefig(self.results_dir + "ex" + str(i) + addon + ".png", dpi=500, bbox_inches="tight")
                plt.close()

    def denormalize(self, input):
        mean = torch.tensor(self.img_preprocessor.image_mean)
        std = torch.tensor(self.img_preprocessor.image_std)
        if input.shape[-1] == 3:
            mean = mean.view(1, 1, 1, 3)
            std = std.view(1, 1, 1, 3)
        else:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)

        return input * std + mean

    def pretrain(self, show_epochs=False, trial_n=None, trial=None, double_output=False, continue_training=False):
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
            _, _ = self.summarize_performance_pretrain(show_test=False, show_process=False)

        if continue_training:
            self.starting_epoch += len(self.train_losses)
        epoch_range = range(self.starting_epoch, self.starting_epoch + self.epochs)

        if show_epochs:
            print("\nStarting the training phase...")
        for epoch in epoch_range:
            net.train()

            train_loss = 0
            for batch in self.train_loader:
                self.optimizer_pretrain.zero_grad()
                loss = self.apply_network_pretrain(net, batch, is_training=True)
                train_loss += loss.item()

                loss.backward()
                self.optimizer_pretrain.step()
                if self.train_parameters["scheduler"] == "cosine":
                    self.scheduler.step()

            train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            if show_epochs and (epoch % 10 == 0 or epoch % self.val_epochs == 0):
                print()
                print("Epoch " + str(epoch + 1) + "/" + str(self.starting_epoch + self.epochs) + " completed...")
                print(" > train loss = " + str(np.round(train_loss, 5)))

            if epoch == self.starting_epoch or epoch % self.val_epochs == 0:
                val_loss = self.test_pretrain(set_type=SetType.VAL)
                self.val_losses.append(val_loss)
                self.val_eval_epochs.append(epoch)
                if show_epochs:
                    print(" > val loss = " + str(np.round(val_loss, 5)))

                if self.train_parameters["scheduler"] == "reduce_lr_on_plateau":
                    self.scheduler.step(val_loss)

                    # Update and store training curves
                    if epoch != self.starting_epoch:
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

    def apply_network_pretrain(self, net, instance_batch, is_training=False, set_type=SetType.TRAIN):
        proj_batch = []
        item, extra = instance_batch
        projection_type_batch, projection_batch, _ = item
        if self.preprocess_inputs:
            projection_batch = self.preprocess_fn(projection_batch, projection_type_batch, extra, set_type)
            projection_batch = projection_batch.numpy().astype(np.uint16)

        for i in range(projection_batch.shape[0]):
            proj = projection_batch[i]
            proj = torch.tensor(proj).float()
            proj = torch.concat([proj] * 3, dim=0)
            proj -= torch.min(proj)
            proj /= (torch.max(proj) - torch.min(proj) + 1e-8)

            if is_training:
                proj = ConvBaseNetwork.training_data_transforms(proj)
            proj = self.inference_data_transforms(proj)
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
                temp_loss = self.apply_network_pretrain(net, batch, set_type=set_type)
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

    def get_layerwise_lr_decay_params(self, base_lr=3e-5, layer_decay=0.75):
        param_groups = []
        num_layers = self.net_pretrain.config.num_hidden_layers
        lr_map = {}

        for name, param in self.net_pretrain.named_parameters():
            if not param.requires_grad:
                continue

            lr = base_lr
            if name.startswith("vit.encoder.layer"):
                layer_num = int(name.split(".")[3])
                lr = base_lr * (layer_decay ** (num_layers - 1 - layer_num))

            elif name.startswith("vit.embeddings"):
                # Embeddings usually get the lowest LR
                lr = base_lr * (layer_decay ** num_layers)

            param_groups.append({"params": [param], "lr": lr})
            lr_map[name] = lr
        return param_groups

    @staticmethod
    def show_img(img, title):
        assert img.shape[2] == 3
        plt.imshow(torch.clip(img * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis("off")

    @staticmethod
    def load_model(working_dir, model_name, trial_n=None, use_cuda=True, train_data=None, val_data=None, test_data=None,
                   projection_dataset=False, s3=None, batch_size=None, new_epochs=None):
        file_name = model_name if trial_n is None else "trial_" + str(trial_n)
        print("Loading " + file_name + "...")
        filepath = (working_dir + XrayDataset.results_fold + XrayDataset.models_fold + model_name + "/" +
                    file_name + ".pt")
        if s3 is not None:
            filepath = s3.open(filepath)
        checkpoint = torch.load(filepath, weights_only=False)
        network_trainer = ViTMAETrainer(model_name=model_name, working_dir=working_dir, train_data=train_data,
                                        val_data=val_data, test_data=test_data, decoder_net_type=checkpoint["net_type"],
                                        epochs=checkpoint["epochs"], val_epochs=checkpoint["val_epochs"],
                                        decoder_net_params=checkpoint["net_params"], use_cuda=use_cuda,
                                        projection_dataset=projection_dataset,
                                        train_parameters=checkpoint["train_parameters"],
                                        preprocess_inputs=checkpoint["preprocess_inputs"],
                                        enhance_images=checkpoint["enhance_images"])
        network_trainer.net_pretrain.load_state_dict(checkpoint["model_state_dict"])
        network_trainer.train_losses = checkpoint["train_losses"]
        network_trainer.val_losses = checkpoint["val_losses"]
        network_trainer.val_eval_epochs = checkpoint["val_eval_epochs"]
        network_trainer.optimizer_pretrain = checkpoint["optimizer_pretrain_state_dict"]

        if batch_size is not None:
            trainer1.batch_size = batch_size
        if new_epochs is not None:
            trainer1.epochs = new_epochs

        return network_trainer


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    # working_dir1 = "/media/admin/WD_Elements/Samuele_Pe/DonaldDuck_Pavia/"
    model_name1 = "vitmae_extended_dataset"
    epochs1 = 500
    trial_n1 = None
    val_epochs1 = 10
    use_cuda1 = True
    assess_calibration1 = False
    show_test1 = False
    projection_dataset1 = True
    selected_segments1 = None
    selected_projection1 = None
    preprocess_inputs1 = False
    enhance_images1 = False
    store_img1 = False

    # Load data
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_training",
                                           selected_segments=selected_segments1,
                                           selected_projection=selected_projection1)
    train_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="extended_xray_dataset_training",
                                           selected_segments=selected_segments1, selected_projection=selected_projection1,
                                           correct_mistakes=False)
    val_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_validation",
                                         selected_segments=selected_segments1, selected_projection=selected_projection1)
    test_data1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name="xray_dataset_test",
                                          selected_segments=selected_segments1,
                                          selected_projection=selected_projection1)

    # Define trainer
    train_parameters1 = {"base_lr": 1e-5, "beta1": 0.85, "beta2": 0.9, "weight_decay": 1e-4, "layer_decay": 0.75,
                         "eps": 1e-9, "scheduler": "reduce_lr_on_plateau", "min_lr": 1e-8, "batch_size": 64}
    trainer1 = ViTMAETrainer(model_name=model_name1, working_dir=working_dir1, train_data=train_data1,
                             val_data=val_data1, test_data=test_data1, decoder_net_type=None, epochs=epochs1,
                             val_epochs=val_epochs1, decoder_net_params=None, use_cuda=use_cuda1,
                             projection_dataset=projection_dataset1, preprocess_inputs=preprocess_inputs1,
                             enhance_images=enhance_images1, train_parameters=train_parameters1)

    # Apply on an example image
    url1 = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"
    image1 = Image.open(requests.get(url1, stream=True).raw)
    image1 = transforms.ToTensor()(image1)
    image_batch1 = image1.unsqueeze(0)
    # trainer1.visualise_reconstruction(image_batch1, do_normalise_input=False, title="ImageNet example", store_img=True)

    # Apply on an X-ray image
    indices1 = list(range(3))
    instance_batch1 = trainer1.custom_collate_fn([train_data1[i1] for i1 in indices1],
                                                 img_dim=(trainer1.img_preprocessor.size["height"]))
    item1, extra1 = instance_batch1
    proj_type_batch1, proj_batch1, _ = item1
    if preprocess_inputs1:
        proj_batch1 = trainer1.preprocess_fn(proj_batch1, proj_type_batch1, extra1, SetType.TRAIN)

    proj_list1 = []
    for i1 in indices1:
        proj1 = np.stack([proj_batch1[i1, 0]] * 3, axis=0)
        proj1 = torch.tensor(proj1).float()
        proj1 -= torch.min(proj1)
        proj1 /= (torch.max(proj1) - torch.min(proj1))
        proj_list1.append(proj1)
    proj_batch1 = torch.stack(proj_list1, 0)

    NetworkTrainer.set_seed(111099)
    trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)
    NetworkTrainer.set_seed(111099)
    trainer1.visualise_reconstruction(proj_batch1, do_normalise_input=False, title="Before training", store_img=store_img1)

    # Pretrain model
    new_pretraining1 = True
    if new_pretraining1:
        NetworkTrainer.set_seed(111099)
        trainer1.pretrain(show_epochs=True)

        NetworkTrainer.set_seed(111099)
        trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)
        NetworkTrainer.set_seed(111099)
        trainer1.visualise_reconstruction(proj_batch1, do_normalise_input=False, title="After training", store_img=store_img1)

    # Evaluate loaded model
    print()
    epochs2 = None
    trainer1 = ViTMAETrainer.load_model(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1,
                                        use_cuda=use_cuda1, train_data=train_data1, val_data=val_data1,
                                        test_data=test_data1, projection_dataset=projection_dataset1,
                                        new_epochs=epochs2)

    NetworkTrainer.set_seed(111099)
    trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)
    NetworkTrainer.set_seed(111099)
    trainer1.visualise_reconstruction(proj_batch1, do_normalise_input=False, title="After training", store_img=store_img1)

    # Continue pretraining
    '''if not new_pretraining1:
        trainer1.pretrain(show_epochs=True, continue_training=True)
        trainer1.summarize_performance_pretrain(show_test=show_test1, show_process=True)

        NetworkTrainer.set_seed(111099)
        trainer1.visualise_reconstruction(proj_batch1, do_normalise_input=False, title="After training")'''