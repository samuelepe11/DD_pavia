# Import packages
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry
from skimage import transform
from skimage.morphology import erosion, dilation
from scipy.ndimage import binary_fill_holes

from DataUtils.XrayDataset import XrayDataset
from Enumerators.ProjectionType import ProjectionType


# Class
class Preprocessor:
    # Define class attributes
    segmentation_fold = "segmentation_results/"

    def __init__(self, dataset):
        self.dataset = dataset
        self.segmentation_dir = dataset.results_dir + Preprocessor.segmentation_fold

        # Load MedSAM model
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        medsam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
        self.medsam = medsam.to(self.device)
        self.medsam.eval()

        self.temp_path = None

    def preprocess(self, pt_ind=None, segm_ind=None, proj_id=None, img=None, segm=None, downsampling_iterates = 3,
                   temp_path_addon="", show=True):
        if show:
            plt.figure(figsize=(10, 5))
            plt.suptitle("IMAGE PREPROCESSING")

        # Read image
        if img is None:
            img = self.dataset.patient_data[pt_ind].pt_data[segm_ind][proj_id][1]
        img = Preprocessor.normalize_img(img)
        shape = img.shape
        if show:
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Original image")

        # Adjust contrast
        contrast_factor = 2
        mean = np.mean(img)
        img = contrast_factor * (img - mean) + mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Adjust brightness
        if segm is not None and segm == "D":
            brightness_factor = 0.7
        else:
            brightness_factor = 0.3
        h, w = img.shape
        gradient = np.linspace(1, 0, h)
        brightness_func = np.tile(gradient, (w, 1)).T
        img = img.astype(np.float32) * (1 + brightness_func * brightness_factor)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if show:
            plt.subplot(1, 3, 2)
            plt.imshow(img, cmap="gray")
            plt.title("Enhanced image")

        # Gaussian pyramid downsampling
        if downsampling_iterates is not None:
            for _ in range(downsampling_iterates):
                img = cv2.pyrDown(img)
            img = transform.resize(img, shape, order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            if show:
                plt.subplot(1, 3, 3)
                plt.imshow(img, cmap="gray")
                plt.title("Downsampled image")

                self.temp_path = self.segmentation_dir + self.dataset.set_type.value
                if pt_ind is not None:
                    self.temp_path += "/pt" + str(pt_ind) + "_segm" + str(segm_ind) + "_proj" + str(proj_id) + "__"
                else:
                    self.temp_path += temp_path_addon
                plt.savefig(self.temp_path + "preprocess.png", format="png", bbox_inches="tight", pad_inches=0,
                            dpi=300)
                plt.close()
        return img

    def segment_dataset(self, overlay=False, n_instances=None):
        if overlay or n_instances is not None:
            addon = "_overlay"
        else:
            addon = ""

        if n_instances is None:
            indices = range(self.dataset.len)
            show = False
        else:
            indices = np.random.randint(0, self.dataset.len, n_instances)
            show = True

        if self.dataset.set_type.value + addon not in os.listdir(self.segmentation_dir):
            os.mkdir(self.segmentation_dir + self.dataset.set_type.value + addon)

        if n_instances is not None:
            addon += "/examples"
        for i in indices:
            # Create folder
            items, info = self.dataset.__getitem__(i)
            pt_id, segm = info
            fold_name = f"{pt_id:03}" + segm.lower()
            fold = self.segmentation_dir + self.dataset.set_type.value + addon + "/" + fold_name + "/"
            if fold_name not in os.listdir(self.segmentation_dir + self.dataset.set_type.value + addon):
                os.mkdir(fold)

            # Generate masks
            for j in range(len(items)):
                filename = "projection" + str(j)
                proj, raw_img, _ = items[j]

                if n_instances is None:
                    temp_path_addon = ""
                else:
                    temp_path_addon = addon + "/" + fold_name + "/projection" + str(j) + "_"
                img = self.preprocess(img=raw_img, show=show, temp_path_addon=temp_path_addon)

                if n_instances is None:
                    temp_path = None
                else:
                    temp_path = self.temp_path
                sam_mask = preprocessor.medsam_segmentation(img, show=show, temp_path=temp_path)
                thresh_mask = preprocessor.threshold_segmentation(img, proj=proj, segm=segm, show=show,
                                                                  temp_path=temp_path)
                hist_mask = preprocessor.hist_based_segmentation(img, proj=proj, segm=segm, show=show,
                                                                 temp_path=temp_path)

                # Merge masks
                mask = np.uint8(np.logical_or(sam_mask, thresh_mask))
                mask = np.uint8(np.logical_and(mask, hist_mask))
                unique = np.unique(mask)
                if len(unique) == 1 and unique[0] == 0:
                    mask = 255 - mask

                # Store mask
                if overlay or n_instances is not None:
                    plt.imshow(raw_img, cmap="gray")
                    alpha = 0.2
                    cmap = "jet"
                else:
                    alpha = None
                    cmap = "gray"
                plt.imshow(mask, alpha=alpha, cmap=cmap)
                plt.axis("off")
                plt.savefig(fold + filename + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close()

    def medsam_inference(self, img, box, w, h):
        # Adjust image
        prc05 = np.percentile(img, 0.5)
        prc995 = np.percentile(img, 99.5)
        img = np.clip(img, prc05, prc995)

        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        img = transform.resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Define bounding box
        box = np.array([box])
        box = box / np.array([w, h, w, h]) * 1024
        box = torch.as_tensor(box, dtype=torch.float, device=self.device)
        if len(box.shape) == 2:
            box = box[:, None, :]

        with torch.no_grad():
            img_embed = self.medsam.image_encoder(img)
            sparse_embeddings, dense_embeddings = self.medsam.prompt_encoder(points=None, boxes=box, masks=None)
            low_res_logits, _ = self.medsam.mask_decoder(image_embeddings=img_embed,
                                                         image_pe=self.medsam.prompt_encoder.get_dense_pe(),
                                                         sparse_prompt_embeddings=sparse_embeddings,
                                                         dense_prompt_embeddings=dense_embeddings,
                                                         multimask_output=False)
            low_res_pred = torch.sigmoid(low_res_logits)
            low_res_pred = F.interpolate(low_res_pred, size=(h, w), mode="bilinear", align_corners=False)
            low_res_pred = low_res_pred.squeeze().cpu().detach().numpy()
            medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def medsam_segmentation(self, img, box=None, show=True, temp_path=None):
        if show:
            plt.figure(figsize=(10, 5))
            plt.suptitle("MEDSAM SEGMENTATION")

        # Show original image
        h, w = img.shape
        if show:
            ax = plt.subplot(1, 3, 1)

            plt.imshow(img, cmap="gray")
            plt.title("Original image")

        # Show box
        if box is None:
            box = [0, 0, w - 1, h - 1]
        if show:
            Preprocessor.draw_rect(ax, box)

        # Apply MedSAM
        mask = self.medsam_inference(img, box, w, h)
        if show:
            ax = plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Raw MedSAM mask")
            Preprocessor.draw_rect(ax, box)

        # Show results
        mask = preprocessor.postprocess_mask(mask)
        if show:
            plt.subplot(1, 3, 3)
            plt.imshow(img, cmap="gray")
            plt.imshow(mask, alpha=0.2, cmap="jet")
            plt.title("Segmented image")

            if temp_path is None:
                temp_path = self.temp_path
            plt.savefig(temp_path + "medsam.png", format="png", bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()
        return mask

    @staticmethod
    def draw_rect(ax, box):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r",
                                 facecolor="none")
        ax.add_patch(rect)

    @staticmethod
    def normalize_img(img):
        img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
        img = np.uint8(img * 255)
        return img

    @staticmethod
    def hist_based_segmentation(img, proj=None, segm=None, show=True, temp_path=None):
        if segm is not None and segm == "S" and proj == ProjectionType.AP:
            beam_factor = 2.5
        else:
            beam_factor = 2.1
        # Check for supine images
        shape = img.shape
        flag = False
        if segm is not None and segm in ["C", "D"] and shape[1] > shape[0]:
            flag = True
            img = np.rot90(img, k=-1)

        if show:
            plt.figure(figsize=(10, 5))
            plt.suptitle("HISTOGRAM-BASED SEGMENTATION")

            # Show original image
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Original image")

        # Draw columns intensity histogram
        histogram = np.sum(img / 255, 0)
        if show:
            plt.subplot(1, 3, 2)
            plt.bar(range(len(histogram)), histogram)
            plt.title("Column intensity histogram")

        # Select thresholds
        middle_value = histogram[len(histogram) // 2]
        std = np.std(histogram)
        threshold1 = middle_value - std * beam_factor
        threshold2 = middle_value + std * beam_factor
        selected_cols = np.logical_and(threshold1 < histogram, threshold2 > histogram).astype(int)
        selected_cols = preprocessor.get_middle_cols(selected_cols)
        ind1 = np.argmax(selected_cols)
        ind2 = ind1 + np.argmin(selected_cols[ind1:]) - 1
        if show:
            plt.vlines(ind1, 0, np.max(histogram), linestyles="dashed", colors="r")
            plt.vlines(ind2, 0, np.max(histogram), linestyles="dashed", colors="r")

        # Select columns
        mask = np.repeat(selected_cols[np.newaxis, :], img.shape[0], 0)
        if show:
            unique = np.unique(mask)
            if len(unique) == 1 and unique[0] == 1:
                mask[0, 0] = 0
            plt.subplot(1, 3, 3)
            plt.imshow(img, cmap="gray")
            plt.imshow(mask, alpha=0.2, cmap="jet")
            plt.title("Segmented image")

            plt.savefig(temp_path + "hist_based.png", format="png", bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

        if flag:
            mask = np.rot90(img, k=1)
        return mask

    @staticmethod
    def threshold_segmentation(img, proj=None, segm=None, show=True, temp_path=None):
        if segm is not None and segm == "D" and proj == ProjectionType.AP:
            perc = 0
        else:
            perc = 10

        if show:
            plt.figure(figsize=(10, 5))
            plt.suptitle("THRESHOLD SEGMENTATION")

            # Show original image
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Original image")

        # Binarize image
        threshold = np.percentile(img, perc)
        mask = (img > threshold)
        if show:
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Raw threshold mask")

        # Show results
        mask = preprocessor.postprocess_mask(mask)
        if show:
            plt.subplot(1, 3, 3)
            plt.imshow(img, cmap="gray")
            plt.imshow(mask, alpha=0.2, cmap="jet")
            plt.title("Segmented image")

            plt.savefig(temp_path + "threshold" + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()
        return mask

    @staticmethod
    def postprocess_mask(mask, morph_iterations=30):
        # Join disconnected parts
        for _ in range(morph_iterations):
            mask = (dilation(mask)).astype(np.uint8)
        for _ in range(morph_iterations):
            mask = (erosion(mask)).astype(np.uint8)

        # Fill holes
        mask = binary_fill_holes(mask)

        # Remove small objects
        for _ in range(morph_iterations):
            mask = (erosion(mask)).astype(np.uint8)
        for _ in range(morph_iterations):
            mask = (dilation(mask)).astype(np.uint8)

        # Fill new holes
        mask = binary_fill_holes(mask)

        return mask

    @staticmethod
    def get_middle_cols(selected_cols):
        changes = np.diff(selected_cols)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if selected_cols[0] == 1:
            starts = np.insert(starts, 0, 0)
        if selected_cols[-1] == 1:
            ends = np.append(ends, len(selected_cols))

        middle_index = len(selected_cols) // 2
        out_cols = np.zeros_like(selected_cols)
        for start, end in zip(starts, ends):
            if start <= middle_index <= end:
                out_cols[start:end] = selected_cols[start:end]
                return out_cols

        return np.ones_like(selected_cols)


# Main
if __name__ == "__main__":
    # Set seed
    seed = 10
    np.random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    dataset_name1 = "xray_dataset_training"

    # Define data
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Preprocess image
    preprocessor = Preprocessor(dataset1)
    # img1 = preprocessor.preprocess(pt_ind=0, segm_ind=0, proj_id=0, show=True)

    # MedSAM preprocessing
    box1 = None
    # preprocessor.medsam_segmentation(img1, show=True, None)

    # Manual preprocessing
    # Preprocessor.hist_based_segmentation(img1, show=True, temp_path=preprocessor.temp_path)

    # Threshold preprocessing
    # Preprocessor.threshold_segmentation(img1, show=True, temp_path=preprocessor.temp_path)

    # Segment dataset
    overlay1 = False
    n_instances1 = None
    preprocessor.segment_dataset(overlay=overlay1, n_instances=n_instances1)
