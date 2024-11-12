# Import packages
import os
import matplotlib.pyplot as plt
import scipy
import random
import numpy as np
import cv2
import pandas as pd
import SimpleITK as sitk
import seaborn as sns
from statsmodels.stats import inter_rater as ir

from DataUtils.XrayDataset import XrayDataset


# Class
class AnnotateMasks:
    result_folder = "gt_masks"
    annotator_labels = ["s1", "j1", "s2", "j2"]

    def __init__(self, dataset, desired_instances, mask_folder):
        self.dataset = dataset
        self.desired_instances = desired_instances
        self.mask_dir = dataset.data_dir + mask_folder + "/"
        self.result_dir = self.mask_dir + self.result_folder

        self.annotators = [x for x in os.listdir(self.mask_dir) if "." not in x and x not in self.annotator_labels and
                           self.result_folder not in x]
        self.annotators_dict = dict(zip(self.annotators, self.annotator_labels))

        self.sens = []
        self.spec = []
        self.fleiss_k = None
        self.cohen_k = None

    def preprocess_masks(self, instance_name=None):
        if instance_name is not None:
            instance_list = [instance_name]
        else:
            instance_list = self.desired_instances

        for instance in instance_list:
            for user in self.annotators:
                # Create folder for preprocessed masks
                new_folder = self.annotators_dict[user]
                if new_folder not in os.listdir(self.mask_dir):
                    os.mkdir(self.mask_dir + "/" + new_folder)
                if instance not in os.listdir(self.mask_dir + new_folder):
                    os.mkdir(self.mask_dir + new_folder + "/" + instance)

                # Apply fill holes to every mask
                instance_dir = self.mask_dir + "/" + user + "/" + instance + "/"
                files = os.listdir(instance_dir)
                if instance_name is not None:
                    x, _ = self.dataset.get_data_from_name(instance)
                    temp_files = []
                    for j, projection in enumerate(x):
                        proj_files = [f for f in files if f.startswith("projection" + str(j))]
                        image_dim = x[j][1].shape
                        if len(proj_files) > 0:
                            temp_masks = []
                            for proj_file in proj_files:
                                temp_masks.append(self.read_mask(instance_dir + proj_file, image_dim))
                            temp_files.append(np.round(np.mean(temp_masks, 0)).astype(int))
                        else:
                            temp_files.append(np.zeros(image_dim))
                    files = temp_files
                    plt.figure(figsize=(15, 15))

                for i, file in enumerate(files):
                    if isinstance(file, str):
                        mask = self.read_mask(instance_dir + file)
                    else:
                        mask = file

                    if instance_name is not None:
                        plt.subplot(2, len(x), i + 1)
                        plt.imshow(mask, "gray")
                        plt.title("PROJECTION " + str(i))

                    # Fill holes
                    mask = scipy.ndimage.binary_fill_holes(mask).astype(int)
                    if instance_name is not None:
                        plt.subplot(2, len(x), i + 1 + len(x))
                        plt.imshow(mask, "gray")
                    else:
                        # Store filled mask
                        fig = plt.figure()
                        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
                        ax.axis("off")
                        plt.imshow(mask, "gray")
                        plt.savefig(self.mask_dir + new_folder + "/" + instance + "/" + file, format="jpg",
                                    bbox_inches="tight", pad_inches=0, dpi=300)
                        plt.close()

                if instance_name is not None:
                    plt.savefig(self.mask_dir + "fill_holes_example_" + instance_name + "_" +
                                self.annotators_dict[user] + ".jpg", format="jpg", dpi=500)
                    plt.close()

    def count_masks(self):
        col_names = ["Item name"] + self.annotators
        cols = []
        for instance in self.desired_instances:
            row = [instance]
            for annotator in self.annotators:
                files = os.listdir(self.mask_dir + self.annotators_dict[annotator] + "/" + instance + "/")
                projections = np.unique([file.split("_")[0] for file in files])
                row.append(len(projections))
            cols.append(row)

        df = pd.DataFrame(cols, columns=col_names)
        df.to_csv(self.mask_dir + "mask_count.csv", index=False)

        # Save result as a confusion matrix
        df.set_index("Item name", inplace=True)
        plt.figure(figsize=(5, 20))
        sns.heatmap(df, annot=True, cmap="Reds", cbar=False)
        plt.xlabel("ANNOTATOR")
        plt.ylabel("ITEM")
        plt.savefig(self.mask_dir + "mask_count.png", format="png", dpi=500)

    def build_gt_mask(self, instance_name=None, annotator_type=None):
        self.sens, self.spec = [], []
        if annotator_type is None:
            annotators = self.annotators
            num_annotators = len(annotators)
            addon = ""
        else:
            annotators = [x for x in self.annotators if annotator_type in self.annotators_dict[x]]
            num_annotators = len(annotators)
            addon = "_" + annotator_type

        if self.result_folder + addon not in os.listdir(self.mask_dir):
            os.mkdir(self.mask_dir + self.result_folder + addon)

        for instance in self.desired_instances:
            if instance not in os.listdir(self.mask_dir + self.result_folder + addon):
                os.mkdir(self.mask_dir + self.result_folder + addon + "/" + instance)

            projections, _ = self.dataset.get_data_from_name(instance)
            for i, projection in enumerate(projections):
                if instance_name is not None:
                    if instance_name == instance:
                        fig = plt.figure(figsize=(15, 15))
                    else:
                        continue

                staple_filter = sitk.STAPLEImageFilter()
                annotators_masks = []
                annotators_masks_raw = []
                for j, annotator in enumerate(annotators):
                    folder = self.mask_dir + self.annotators_dict[annotator] + "/" + instance
                    files = os.listdir(folder)
                    required_files = [file for file in files if file.split("_")[0].endswith(str(i))]
                    if len(required_files) == 1:
                        mask = self.read_mask(folder + "/" + required_files[0])
                    elif len(required_files) > 1:
                        projection_masks = []
                        for required_file in required_files:
                            mask = self.read_mask(folder + "/" + required_file)
                            projection_masks.append(mask)
                        mask = np.round(np.mean(projection_masks, 0)).astype(int)
                    else:
                        print("Annotator " + annotator + " did not provide projection " + str(i) + " for instance "
                              + instance + " ...")
                        mask = np.zeros(projection[1].shape).astype(int)
                    annotators_masks_raw.append(mask)
                    annotators_masks.append(sitk.GetImageFromArray(mask))
                    if instance_name is not None and instance_name == instance:
                        plt.subplot(2, num_annotators, j + 1)
                        plt.imshow(mask, "gray")
                        plt.title(annotator.upper())

                # Process and store mask
                if np.all([self.is_empty_mask(x) for x in annotators_masks_raw]):
                    dim_mask = annotators_masks_raw[0].shape
                    gt_mask = np.zeros(dim_mask).astype(int)
                    sens, spec = [], []
                    for a in range(num_annotators):
                        sens.append(1)
                        spec.append(1)
                    sens = tuple(sens)
                    spec = tuple(spec)
                else:
                    try:
                        staple_result = staple_filter.Execute(annotators_masks)
                        staple_result_array = sitk.GetArrayFromImage(staple_result)
                        gt_mask = np.round(staple_result_array).astype(int)

                        sens = staple_filter.GetSensitivity()
                        spec = staple_filter.GetSpecificity()
                    except:
                        print("STAPLE was NOT able to process projection " + str(i) + " for instance " + instance + " ...")
                        dim_mask = annotators_masks_raw[0].shape
                        gt_mask = np.zeros(dim_mask).astype(int)
                        sens, spec = [], []
                        for a in range(num_annotators):
                            sens.append(0)
                            num_px = dim_mask[0] * dim_mask[1]
                            spec.append(1 - np.count_nonzero(annotators_masks_raw[a]) / num_px)
                        sens = tuple(sens)
                        spec = tuple(spec)

                # Check for NaNs in sensitivity and specificity
                if np.any(np.isnan(sens)) or np.any(np.isnan(spec)):
                    print("WATCH OUT! NaN values for instance " + instance + " projection " + str(i) +
                          " were detected.")

                if instance_name is not None and instance_name == instance:
                    print("Results for " + instance + " projection " + str(i))
                    for a in range(num_annotators):
                        print(" - Annotator " + annotators[a] + ":")
                        print("    > Sensitivity = " + self.show_percentage(sens[a]))
                        print("    > Specificity = " + self.show_percentage(spec[a]))
                    print()
                    plt.subplot(2, 1, 2)
                else:
                    self.sens.append(sens)
                    self.spec.append(spec)
                    fig = plt.figure()
                    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
                    ax.axis("off")
                plt.imshow(gt_mask, "gray")

                if instance_name is not None and instance_name == instance:
                    plt.title("STAPLE results")
                    plt.savefig(self.mask_dir + "staple_example_" + instance + "_" + str(i) + addon + ".jpg",
                                format="jpg")
                else:
                    plt.savefig(self.mask_dir + self.result_folder + addon + "/" + instance + "/projection" +
                                str(i) + ".jpg", format="jpg", bbox_inches="tight", pad_inches=0)
                plt.close(fig)

            if instance_name is None:
                print("===================================================================================================")

        # Return annotators' performances
        if instance_name is None:
            self.sens = np.array(self.sens)
            self.spec = np.array(self.spec)
            print()
            for i in range(len(annotators)):
                print("Annotator " + annotators[i] + ":")
                print("  > Sensitivity = " + self.show_percentage(np.mean(self.sens[:, i])))
                print("  > Specificity = " + self.show_percentage(np.mean(self.spec[:, i])))

    def compare_all_annotators(self, instance_names=None):
        self.fleiss_k = []
        for instance in self.desired_instances:
            x, _ = self.dataset.get_data_from_name(instance)
            for i in range(len(x)):
                img_dim = x[i][1].shape
                vec_dim = img_dim[0] * img_dim[1]
                annotators_masks = []
                for annotator in self.annotators:
                    try:
                        desired_folder = self.mask_dir + self.annotators_dict[annotator] + "/" + instance
                        desired_files = [f for f in os.listdir(desired_folder) if f.startswith("projection" + str(i))]
                        desired_masks = [self.read_mask(desired_folder + "/" + f, img_dim) for f in desired_files]
                        mask = np.round(np.mean(desired_masks, 0)).astype(int)

                        mask = mask.reshape(vec_dim)
                        annotators_masks.append(mask)
                    except:
                        print("Annotator " + annotator + " did not provide projection " + str(i) + " for instance "
                              + instance + " ...")
                        annotators_masks.append(np.zeros(vec_dim))

                # Compute Fleiss' kappa
                try:
                    masks = np.array(annotators_masks)
                    if self.is_empty_mask(masks):
                        # If every mask is empty there is perfect agreement
                        fleiss_k = 1.0
                    else:
                        dats, _ = ir.aggregate_raters(masks)
                        fleiss_k = ir.fleiss_kappa(dats, method="fleiss")
                    self.fleiss_k.append(fleiss_k)

                    if instance_names is not None and instance in instance_names:
                        kappa = str(np.round(fleiss_k, 3))
                        print(" - Fleiss' kappa for instance " + instance + " projection " + str(i) + " is " +
                              str(kappa))

                except:
                    print("Unable to compute Fleiss' kappa for instance " + instance + " projection " + str(i) + "!")
            print()

        mean_k = np.mean(np.array(self.fleiss_k))
        print("Mean Fleiss' kappa = " + str(np.round(mean_k, 3)))

    def compare_binary_subgroups(self, instance_names=None, binary_subgroup=None):
        if binary_subgroup is None:
            print("Comparing senior vs. junior annotators...")
            annotators = ["j", "s"]
        else:
            print("Comparing " + binary_subgroup[0] + " vs. " + binary_subgroup[1] + "...")
            annotators = binary_subgroup
        self.cohen_k = []
        for instance in self.desired_instances:
            x, _ = self.dataset.get_data_from_name(instance)
            for i in range(len(x)):
                img_dim = x[i][1].shape
                vec_dim = img_dim[0] * img_dim[1]
                annotators_masks = []
                for annotator in annotators:
                    if binary_subgroup is None:
                        desired_folder = self.result_dir + "_" + annotator
                    else:
                        desired_folder = self.mask_dir + annotator
                    desired_folder += "/" + instance

                    if binary_subgroup is None:
                        filename = "projection" + str(i) + ".jpg"
                        mask = self.read_mask(desired_folder + "/" + filename, img_dim)
                    else:
                        desired_files = [f for f in os.listdir(desired_folder) if f.startswith("projection" + str(i))]
                        desired_masks = [self.read_mask(desired_folder + "/" + f, img_dim) for f in desired_files]
                        if len(desired_masks) > 0:
                             mask = np.round(np.mean(desired_masks, 0)).astype(int)
                        else:
                            mask = np.zeros(img_dim)

                    mask = mask.reshape(vec_dim)
                    annotators_masks.append(mask)

                # Compute Cohen's kappa
                try:
                    masks = np.array(annotators_masks)
                    if self.is_empty_mask(masks):
                        # If every mask is empty there is perfect agreement
                        cohen_k = 1.0
                    else:
                        dats, _ = ir.aggregate_raters(masks)
                        cohen_k = ir.cohens_kappa(dats, return_results=False)
                    self.cohen_k.append(cohen_k)

                    if instance_names is not None and instance in instance_names:
                        kappa = str(np.round(cohen_k, 3))
                        print(" - Cohen's kappa for instance " + instance + " projection " + str(i) + " is " +
                              str(kappa))
                except:
                    print("Unable to compute Cohen's kappa for instance " + instance + " projection " + str(i) + "!")

        print()
        mean_k = np.mean(np.array(self.cohen_k))
        print("Mean Cohen's kappa = " + str(np.round(mean_k, 3)))

    @staticmethod
    def read_mask(filepath, image_dim=None):
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image_dim is not None:
            mask = cv2.resize(mask, (image_dim[1], image_dim[0]))
        return (np.round(mask / 255)).astype(int)

    @staticmethod
    def show_percentage(value):
        return str(np.round(value * 100, 2)) + "%"

    @staticmethod
    def is_empty_mask(mask):
        return len(np.unique(mask)) == 1 and mask[0, 0] == 0


# Main
if __name__ == "__main__":
    # Set seed
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    working_dir1 = "./../../"
    dataset_name1 = "xray_dataset_validation"
    mask_folder1 = dataset_name1 + "_masks"

    # Define data
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Define annotator
    annotator1 = AnnotateMasks(dataset=dataset1, desired_instances=dataset1.dicom_instances, mask_folder=mask_folder1)

    # Preprocess masks
    instance_names1 = ["320l", "356s", "418l"]
    # instance_names1 = [None]
    # for instance_name1 in instance_names1:
    #     annotator1.preprocess_masks(instance_name=instance_name1)

    # Count masks
    annotator1.count_masks()

    # Compare all annotators' performances
    # annotator1.compare_all_annotators(instance_names=instance_names1)

    # Generate STAPLE masks
    annotator_type1 = "s"
    # for instance_name1 in instance_names1:
    #    annotator1.build_gt_mask(instance_name=instance_name1, annotator_type=annotator_type1)

    # Compare senior and junior annotators' performances
    # annotator1.compare_binary_subgroups(instance_names=instance_names1, binary_subgroup=["s1", "s2"])
    print("=================================================================")
    # annotator1.compare_binary_subgroups(instance_names=instance_names1, binary_subgroup=["j1", "j2"])
    print("=================================================================")
    # annotator1.compare_binary_subgroups(instance_names=instance_names1)

