# Import packages
from DataUtils.XrayDataset import XrayDataset


# Class
class XrayProjectionDataset(XrayDataset):

    def __init__(self, working_dir, dataset):
        super().__init__(working_dir)
        self.__dict__.update(dataset.__dict__)

        # Define projection instance names
        self.dicom_projection_instances = []
        for i in range(len(self.dicom_instances)):
            segm, _ = super().__getitem__(i)
            self.dicom_projection_instances += [f"{self.dicom_instances[i]}_{j}" for j in range(len(segm))]
        self.len = len(self.dicom_projection_instances)

    def __getitem__(self, ind):
        projection_name = self.dicom_projection_instances[ind]
        segment_name, proj_ind = projection_name.split("_")
        instance_ind = self.dicom_instances.index(segment_name)
        segment_data, extra = super().__getitem__(instance_ind)
        pt_id, segment_id = extra

        try:
            return [segment_data[int(proj_ind)]], (pt_id, segment_id)
        except IndexError:
            print(proj_ind)
            print(len(segment_data))
            print(projection_name)

    def __len__(self):
        return self.len


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"

    # Load an already split datasets
    dataset_name1 = "xray_dataset_training"
    dataset1 = XrayDataset.load_dataset(working_dir=working_dir1, dataset_name=dataset_name1)

    # Instantiate Projection Dataset
    dataset1 = XrayProjectionDataset(working_dir=working_dir1, dataset=dataset1)
