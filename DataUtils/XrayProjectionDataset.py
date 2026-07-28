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
        self.dicom_projection_instances = list(dict.fromkeys(self.dicom_projection_instances))
        if hasattr(self, "wrong_label_instances") and self.wrong_label_instances is not None:
            for patient in self.patient_data:
                for s, segment in enumerate(patient.pt_data):
                    for p, projection in enumerate(segment):
                        if f"{patient.id:03}" + patient.segments[s].lower() + "_" + str(p) in self.wrong_label_instances:
                            if projection[2] != "":
                                tmp = ""
                            else:
                                tmp = projection[3].split("vertebra_name='")[-1].split("'")[0]
                            segment[p] = (projection[0], projection[1], tmp, projection[3])
        if hasattr(self, "removable_instances") and self.removable_instances is not None:
            self.dicom_projection_instances = [instance for instance in self.dicom_projection_instances if instance not
                                               in self.removable_instances]
        self.len = len(self.dicom_projection_instances)

    def __getitem__(self, ind):
        projection_name = self.dicom_projection_instances[ind]
        try:
            segment_name, proj_ind = projection_name.split("_")
        except ValueError:
            print(projection_name)
        instance_ind = self.dicom_instances.index(segment_name)
        segment_data, extra = super().__getitem__(instance_ind)
        pt_id, segment_id = extra

        return [segment_data[int(proj_ind)]], (pt_id, segment_id, int(proj_ind))

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
