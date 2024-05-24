import csv
import glob
import logging as log
import os
from torch.utils.data import Dataset
from .dataset_type import DatasetType
from .xray_helper import XRayNormalizationType, normalize_xray
from xray import XRay


def get_xray(data_path: str, patient: str):
    xray = XRay(os.path.join(data_path, patient), load_metadata=False)
    return xray.img


def get_normalized_xray(data_path: str, patient: str, xray_normalization: XRayNormalizationType):
    xray = get_xray(data_path, patient)
    return normalize_xray(xray, xray_normalization)

        
class ChexpertSmall(Dataset):
    def __init__(self,
             data_path: str='/home/dileo/thesis/data/CheXpert-v1.0-small',
             dataset_type: DatasetType=DatasetType.TRAIN,
             device: str='cuda:1',
             xray_normalization: XRayNormalizationType=XRayNormalizationType.IODICE
            ):
        super().__init__()
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.device = device
        self.xray_normalization = xray_normalization
        
        csv_filename = 'train.csv' if dataset_type == DatasetType.TRAIN else 'valid.csv'

        self.patients = []

        with open(os.path.join(data_path, csv_filename)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Frontal/Lateral'] == 'Frontal':
                    self.patients.append(f'../{row["Path"]}')
                    
    def __getitem__(self, i):
        patient = self.patients[i]
        
        log.debug(f'Patient {patient}')
        
        if self.xray_normalization is not None:
            xray = get_normalized_xray(self.data_path, patient, self.xray_normalization)
        else:
            xray = get_xray(self.data_path, patient)
        
        return xray, xray
    
    def __len__(self):
        return len(self.patients)