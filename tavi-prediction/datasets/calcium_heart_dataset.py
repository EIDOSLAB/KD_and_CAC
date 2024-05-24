# CT scans from Calcium Score dataset cropped around heart

import glob
import os
import pickle
import sqlite3
import torch
import torchvision
from ct import Ct
from .disk_cache import get_cache
from torch.utils.data import Dataset
from utils import natural_key
from .dataset_type import DatasetType


patients_to_discard = ['CAC_074', 'CAC_197', 'CAC_229', 'CAC_298', 'CAC_552', 'CS_091']
test_set = ['CS_005', 'CAC_521', 'CAC_487', 'CAC_381', 'CAC_162', 'CAC_143', 'CAC_295', 'CAC_177', 'CAC_172', 'CAC_494', 'CAC_280', 'CS_081', 'CS_038', 'CAC_026', 'CAC_454', 'CAC_312', 'CAC_316', 'CS_092', 'CS_083', 'CAC_334', 'CAC_368', 'CAC_401', 'CAC_140', 'CAC_377', 'CAC_061', 'CAC_107', 'CAC_483', 'CAC_128', 'CAC_036', 'CS_037', 'CAC_137', 'CAC_338', 'CAC_022', 'CAC_223', 'CAC_129', 'CAC_301', 'CS_058', 'CAC_481', 'CAC_248', 'CAC_275', 'CAC_070', 'CAC_196', 'CAC_313', 'CAC_359', 'CAC_263', 'CAC_175', 'CAC_123', 'CAC_354', 'CAC_094', 'CS_030', 'CAC_335', 'CAC_124', 'CS_017', 'CAC_048', 'CAC_150', 'CAC_225', 'CAC_426', 'CS_047', 'CAC_180', 'CAC_077', 'CAC_209', 'CS_049', 'CAC_216', 'CAC_059', 'CAC_433', 'CS_087', 'CAC_460', 'CAC_224', 'CAC_016', 'CAC_399', 'CAC_195', 'CAC_466', 'CAC_547', 'CAC_080', 'CAC_194', 'CAC_231', 'CS_020', 'CAC_114', 'CAC_254', 'CAC_355', 'CAC_226', 'CAC_034', 'CS_019', 'CAC_176']

ct_cache = get_cache('calcium_ct')
@ct_cache.memoize(typed=True, expire=None, tag='ct')
def get_ct(data_path: str, patient: str):
    ct = Ct(f'{data_path}/{patient["id"]}/ct/', file_pattern=f'{patient["files_pattern"]}*.dcm')
    ct.crop(z=(patient['crop_box']['min_z'], patient['crop_box']['max_z']),
            y=(patient['crop_box']['min_y'], patient['crop_box']['max_y']),
            x=(patient['crop_box']['min_x'], patient['crop_box']['max_x']))
    ct.scale(size=(100,256,256))

    img = ct.img / 1000
    return img.detach().clone()


class CalciumHeartDataset(Dataset):
    PATIENT_TABLE = 'patient'
    
    def __init__(self,
                 data_path: str='/data/calcium_processed',
                 crop_path: str='./results/heart_crop_2',
                 db_path: str='/data/calcium_processed/site.db',
                 dataset_type: DatasetType=DatasetType.TRAIN
                ):
        super().__init__()
        self.data_path = data_path
        self.dataset_type = dataset_type
        
        crop_files = glob.glob(f'{crop_path}/*.pkl')
        crop_files.sort(key=natural_key)
        
        patients_ids = [os.path.basename(file)[:-4] for file in crop_files
                        if os.path.basename(file)[:-4] not in patients_to_discard]
        
        db_connection = sqlite3.connect(db_path)
        self.db = db_connection.cursor()
        
        if dataset_type == DatasetType.TEST:
            patients_ids = [patient_id for patient_id in patients_ids if patient_id in test_set]
        else:
            patients_ids = [patient_id for patient_id in patients_ids if patient_id not in test_set]
        
        self.patients = []
        for patient in patients_ids:
            try:
                cac_score = self.get_cac_score(patient)
            except Exception as exc:
                print(f'Patient {patient} CAC score not found on db: {exc}')
                continue
            
            try:
                crop_box = self.get_crop_box(f'{crop_path}/{patient}.pkl')
            except Exception as exc:
                print(f'Patient {patient} crop box not found at path: {crop_path}/{patient}.pkl: {exc}')
                continue
                
            try:
                files_pattern = self.get_files_pattern(f'{data_path}/{patient}/ct/')
            except Exception as exc:
                print(f'Patient {patient} files pattern not found at path: {data_path}/{patient}/ct/: {exc}')
                continue
            
            self.patients.append({'id': patient,
                                  'cac_score': cac_score,
                                  'crop_box': crop_box,
                                  'files_pattern': files_pattern
                                 })

        print(f'loaded dataset with {len(self.patients)} patients')
        
        scores = [patient['cac_score'] for patient in self.patients]
        print(sum(i>0 for i in scores), sum(i>10 for i in scores), sum(i>70 for i in scores), sum(i>100 for i in scores), len(scores))
        
    def get_cac_score(self, patient: str):
        self.db.execute(f'SELECT cac_score FROM {CalciumHeartDataset.PATIENT_TABLE} WHERE id = \'{patient}\';')
        rows = self.db.fetchall()
        if len(rows) != 1:
            raise Exception(f'Found {len(rows)} rows patient {patient}')
        return rows[0][0]
    
    def get_crop_box(self, crop_file: str):
        with open(crop_file, 'rb') as file:
            return pickle.load(file)
        
    def get_files_pattern(self, folder: str):
        files = os.listdir(folder)
        files.sort(key=natural_key)
        if len(files) == 0:
            raise Exception(f'Folder {folder} is empty')
        elif len(files) < 10:
            raise Exception(f'Folder {folder} contains {len(files)} files only')

        # found the pattern with highest number of dcm
        files_patterns = list(set([file[:8] for file in files]))
        files_patterns_count = dict()
        for files_pattern in files_patterns:
            files_patterns_count[files_pattern] = 0;
        for file in files:
            files_patterns_count[file[:8]] += 1;
        files_pattern = max(files_patterns_count, key=files_patterns_count.get)
        return files_pattern
    
    def __getitem__(self, i):
        patient = self.patients[i]

        img = get_ct(self.data_path, patient)
        label = [0.0,1.0] if patient["cac_score"] > 0 else [1.0,0.0]
        
        return img.unsqueeze(0), torch.tensor(label)
    
    def __len__(self):
        return len(self.patients)
    
        
if __name__ == '__main__':
    train_dataset = CalciumHeartDataset(dataset_type=DatasetType.TRAIN)
    valid_dataset = CalciumHeartDataset(dataset_type=DatasetType.VALIDATION)
    test_dataset = CalciumHeartDataset(dataset_type=DatasetType.TEST)

    #import pprint
    #pp = pprint.PrettyPrinter(indent=2)
    #pp.pprint(dataset.patients)
    
    image, cac = train_dataset[3]
    print(image.shape, cac)
    image, cac = train_dataset[3]
    print(image.shape, cac)
    
    image, cac = train_dataset[0]
    print(image.shape, cac)
