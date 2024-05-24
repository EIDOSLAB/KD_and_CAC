import glob
import logging as log
import numpy as np
import os
import sqlite3
import time
import torch
import torch.nn.functional as F
from .modality import Ct, XRay
from .utils import get_cache, DatasetType
from .xray_helper import XRayNormalizationType, normalize_xray
from monai.transforms import RandAffine
from scipy.spatial import ConvexHull, Delaunay
from skimage.measure import label, regionprops
from skimage.morphology import dilation, erosion, cube
from torch.utils.data import Dataset
from utils import natural_key


# CT scans from Calcium Score dataset cropped around heart using FPN trained with run_heart_segmentation.py
# The heart is cropped around heart and background is removed
# All scans are resized to 150*220*280 (D*H*W) ->
#    smaller images are filled with -1000 (air density),
#    larger are down-scaled proportionally to fit larger dimension and filled with -1000 if needed to fit exactly the desired size


#patients_to_discard = ['CAC_074', 'CAC_197', 'CAC_229', 'CAC_298', 'CAC_552', 'CS_091']
patients_to_discard = ['CAC_477'] # cannot be opened
test_set = ['CS_005', 'CAC_521', 'CAC_487', 'CAC_381', 'CAC_162', 'CAC_143', 'CAC_295', 'CAC_177', 'CAC_172', 'CAC_494', 'CAC_280', 'CS_081', 'CS_038', 'CAC_026', 'CAC_454', 'CAC_312', 'CAC_316', 'CS_092', 'CS_083', 'CAC_334', 'CAC_368', 'CAC_401', 'CAC_140', 'CAC_377', 'CAC_061', 'CAC_107', 'CAC_483', 'CAC_128', 'CAC_036', 'CS_037', 'CAC_137', 'CAC_338', 'CAC_022', 'CAC_223', 'CAC_129', 'CAC_301', 'CS_058', 'CAC_481', 'CAC_248', 'CAC_275', 'CAC_070', 'CAC_196', 'CAC_313', 'CAC_359', 'CAC_263', 'CAC_175', 'CAC_123', 'CAC_354', 'CAC_094', 'CS_030', 'CAC_335', 'CAC_124', 'CS_017', 'CAC_048', 'CAC_150', 'CAC_225', 'CAC_426', 'CS_047', 'CAC_180', 'CAC_077', 'CAC_209', 'CS_049', 'CAC_216', 'CAC_059', 'CAC_433', 'CS_087', 'CAC_460', 'CAC_224', 'CAC_016', 'CAC_399', 'CAC_195', 'CAC_466', 'CAC_547', 'CAC_080', 'CAC_194', 'CAC_231', 'CS_020', 'CAC_114', 'CAC_254', 'CAC_355', 'CAC_226', 'CAC_034', 'CS_019', 'CAC_176']


def flood_fill_hull(image):    
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


def get_largest_connected_component(image):
    labels = label(image.numpy())
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = erosion(largestCC, cube(10))
    largestCC = dilation(largestCC, cube(20))
    largestCC = erosion(largestCC, cube(10))
    return largestCC


def copy_image_in_the_middle(src, dst, patient):
    if (src.shape[0] > dst.shape[0]
        or src.shape[1] > dst.shape[1]
        or src.shape[2] > dst.shape[2]
    ):
        ratio = max([src.shape[0]/dst.shape[0],src.shape[1]/dst.shape[1],src.shape[2]/dst.shape[2]])
        new_shape = (
            round(src.shape[0]/ratio),
            round(src.shape[1]/ratio),
            round(src.shape[2]/ratio)
        )
        log.info(f'{patient["id"]} ct scan downscaled by ratio: {ratio}, initial size: {src.shape}')
        src = F.interpolate(src.unsqueeze(0).unsqueeze(0), size=new_shape, mode='nearest').squeeze(0).squeeze(0)
                     
    start_depth = (dst.shape[0]-src.shape[0])//2
    end_depth = start_depth + src.shape[0]
    start_height = (dst.shape[1]-src.shape[1])//2
    end_height = start_height + src.shape[1]
    start_width = (dst.shape[2]-src.shape[2])//2
    end_width = start_width + src.shape[2]
    dst[start_depth:end_depth,start_height:end_height,start_width:end_width] = src


ct_cache = get_cache('calcium_no_bg_hull_ct_dist')
@ct_cache.memoize(typed=True, expire=None, tag='ct')
def get_ct(data_path: str, segmentation_model_path: str, device: str, patient: str):
    """
    Download the ct scan, use the segmentation network to find the heart, find the largest connected
    component (heart) and discard the rest
    """
    segmentation_network = torch.load(segmentation_model_path, map_location=device)
    ct = Ct(f'{data_path}/{patient["id"]}/ct/', file_pattern=f'{patient["files_pattern"]}*.dcm')
    
    mask = torch.empty(ct.img.shape)
    for i in range(0, ct.img.shape[0]):
        image = ct.img[i] / 1000
        mask[i] = segmentation_network.predict(image.to(device).unsqueeze(0).unsqueeze(0)) > 0.5
    
    largestCC = get_largest_connected_component(mask.to('cpu'))
    props = regionprops(largestCC.astype(np.uint8))

    # regionprops implementation of convex hull hidden in this lazy property takes really long to compute
    # convex_hull = props[0]['image_convex']
    
    convex_hull, _ = flood_fill_hull(largestCC)
    ct.img[convex_hull==0] = -1000
    
    # alternative not using convex hull is to remove everything that is not the mask
    #ct.img[largestCC==0] = -1000
    
    ret_img = torch.full((150,220,280), -1000)
    
    ct.crop(z=(props[0]['bbox'][0], props[0]['bbox'][3]),
            y=(props[0]['bbox'][1], props[0]['bbox'][4]),
            x=(props[0]['bbox'][2], props[0]['bbox'][5]))

    copy_image_in_the_middle(ct.img, ret_img, patient)
    return ret_img / 1000


xray_cache = get_cache('calcium_xray')
@xray_cache.memoize(typed=True, expire=None, tag='xray')
def get_xray(data_path: str, patient_id: str):
    xray = XRay(f'{data_path}/{patient_id}/rx/')
    return xray.img


xray_norm_cache = get_cache('normalized_xray')
@xray_norm_cache.memoize(typed=True, expire=None, tag='xray_norm')
def get_normalized_xray(data_path: str, patient_id: str, xray_normalization: XRayNormalizationType):
    xray = XRay(f'{data_path}/{patient_id}/rx/')
    return normalize_xray(xray.img, xray_normalization)


class CalciumDataset(Dataset):
    PATIENT_TABLE = 'patient'
    
    def __init__(self,
                 data_path: str='/data/calcium_processed',
                 segmentation_model_path: str='./results/heart_segmentation_2/best_model_fpn.pth',
                 db_path: str='/data/calcium_processed/site.db',
                 dataset_type: DatasetType=DatasetType.TRAIN,
                 device: str='cuda:1',
                 use_augmentation: bool=False,
                 xray_only: bool=True,
                 xray_normalization: XRayNormalizationType=None
                ):
        super().__init__()
        self.data_path = data_path
        self.segmentation_model_path = segmentation_model_path
        self.dataset_type = dataset_type
        self.device = device
        self.use_augmentation = use_augmentation
        self.xray_only = xray_only
        self.xray_normalization = xray_normalization
        
        folders = glob.glob(f'{data_path}/*/ct/')
        folders.sort(key=natural_key)
        
        ct_patients_ids = [os.path.basename(os.path.normpath(folder[:-4])) for folder in folders
                           if os.path.basename(os.path.normpath(folder[:-4])) not in patients_to_discard]
        
        db_connection = sqlite3.connect(db_path)
        db = db_connection.cursor()
        
        self.patients = []
        for patient in ct_patients_ids:
            try:
                cac_score = self.get_cac_score(db, patient)
            except Exception as exc:
                log.debug(f'Patient {patient} CAC score not found on db: {exc}')
                continue
            
            try:
                files_pattern = self.get_files_pattern(f'{data_path}/{patient}/ct/')
            except Exception as exc:
                log.debug(f'Patient {patient} files pattern not found at path: {data_path}/{patient}/ct/: {exc}')
                continue
            
            self.patients.append({'id': patient,
                                  'cac_score': cac_score,
                                  'files_pattern': files_pattern,
                                 })
        
        if dataset_type == DatasetType.TEST:
            # look for patients that have xray but not ct scan
            xray_folders = glob.glob(f'{data_path}/*/rx/')
            xray_folders.sort(key=natural_key)
            xray_patients_ids = [os.path.basename(os.path.normpath(folder[:-4])) for folder in xray_folders
                                 if os.path.basename(os.path.normpath(folder[:-4])) not in patients_to_discard]

            xray_only_patients_ids = [patient_id for patient_id in xray_patients_ids
                                      if patient_id not in map(lambda x: x['id'], self.patients)]
            
            # keep patients that have a ct scan and are in test_set
            self.patients = [patient for patient in self.patients if patient['id'] in test_set]
            
            # add patients that do not have a ct scan
            for patient in xray_only_patients_ids:
                try:
                    cac_score = self.get_cac_score(db, patient)
                except Exception as exc:
                    log.debug(f'Patient {patient} CAC score not found on db: {exc}')
                    continue

                self.patients.append({'id': patient,
                                      'cac_score': cac_score,
                                      'files_pattern': None,
                                     })
        else:
            self.patients = [patient for patient in self.patients if patient['id'] not in test_set]
            
        log.info(f'loaded dataset with {len(self.patients)} patients')
        
    def get_cac_score(self, db, patient: str):
        db.execute(f'SELECT cac_score FROM {CalciumDataset.PATIENT_TABLE} WHERE id = \'{patient}\';')
        rows = db.fetchall()
        if len(rows) != 1:
            raise Exception(f'Found {len(rows)} rows patient {patient}')
        return rows[0][0]
        
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
        
        log.debug(f'Patient {patient["id"]}')
        
        label = [0.0,1.0] if patient["cac_score"] > 0 else [1.0,0.0]
        if self.xray_normalization is not None:
            xray = get_normalized_xray(self.data_path, patient["id"], self.xray_normalization)
        else:
            xray = get_xray(self.data_path, patient["id"])
        
        time.sleep(0.15)
        
        if self.xray_only or self.dataset_type == DatasetType.TEST:
            return xray, torch.tensor(label)
        else:
            img = get_ct(self.data_path, self.segmentation_model_path, self.device, patient).unsqueeze(0)

            if self.use_augmentation:
                transform = RandAffine(prob=0.3, rotate_range=(0, 0.2), scale_range=(-0.1, 0))
                img = transform(img)
                time.sleep(0.1)

            return patient["id"], img, xray, torch.tensor(label)
    
    def __len__(self):
        return len(self.patients)
    
    @property
    def samples_shape(self):
        shape = (150,220,280) 
        return shape
