# Kaggle heart dataset from https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation

import glob
import imageio.v2 as imageio
import torch
import torchvision
from enum import Enum
from torch.utils.data import Dataset
from utils import natural_key
from .dataset_type import DatasetType


class HeartDataset(Dataset):
    def __init__(self,
                 data_path: str='../data/heart-segmentation',
                 dataset_type: DatasetType=DatasetType.TRAIN,
                 use_augmentation: bool=False,
                 dtype: torch.dtype=torch.float32):
        self.images = glob.glob(f'{data_path}/train/*/image/*.png')
        self.images.sort(key=natural_key)
        if dataset_type == DatasetType.TRAIN:
            self.images = [image for image in self.images if '/100056/' not in image
                                                         and '/100072/' not in image
                                                         and '/100089/' not in image
                                                         and '/100093/' not in image]
            self.images = [image for image in self.images if image not in self.images[::5]]
        elif dataset_type == DatasetType.VALIDATION:
            self.images = [image for image in self.images if '/100056/' not in image
                                                         and '/100072/' not in image
                                                         and '/100089/' not in image
                                                         and '/100093/' not in image]
            self.images = self.images[::5]
        elif dataset_type == DatasetType.TEST:
            self.images = [image for image in self.images if '/100056/' in image
                                                          or '/100072/' in image
                                                          or '/100089/' in image
                                                          or '/100093/' in image]
        else:
            raise Exception(f'Invalid dataset type {dataset_type}')
            
            
        self.masks = glob.glob(f'{data_path}/train/*/mask/*.png')
        self.masks.sort(key=natural_key)
        if dataset_type == DatasetType.TRAIN:
            self.masks = [mask for mask in self.masks if '/100056/' not in mask
                                                     and '/100072/' not in mask
                                                     and '/100089/' not in mask
                                                     and '/100093/' not in mask]
            self.masks = [mask for mask in self.masks if mask not in self.masks[::5]]
        elif dataset_type == DatasetType.VALIDATION:
            self.masks = [mask for mask in self.masks if '/100056/' not in mask
                                                     and '/100072/' not in mask
                                                     and '/100089/' not in mask
                                                     and '/100093/' not in mask]
            self.masks = self.masks[::5]
        elif dataset_type == DatasetType.TEST:
            self.masks = [mask for mask in self.masks if '/100056/' in mask
                                                      or '/100072/' in mask
                                                      or '/100089/' in mask
                                                      or '/100093/' in mask]
            
        self.use_augmentation = use_augmentation
        self.dtype = dtype

        if len(self.images) != len(self.masks):
            raise Exception(f'images[{len(self.images)}] and masks[{len(self.masks)}] are not the same amount')
        print(f'loaded dataset with {len(self.images)} images')
    
    def __getitem__(self, i):
        image_raw = imageio.imread(self.images[i])
        image = (torch.from_numpy(image_raw).type(self.dtype) / 127.5) - 1 # normalize in range [-1; 1]
        image = image.unsqueeze(0)

        mask_raw = imageio.imread(self.masks[i])
        mask = (torch.from_numpy(mask_raw) > 0).type(self.dtype)
        mask = mask.unsqueeze(0)

        if self.use_augmentation:
            transform = torchvision.transforms.RandomAffine(degrees=(0,0))
            augmentation_params = transform.get_params(
                degrees=(-10, 10),
                translate=(0.1, 0.1),
                scale_ranges=(0.8, 1.4),
                shears=None,
                img_size=image.size()
            )
            
            image = torchvision.transforms.functional.affine(image, *augmentation_params, fill=-1)
            mask = torchvision.transforms.functional.affine(mask, *augmentation_params, fill=0)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)
    

if __name__ == '__main__':
    dataset = HeartDataset()
    image, mask = dataset[70]
    print(f'loaded image: {image.shape} and mask: {mask.shape}')
    
    validation = HeartDataset(dataset_type=DatasetType.VALIDATION)
    test = HeartDataset(dataset_type=DatasetType.TEST)
