import glob
import logging as log
import numpy as np
import os
import SimpleITK as sitk
import torch
from ct_metadata import CtMetadata
from torchvision import transforms as T
from typing import Tuple
from utils import natural_key


class XRay:
    def __init__(self, path: str, file_pattern: str='*.dcm', normalize: bool=False, load_metadata: bool=True) -> None:
        self.metadata = {}
        if os.path.isdir(path):
            img_path = self._read_folder(path, file_pattern)
        elif os.path.isfile(path):
            img_path = path
        else:
            raise ValueError(f'Path {path} not found')
        
        log.debug(f'Opening XRay img from path {img_path}')
        img = sitk.ReadImage(img_path)
        if load_metadata:
            self.metadata = CtMetadata(img)
        
        img_np = np.array(sitk.GetArrayFromImage(img), dtype=np.float32)
        self.img = torch.from_numpy(img_np)
        
        if normalize:
            minimum = self.img.min()
            maximum = self.img.max()
            offset = (maximum + minimum) / 2
            ratio = (maximum - minimum) / 2
            self.img = (self.img - offset) / ratio
        
            norm = T.Normalize(self.img.mean(), self.img.std())
            self.img = norm(self.img)

    def _read_folder(self, path, file_pattern):
        img_files = glob.glob(f'{path}/{file_pattern}')
        img_files.sort(key=natural_key)
        
        if 'CAC_045' in path:
            return img_files[1]
        elif 'CAC_316' in path:
            return img_files[1]
        
        if len(img_files) == 0:
            raise ValueError(f'Path {path} contains {len(img_files)} files')
        elif len(img_files) == 1:
            img_path = img_files[0]
        else:
            img_path = None
            for img_file in img_files:
                img = sitk.ReadImage(img_file)
                metadata = CtMetadata(img)
                
                if 'Series Description' in metadata:
                    desc = metadata['Series Description'].split()
                    desc_low = metadata['Series Description'].lower().split()
                    if 'PA' in desc or 'AP' in desc or 'a.p.' in desc or 'p.a.' in desc or 'letto' in desc_low:
                        img_path = img_file
                        break
                elif 'Acquisition Device Processing Description' in metadata:
                    desc = metadata['Acquisition Device Processing Description'].split()
                    desc_low = metadata['Acquisition Device Processing Description'].lower().split()
                    if 'PA' in desc or 'AP' in desc or 'a.p.' in desc or 'p.a.' in desc or 'letto' in desc_low:
                        img_path = img_file
                        break
                elif 'Protocol Name' in metadata:
                    desc = metadata['Protocol Name'].split()
                    desc_low = metadata['Protocol Name'].lower().split()
                    if 'PA' in desc or 'AP' in desc or 'a.p.' in desc or 'p.a.' in desc or 'letto' in desc_low:
                        img_path = img_file
                        break
            if img_path is None:
                img_path = img_files[0]
        return img_path
