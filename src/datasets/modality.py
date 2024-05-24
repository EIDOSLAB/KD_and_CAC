import copy
import glob
import logging as log
import numpy as np
import os
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from ct_metadata import CtMetadata
from typing import Tuple
from utils import natural_key
from viz import VolumePlot


class Ct:
    def __init__(self, path: str, file_pattern: str='*.dcm', clip: bool=True, load_metadata: bool=True) -> None:
        self.metadata = {}
        if os.path.isdir(path):
            img = self.__init_series(path, file_pattern, load_metadata)
        elif os.path.isfile(path):
            img = self.__init_single_image(path, load_metadata)
        else:
            raise ValueError(f'Path {path} not found')

        img_np = np.array(sitk.GetArrayFromImage(img), dtype=np.float32)
        if clip:
            img_np.clip(-1000, 1000, img_np)
        self.img = torch.from_numpy(img_np)


    def __init_series(self, path: str, file_pattern: str, load_metadata: bool) -> sitk.Image:
        img_series = glob.glob(f'{path}/{file_pattern}')
        img_series.sort(key=natural_key)
        log.debug(f'Opening CT Scan series from folder {path} ({len(img_series)} images)')
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(img_series)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        img = series_reader.Execute()
        if load_metadata:
            self.metadata = CtMetadata(series_reader)
        return img


    def __init_single_image(self, path: str, load_metadata: bool):
        log.debug(f'Opening CT Scan img from path {path}')
        img = sitk.ReadImage(path)
        if load_metadata:
            self.metadata = CtMetadata(img)
        return img

    
    def crop(self, z: Tuple[int,int], y: Tuple[int,int], x: Tuple[int,int]):
        self.img = self.img[z[0]:z[1], y[0]:y[1], x[0]:x[1]]


    def scale(self, size: Tuple[int,int,int], mode: str='nearest'):
        # possible upsampling modes: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
        if len(size) != 3:
            raise ValueError(f'Invalid scale size {size}. A 3 dimensions tuple expected')
        self.img = F.interpolate(self.img.unsqueeze(0).unsqueeze(0), size=size, mode=mode).squeeze(0).squeeze(0)


    def __create_split(self, start_index, end_index) -> 'Ct':
        split_ct = copy.copy(self)
        split_ct.img = split_ct.img[start_index:end_index]
        split_ct.metadata = copy.copy(self.metadata)
        for key, value in self.metadata.items():
            if isinstance(value, list):
                split_ct.metadata[key] = value[start_index:end_index]
        return split_ct


    def __split(self, slices):
        splitted_cts = []
        for i in range(1, len(slices)):
            splitted_cts.append(self.__create_split(slices[i-1], slices[i]-1))
        if len(slices) > 0:
            splitted_cts.append(self.__create_split(slices[-1], len(self.img)))
        return splitted_cts


    def split_on_discontinuities(self):
        discontinuities = self.metadata.get_slice_location_discontinuities()
        return self.__split(discontinuities)


    @staticmethod
    def load_all_series(path: str):
        if not os.path.exists(path):
            raise ValueError(f'Path {path} not found')
        if not os.path.isdir(path):
            raise ValueError(f'Path {path} is not a folder')

        cts = []
        subfolders = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        for folder in subfolders:
            cts.append(Ct(folder))
        return cts




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
