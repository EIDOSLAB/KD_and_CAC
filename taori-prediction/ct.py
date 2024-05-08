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


    def __split(self, slices: list[int]) -> list['Ct']:
        splitted_cts = []
        for i in range(1, len(slices)):
            splitted_cts.append(self.__create_split(slices[i-1], slices[i]-1))
        if len(slices) > 0:
            splitted_cts.append(self.__create_split(slices[-1], len(self.img)))
        return splitted_cts


    def split_on_discontinuities(self) -> list['Ct']:
        discontinuities = self.metadata.get_slice_location_discontinuities()
        return self.__split(discontinuities)


    @staticmethod
    def load_all_series(path: str) -> list['Ct']:
        if not os.path.exists(path):
            raise ValueError(f'Path {path} not found')
        if not os.path.isdir(path):
            raise ValueError(f'Path {path} is not a folder')

        cts = []
        subfolders = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        for folder in subfolders:
            cts.append(Ct(folder))
        return cts


if __name__ == '__main__':
    log.basicConfig(level='DEBUG', format='%(asctime)s [%(levelname)s]: %(message)s')
    #ct = Ct('../data/manifest-1668678461097/NSCLC Radiogenomics/AMC-027/04-28-1994-NA-VascularGATEDCHESTCTA Adult-45663/6.000000-BEST DIASTOLIC 69 -31024')
    ct = Ct('../data/OSIC/ID00388637202301028491611/')
    ct.metadata.print()
    #ct.scale((600, 512, 512))
    #cts = ct.split_on_discontinuities()

    plotter = VolumePlot(ct.img)
    plotter.plot_interactive()
