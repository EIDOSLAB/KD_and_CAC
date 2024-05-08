import logging as log
import SimpleITK as sitk
import statistics
from dicom_metadata_keys import dicom_metadata_keys as dcm_keys
from typing import Tuple


class CtMetadata(dict):
    def __init__(self, metadata_reader):
        super(CtMetadata, self).__init__()
        if isinstance(metadata_reader, sitk.Image):
            # single image in dicom
            for key in metadata_reader.GetMetaDataKeys():
                self[self.__get_dicom_metadata_name(key)] = metadata_reader.GetMetaData(key)
        elif isinstance(metadata_reader, sitk.ImageSeriesReader):
            # dicom series
            img_series = metadata_reader.GetFileNames()
            for i in range(len(img_series)):
                if (metadata_reader.GetMetaDataKeys(0) != metadata_reader.GetMetaDataKeys(i)):
                    log.debug('Image metadata are non uniform in the serie')
                    break

            for key in metadata_reader.GetMetaDataKeys(0):
                values = []
                for i in range(len(img_series)):
                    try:
                        values.append(metadata_reader.GetMetaData(i, key))
                    except:
                        values.append(None)
                self[self.__get_dicom_metadata_name(key)] = values
        else:
            raise Exception('Unsupported metadata reader! Cannot create metadata of Ct')

            
    def __setitem__(self, k, v):
        if isinstance(v, list) and len(set(v)) == 1:
            # remove replicated values if all slices have same value
            super(CtMetadata, self).__setitem__(k, v[0])
        else:
            super(CtMetadata, self).__setitem__(k, v)
            
            
    def __get_dicom_metadata_name(self, key: str):
        try:
            return dcm_keys[key.upper()]
        except:
            return key
        
    
    def get_voxel_size(self) -> Tuple[int]:
        w, h = [float(x) for x in self['Pixel Spacing'].split('\\')]
        depths = [abs(float(self['Slice Location'][i]) - float(self['Slice Location'][i-1]))
                  for i in range(1, len(self['Slice Location']))]
        d = statistics.mode(depths)
        return (d, w, h)


    def get_slice_location_discontinuities(self) -> list[int]:
        discontinuities = []
        direction = None
        for i in range(1, len(self['Slice Location'])):
            if float(self['Slice Location'][i]) > float(self['Slice Location'][i-1]):
                if direction == None:
                    direction = 'ASC'
                    discontinuities.append(i-1)
                elif direction != 'ASC':
                    direction = None
            else:
                if direction == None:
                    direction = 'DESC'
                    discontinuities.append(i-1)
                elif direction != 'DESC':
                    direction = None
        return discontinuities
    

    def print(self):
        for k in self.keys():
            print(f'({k}) -> "{self[k]}"')