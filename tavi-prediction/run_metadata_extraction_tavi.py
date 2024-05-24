import csv
import glob
import logging as log
import os
from ct import Ct
from utils import natural_key


TAVI_FOLDER = '/data/tavi'


def get_all_metadata():
    metadata = dict()
    
    folders = glob.glob(f'{TAVI_FOLDER}/MOL*/')
    folders.sort(key=natural_key)

    patient_no = 1
    for folder in folders:
        patient = os.path.basename(os.path.normpath(folder))
        ct_path = os.path.join(folder, 'ct')
        log.info(f'Reading patient {patient_no}/{len(folders)} {patient} path {ct_path}')
        ct = Ct(ct_path)
        metadata[patient] = ct.metadata
        metadata[patient]['torch-shape'] = list(ct.img.shape)
        
        sub_cts = ct.split_on_discontinuities()
        for i in range(0,len(sub_cts)):
            metadata[f'{patient}_{i}'] = sub_cts[i].metadata
            metadata[f'{patient}_{i}']['torch-shape'] = list(sub_cts[i].img.shape)
            metadata[f'{patient}_{i}']['sub-ct'] = True
        
        patient_no += 1
    return metadata


def get_all_keys(metadata):
    keys = set()
    for patient, meta in metadata.items():
        keys |= set(meta.keys())
    return keys


def get_trivial_keys(keys, metadata):
    # remove keys that are equal for all patients and put on a separate file
    trivial_keys = dict()
    for key in keys:
        all_equals = True
        value = None
        for patient in metadata.keys():
            try:
                if value is None:
                    value = metadata[patient][key]
                elif value != metadata[patient][key]:
                    raise Exception('values are different')
            except:
                all_equals = False
                break
        if all_equals == True:
            trivial_keys[key] = value
    return trivial_keys


if __name__ == '__main__':
    log.basicConfig(level='DEBUG', format='%(asctime)s [%(levelname)s]: %(message)s')

    if not os.path.exists('metadata/tavi'):
        os.makedirs('metadata/tavi', exist_ok=True)
    
    metadata = get_all_metadata()
    log.info(f'{len(metadata)} metadata read')
    
    for patient, meta in metadata.items():
        try:
            discontinuities = meta.get_slice_location_discontinuities()
            meta['discontinuities'] = discontinuities
            meta['discontinuities count'] = len(discontinuities)
        except Exception as e:
            log.error(f'Error finding discontinuities for {patient}: {e}')

    for patient, meta in metadata.items():
        with open(os.path.join('metadata', 'tavi', f'{patient}.txt'), 'w') as file:
            for key, value in meta.items():
                file.write(f'{key}="{value}"\n')
    
    keys = get_all_keys(metadata)
    trivial_keys = get_trivial_keys(keys, metadata)
    
    with open(os.path.join('metadata', 'tavi', 'general_metadata.txt'), 'w') as file:
        for key, value in trivial_keys.items():
            file.write(f'{key}="{value}"\n')
            
    # remove common metadata equal for all series
    for key in trivial_keys:
        keys.remove(key)
    
    # remove useless metadata
    keys.remove('SOP Instance UID')
    
    #remove long metadata and add them at end of line (excel truncate long lines)
    long_keys = ['Slice Location',
                 'Nominal Percentage of Cardiac Phase',
                 'Image Position (Patient)',
                 'X-Ray Tube Current',
                 'Content Time',
                 'Instance Creation Time',
                 'Series Time',
                 'Instance Number']
    for key in long_keys:
        try:
            keys.remove(key)
        except:
            pass
    keys = list(keys)
    for key in long_keys:
        keys.append(key)

    with open(os.path.join('metadata', 'tavi', 'metadata.csv'), 'w') as file:
        csv_writer = csv.writer(file, dialect='excel')
        csv_writer.writerow(['MolID'] + keys)

        for patient in metadata.keys():
            patient_row = [patient]
            for key in keys:
                try:
                    value = str(metadata[patient][key])
                    value = value.replace('\r\n', ' ').replace('\n', ' ').strip()
                    value = (value[:10000] + '...') if len(value) > 10000 else value
                    patient_row.append(value)
                except:
                    patient_row.append('N/A')
            csv_writer.writerow(patient_row)
