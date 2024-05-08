# Use pretrained model from run_heart_segmentation.py to identify box around heart and store its coordinates

import cv2
import glob
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision
from collections import namedtuple
from ct import Ct
from pathlib import Path
from tqdm import tqdm
from utils import natural_key
from viz import VolumePlot

DEVICE = 'cuda:1'
DATA_FOLDER = '/data/calcium_processed/'
MODEL_PATH = './results/heart_segmentation_2/best_model_fpn.pth'
RESULTS_PATH = './results/heart_crop_2/'

Rectangle = namedtuple('Rectangle', 'x y w h')

def get_bounding_rect(mask_slice):
    mask_np = mask_slice.numpy().astype(np.uint8)

    contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    return mx


if __name__ == '__main__':
    log.basicConfig(filename=f'{RESULTS_PATH}/{Path(__file__).stem}.log',
                    level='INFO',
                    format='%(asctime)s [%(levelname)s]: %(message)s')
    
    log.info(f'Using device {DEVICE}')
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    log.info(f'Using model {MODEL_PATH}')
    
    folders = glob.glob(f'{DATA_FOLDER}/*/ct/')
    folders.sort(key=natural_key)
    
    for folder_index in tqdm(range(0, len(folders))):
        folder = folders[folder_index]
        patient = os.path.basename(os.path.normpath(folder[:-4]))
        log.info(f'Processing ct scan {patient}')

        files = os.listdir(folder)
        files.sort(key=natural_key)
        try:
            if len(files) == 0:
                raise Exception(f'Folder {folder} is empty')
            elif len(files) < 10:
                raise Exception(f'Folder {folder} contains {len(files)} files only')
        except Exception as exc:
            log.error(f'EXCEPTION patient {patient}! {str(exc)}')
            continue
            
        # found the pattern with highest number of dcm
        files_patterns = list(set([file[:8] for file in files]))
        files_patterns_count = dict()
        for files_pattern in files_patterns:
            files_patterns_count[files_pattern] = 0;
        for file in files:
            files_patterns_count[file[:8]] += 1;
        files_pattern = max(files_patterns_count, key=files_patterns_count.get)
        
        try:
            log.info(f'Reading ct {patient}/ct/{files_pattern}*.dcm')
            ct = Ct(folder, file_pattern=f'{files_pattern}*.dcm') # IM-000X-
        
            mask = torch.empty(ct.img.shape)
            #mixed = torch.empty(ct.img.shape)
            
            bounding_rects = [Rectangle(0,0,0,0)] * ct.img.shape[0]
            bigger_area = 0
            bigger_area_index = -1

            # find heart in each slice using model
            for i in range(0, ct.img.shape[0]):
                image = ct.img[i] / 1000
                mask[i] = model.predict(image.to(DEVICE).unsqueeze(0).unsqueeze(0)) > 0.8
                #mixed[i] = image * (mask[i]<0.1) + (mask[i]*2)

                x,y,w,h = get_bounding_rect(mask[i])
                
                # rectangle is big enough
                if w > 20 and h > 20:
                    bounding_rects[i] = Rectangle(x,y,w,h)
                    if w*h > bigger_area:
                        bigger_area = w*h
                        bigger_area_index = i
            
            # remove areas not connected to bigger area
            for i in range(bigger_area_index, len(bounding_rects)):
                if bounding_rects[i].w == 0 or bounding_rects[i].h == 0 or \
                    ( # previous slice is empty
                     bounding_rects[i-1].w == 0 and bounding_rects[i-1].h == 0
                    ):
                    bounding_rects[i] = Rectangle(0,0,0,0)

            # remove areas not connected to bigger area
            for i in range(bigger_area_index, 0, -1):
                if bounding_rects[i].w == 0 or bounding_rects[i].h == 0 or \
                    ( # next slice is empty
                     bounding_rects[i+1].w == 0 and bounding_rects[i+1].h == 0
                    ):
                    bounding_rects[i] = Rectangle(0,0,0,0)

            # calculate bounding volume
            bounding_volume = {'min_z': mask.shape[0],
                               'min_y': mask.shape[1],
                               'min_x': mask.shape[2],
                               'max_z': 0,
                               'max_y': 0,
                               'max_x': 0}
            
            for i in range(0, len(bounding_rects)):
                x,y,w,h = bounding_rects[i]
                if w > 0 and h > 0:
                    if bounding_volume['min_z'] > i:   bounding_volume['min_z'] = i
                    if bounding_volume['max_z'] < i:   bounding_volume['max_z'] = i
                    if bounding_volume['min_x'] > x:   bounding_volume['min_x'] = x
                    if bounding_volume['max_x'] < x+w: bounding_volume['max_x'] = x+w
                    if bounding_volume['min_y'] > y:   bounding_volume['min_y'] = y
                    if bounding_volume['max_y'] < y+h: bounding_volume['max_y'] = y+h

            with open(f'{RESULTS_PATH}/{patient}.pkl', 'wb') as file:
                pickle.dump(bounding_volume, file)
        except Exception as exc:
            log.error(f'EXCEPTION! {str(exc)}')
            continue
        finally:
            log.info(f'done {patient} {folder_index}/{len(folders)}')
