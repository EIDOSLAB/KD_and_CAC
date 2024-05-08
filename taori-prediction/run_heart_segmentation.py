# Train a pre-trained network from segmentation_models_pytorch package for segmentation of heart from a 2D image of a CT scan.
# Dataset: Kaggle heart dataset from https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation

import glob
import imageio.v2 as imageio
import logging as log
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import ssl
import torch
from datasets import HeartDataset, DatasetType
from torch.utils.data import DataLoader
from utils import natural_key


EPOCHS = 50
DEVICE = 'cuda'


def get_model(name):
    # required to download some pretrained weights
    ssl._create_default_https_context = ssl._create_unverified_context
    
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['heart']
    ACTIVATION = 'sigmoid'
    
    if name == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
            in_channels=1,
            decoder_channels=(128, 64, 32, 16, 8)
        )
    elif name == 'fpn':
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
            in_channels=1
        )
    else:
        raise Exception(f'Invalid model {name}')
    return model
    

if __name__ == '__main__':
    log.basicConfig(level='INFO', format='%(asctime)s [%(levelname)s]: %(message)s')

    train_dataset = HeartDataset(dataset_type=DatasetType.TRAIN, use_augmentation=True)
    valid_dataset = HeartDataset(dataset_type=DatasetType.VALIDATION)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    models = [
        'fpn',
        'unet'
    ]
    
    for model_name in models:
        model = get_model(model_name)
        loss = smp.utils.losses.BCELoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(),
            smp.utils.metrics.Accuracy(),
            smp.utils.metrics.Recall(),
            smp.utils.metrics.Precision()
        ]

        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=1e-4),
        ])
        
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True,
        )
        
        max_score = 0

        log.info(f'Running model {model_name}')
        for i in range(1, EPOCHS+1):
            log.info(f'Epoch: {i}')
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, f'./best_model_{model_name}.pth')
                log.info('Model saved!')

            if i == 26:
                optimizer.param_groups[0]['lr'] = 1e-5
                log.info('Decrease decoder learning rate to 1e-5!')

    test_dataset = HeartDataset(dataset_type=DatasetType.TEST)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    for model_name in models:
        log.info(f'Running best model {model_name} on TEST set')
        best_model = torch.load(f'./best_model_{model_name}.pth', map_location=DEVICE)
        test_epoch = smp.utils.train.ValidEpoch(
            best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )
        test_epoch.run(test_loader)

