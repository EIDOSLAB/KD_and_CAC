import argparse
import logging as log
import os
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import sys
import torch
import torch.nn as nn
from datasets import ChexpertSmall
from datasets import DatasetType, XRayNormalizationType
from helpers.autoencoder_train import TrainAutoencoderEpoch, ValidAutoencoderEpoch
from models.simple_autoencoder import SimpleAutoencoder
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_cache(dataset):
    # prepare cache if needed...
    log.info('Preparing cache...')
    cache_loader = DataLoader(dataset, batch_size=1, num_workers=2)
    for idx, sample in enumerate(tqdm(cache_loader)):
        pass
    log.info('Cache ready!')


def get_args():
    parser = argparse.ArgumentParser(description='Train or test an autoencoder')
    
    # dataset stuff
    parser.add_argument('-d', '--dataset', choices=['chexpert-simple-iodicenorm'
                                                   ], default='chexpert-simple-iodicenorm',
                        help='Which dataset use')
    
    # model stuff
    parser.add_argument('-m', '--model', choices=['simple-autoencoder'
                                                 ], default='simple-autoencoder',
                        help='Which model to use')
    
    # training stuff
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--drop-last', action='store_true',
                        help='Drop last batch of training dataset if not complete')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr-decay-steps', type=int, default=0,
                        help='Decay learning rate (multiply by LR_DECAY_GAMMA) every LR_DECAY_STEPS')
    parser.add_argument('--lr-decay-multisteps', type=str, default=None,
                        help='Decay learning rate (multiply by LR_DECAY_GAMMA) at specified steps')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.5,
                        help='Decay learning rate (multiply by LR_DECAY_GAMMA) every LR_DECAY_STEPS')
    parser.add_argument('--lr-decay-exp', type=float, default=0,
                        help='Decay learning rate exponentially. Multiply LR by LR_DECAY_EXP at every epoch')
    parser.add_argument('--l2-regularization',type=float, default=0,
                        help='L2 regularization penalty')
    
    # misc
    parser.add_argument('-o', '--output-folder', type=str, default='autoencoder',
                        help='Output subfolder ./results/<value>')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable cache pre-loading at startup (if implemented in dataset it will be filled during 1st epoch)')
    
    return parser.parse_args()


def get_dataset(args, dataset_type):
    if args.dataset == 'chexpert-simple-iodicenorm':
        dataset = ChexpertSmall(dataset_type=dataset_type, device=args.device)
        log.info(f'Using ChexpertSmall(dataset_type={dataset_type}, device={args.device})')
    else:
        raise Exception('Invalid dataset type')
    return dataset


def get_model(args):
    if args.model == 'simple-autoencoder':
        model = SimpleAutoencoder()
        log.info(f'Using SimpleAutoencoder()')
    else:
        raise Exception(f'Invalid model type {args.model}')
    model = model.to(torch.device(args.device))
    return model


def get_optimizer(args, model):
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_regularization)


def get_optim_scheduler(args, optimizer):
    if args.lr_decay_steps > 0:
        scheduler = StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma, verbose=True)
    elif args.lr_decay_multisteps is not None:
        milestones = [int(x) for x in args.lr_decay_multisteps.split(',')]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay_gamma, verbose=True)
    elif args.lr_decay_exp > 0:
        scheduler = ExponentialLR(optimizer, gamma=args.lr_decay_exp, verbose=True)
    else:
        scheduler = None
    return scheduler


def configure_logging(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
        
    sys.stdout = open(os.path.join(results_path, f'{Path(__file__).stem}.stdout'), 'a')
    sys.stderr = open(os.path.join(results_path, f'{Path(__file__).stem}.stderr'), 'a')
    
    log.basicConfig(filename=os.path.join(results_path, f'{Path(__file__).stem}.log'),
                    filemode='a',
                    force=True,
                    level='INFO',
                    format='%(asctime)s [%(levelname)s]: %(message)s')


if __name__ == '__main__':
    args = get_args()
    results_path = os.path.join('.', 'results', args.output_folder)
    configure_logging(results_path)
    
    loss = nn.MSELoss(reduction='mean')
    metrics = [
#         smp.utils.metrics.IoU(threshold=0.5),
#         smp.utils.metrics.Fscore(),
#         smp.utils.metrics.Accuracy(),
#         smp.utils.metrics.Recall(),
#         smp.utils.metrics.Precision()
    ]
    
    dataset = get_dataset(args, dataset_type=DatasetType.TRAIN)
    if not args.no_cache:
        prepare_cache(dataset)

    log.info(f'Training on whole train dataset')

    model = get_model(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_optim_scheduler(args, optimizer)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size,
                              shuffle=True, drop_last=True)

    train_epoch = TrainAutoencoderEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=True)

    for i in range(1, args.epochs+1):
        log.info(f'[ALL] Starting Epoch: {i}')
        train_logs = train_epoch.run(train_loader)

        torch.save(model, os.path.join(results_path, f'last_model.pth'))

        for k,v in train_logs.items():
            log.info(f'[ALL Epoch {i}] Train stats {k}: {v}')

        if scheduler is not None:
            scheduler.step()
