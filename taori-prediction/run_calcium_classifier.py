import argparse
import itertools
import logging as log
import os
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import sys
import time
import torch
import torch.nn as nn
import torchvision
from datasets import CalciumHeartNoBgDataset, CalciumHeartWithBgDataset, CalciumHeartFullSliceDataset
from datasets import CalciumHeartMednormDataset, CalciumHeartRetinanormDataset
from datasets import CalciumDataset
from datasets import DatasetType, XRayNormalizationType
from helpers.binary_classifier_train import TrainEpoch, ValidEpoch, TestEpoch
from helpers.custom_metrics import TP, TN, FP, FN, BA
from helpers.distillation_train import TrainDistillationEpoch, ValidDistillationEpoch
from models.calcium_predict import CalciumPredictNet
from models.resnet_cxr import ResnetCxr
from models.simple_calcium_classifier import SimpleCalciumClassifier
from models.simple_cxr_classifier import SimpleCxrClassifier
from models.monai_retina import MonaiRetina
from models.iodice_cac_detector import IodiceCalciumDetector, unfreeze_lastlayer_encoder
from models.densenet_cxr import DensenetCxr
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm


def prepare_cache(dataset):
    # prepare cache if needed...
    log.info('Preparing cache...')
    cache_loader = DataLoader(dataset, batch_size=1, num_workers=2)
    for idx, sample in enumerate(tqdm(cache_loader)):
        pass
    log.info('Cache ready!')

    
def get_args():
    parser = argparse.ArgumentParser(description='Train or test calcium classifier')
    # dataset stuff
    parser.add_argument('-d', '--dataset', choices=['no-bg', 'with-bg', 'full-slice',
                                                    'mednorm', 'retinanorm',
                                                    'xray', 'xray-iodice', 'xray-heart', 'xray-heart-2'
                                                   ], default='no-bg',
                        help='Which dataset use')
    parser.add_argument('--aug', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('-s', '--scale', type=float, default=None,
                        help='Scale dataset images')
    
    # model stuff
    parser.add_argument('-m', '--model', choices=['simple', 'simple-1', 'simple-3',
                                                  'medicalnet10', 'medicalnet10-4', 'medicalnet10-3', 'medicalnet10-2',
                                                  'medicalnet18', 'medicalnet34', 'medicalnet50',
                                                  'monai-retina',
                                                  'simple-cxr', 'simple-cxr-4-1', 'simple-cxr-3-1', 'simple-cxr-copy-classif',
                                                  'simple-cxr-autoencoder',
                                                  'resnet18', 'resnet34', 'resnet50',
                                                  'densenet-cxr',
                                                  'iodice'
                                                 ], default='simple',
                        help='Which model to use')
    parser.add_argument('--freeze', type=int, default=0,
                        help='Freeze starting <FREEZE> layers of the network')
    
    # training stuff
    parser.add_argument('-k', '--kfold', type=int, default=5,
                        help='K-Fold splits to evaluate model, 0 trains the model on whole dataset directly')
    parser.add_argument('--kfold-not-stratified', action='store_true',
                        help='Perform random kfold instead of stratified')
    parser.add_argument('--kfold-limit', type=int, default=0,
                        help='Execute only first KFOLD_LIMIT folds')
    parser.add_argument('--k-epochs', type=int, default=0,
                        help='Number of epochs to train on k-folds (default EPOCHS/KFOLD)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=2,
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
    
    # distillation stuff
    parser.add_argument('--distillation', action='store_true',
                        help='Use distillation')
    parser.add_argument('--distillation-type', choices=['response', 'feature', 'feature-head'], default='response',
                        help='Type of knowledge to distill (logits output or intermediate layer)')
    parser.add_argument('--teacher', type=str, default=None,
                        help='Path to the .pth pre-trained teacher model to use')
    parser.add_argument('--distillation-temperature', type=float, default=1,
                        help='Temperature to use for distillation')
    parser.add_argument('--distillation-alpha', type=float, default=0.1,
                        help='Weight for student-loss (1-DISTILLATION_ALPHA for distillation loss)')
    parser.add_argument('--distillation-loss', choices=['kldivloss', 'mseloss', 'l1loss'], default='kldivloss',
                        help='Which loss to use for distillation')
    parser.add_argument('--distillation-scale', type=float, default=1,
                        help='Distillation loss scale factor')
    
    # misc
    parser.add_argument('-t', '--test-only', action='store_true',
                        help='Perform only test using already trained network')
    parser.add_argument('-o', '--output-folder', type=str, default='calcium',
                        help='Output subfolder ./results/<value>')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable cache pre-loading at startup (if implemented in dataset it will be filled during 1st epoch)')
    parser.add_argument('--delay', type=int, default=0,
                        help='Start execution after DELAY hours')
    parser.add_argument('--jump', type=int, default=0,
                        help='Continue execution from last completed global epoch')
    return parser.parse_args()


def get_dataset(args, dataset_type):
    augmentation = False if dataset_type == DatasetType.TEST else args.aug
    xray_only = not args.distillation
    if args.dataset == 'no-bg':
        dataset = CalciumHeartNoBgDataset(dataset_type=dataset_type, device=args.device,
                                          use_augmentation=augmentation, scale=args.scale)
        log.info(f'Using CalciumHeartNoBgDataset(dataset_type={dataset_type}, device={args.device}, ' \
                 f'use_augmentation={augmentation}, scale={args.scale})')
    elif args.dataset == 'with-bg':
        dataset = CalciumHeartWithBgDataset(dataset_type=dataset_type, device=args.device)
        log.info(f'Using CalciumHeartWithBgDataset(dataset_type={dataset_type}, device={args.device})')
    elif args.dataset == 'full-slice':
        dataset = CalciumHeartFullSliceDataset(dataset_type=dataset_type, device=args.device)
        log.info(f'Using CalciumHeartFullSliceDataset(dataset_type={dataset_type}, device={args.device})')
    elif args.dataset == 'mednorm':
        dataset = CalciumHeartMednormDataset(dataset_type=dataset_type, device=args.device)
        log.info(f'Using CalciumHeartMednormDataset(dataset_type={dataset_type}, device={args.device})')
    elif args.dataset == 'retinanorm':
        dataset = CalciumHeartRetinanormDataset(dataset_type=dataset_type, device=args.device)
        log.info(f'Using CalciumHeartRetinanormDataset(dataset_type={dataset_type}, device={args.device})')
    elif args.dataset == 'xray':
        dataset = CalciumDataset(dataset_type=dataset_type, device=args.device,
                                 use_augmentation=augmentation, xray_only=xray_only)
        log.info(f'Using CalciumDataset(dataset_type={dataset_type}, device={args.device}, ' \
                 f'use_augmentation={augmentation}, xray_only={xray_only})')
    elif args.dataset == 'xray-iodice':
        dataset = CalciumDataset(dataset_type=dataset_type, device=args.device,
                                 use_augmentation=augmentation,
                                 xray_only=xray_only, xray_normalization=XRayNormalizationType.IODICE)
        log.info(f'Using CalciumDataset(dataset_type={dataset_type}, device={args.device}, ' \
                 f'use_augmentation={augmentation}, ' \
                 f'xray_only={xray_only}, xray_normalization=XRayNormalizationType.IODICE)')
    elif args.dataset == 'xray-heart':
        dataset = CalciumDataset(dataset_type=dataset_type, device=args.device,
                                 use_augmentation=augmentation,
                                 xray_only=xray_only, xray_normalization=XRayNormalizationType.HEART_CROP)
        log.info(f'Using CalciumDataset(dataset_type={dataset_type}, device={args.device}, ' \
                 f'use_augmentation={augmentation}, ' \
                 f'xray_only={xray_only}, xray_normalization=XRayNormalizationType.HEART_CROP)')
    elif args.dataset == 'xray-heart-2':
        dataset = CalciumDataset(dataset_type=dataset_type, device=args.device,
                                 use_augmentation=augmentation,
                                 xray_only=xray_only, xray_normalization=XRayNormalizationType.HEART_CROP_2)
        log.info(f'Using CalciumDataset(dataset_type={dataset_type}, device={args.device}, ' \
                 f'use_augmentation={augmentation}, ' \
                 f'xray_only={xray_only}, xray_normalization=XRayNormalizationType.HEART_CROP_2)')
    else:
        raise Exception('Invalid dataset type')
    return dataset


def get_model(args, dataset_shape):
    if args.model == 'simple' or args.model == 'simple-3':
        model = SimpleCalciumClassifier(classifier_layers=3)
        log.info(f'Using SimpleCalciumClassifier(classifier_layers=3)')
    elif args.model == 'simple-1':
        model = SimpleCalciumClassifier(classifier_layers=1)
        log.info(f'Using SimpleCalciumClassifier(classifier_layers=1)')
    elif args.model == 'medicalnet10' or args.model == 'medicalnet10-4':
        model = CalciumPredictNet(dataset_shape[0], dataset_shape[1], dataset_shape[2], device=args.device, net_type='resnet10')
        log.info(f'Using CalciumPredictNet({dataset_shape[0]}, {dataset_shape[1]}, {dataset_shape[2]}, ' \
                 f'device={args.device}, net_type=resnet10)')
    elif args.model == 'medicalnet10-3':
        model = CalciumPredictNet(dataset_shape[0], dataset_shape[1], dataset_shape[2], device=args.device,
                                  net_type='resnet10', encoder_layers=3)
        log.info(f'Using CalciumPredictNet({dataset_shape[0]}, {dataset_shape[1]}, {dataset_shape[2]}, ' \
                 f'device={args.device}, net_type=resnet10, encoder_layers=3)')
    elif args.model == 'medicalnet10-2':
        model = CalciumPredictNet(dataset_shape[0], dataset_shape[1], dataset_shape[2], device=args.device,
                                  net_type='resnet10', encoder_layers=2)
        log.info(f'Using CalciumPredictNet({dataset_shape[0]}, {dataset_shape[1]}, {dataset_shape[2]}, ' \
                 f'device={args.device}, net_type=resnet10, encoder_layers=2)')
    elif args.model == 'medicalnet18':
        model = CalciumPredictNet(dataset_shape[0], dataset_shape[1], dataset_shape[2], device=args.device, net_type='resnet18')
        log.info(f'Using CalciumPredictNet({dataset_shape[0]}, {dataset_shape[1]}, {dataset_shape[2]}, ' \
                 f'device={args.device}, net_type=resnet18)')
    elif args.model == 'medicalnet34':
        model = CalciumPredictNet(dataset_shape[0], dataset_shape[1], dataset_shape[2], device=args.device, net_type='resnet34')
        log.info(f'Using CalciumPredictNet({dataset_shape[0]}, {dataset_shape[1]}, {dataset_shape[2]}, ' \
                 f'device={args.device}, net_type=resnet34)')
    elif args.model == 'medicalnet50':
        model = CalciumPredictNet(dataset_shape[0], dataset_shape[1], dataset_shape[2], device=args.device, net_type='resnet50')
        log.info(f'Using CalciumPredictNet({dataset_shape[0]}, {dataset_shape[1]}, {dataset_shape[2]}, ' \
                 f'device={args.device}, net_type=resnet50)')
    elif args.model == 'monai-retina':
        model = MonaiRetina()
        log.info(f'Using MonaiRetina()')
    elif args.model == 'simple-cxr':
        model = SimpleCxrClassifier(output_encoder_layer=(args.distillation_type == 'feature'),
                                    add_projection_head=(args.distillation_type == 'feature-head'))
        log.info(f'Using SimpleCxrClassifier(output_encoder_layer={args.distillation_type == "feature"}, '
                 f'add_projection_head={args.distillation_type == "feature-head"})')
    elif args.model == 'simple-cxr-4-1':
        model = SimpleCxrClassifier(classifier_layers=1, output_encoder_layer=(args.distillation_type == 'feature'),
                                    add_projection_head=(args.distillation_type == 'feature-head'))
        log.info(f'Using SimpleCxrClassifier(classifier_layers=1, ' \
                 f'output_encoder_layer={args.distillation_type == "feature"}, '
                 f'add_projection_head={args.distillation_type == "feature-head"})')
    elif args.model == 'simple-cxr-3-1':
        model = SimpleCxrClassifier(conv_layers=3, classifier_layers=1,
                                    output_encoder_layer=(args.distillation_type == 'feature'),
                                    add_projection_head=(args.distillation_type == 'feature-head'))
        log.info(f'Using SimpleCxrClassifier(conv_layers=3, classifier_layers=1, ' \
                 f'output_encoder_layer={args.distillation_type == "feature"}, '
                 f'add_projection_head={args.distillation_type == "feature-head"})')
    elif args.model == 'simple-cxr-copy-classif':
        model = SimpleCxrClassifier(classifier_layers=1, output_encoder_layer=(args.distillation_type == 'feature'),
                                    add_projection_head=(args.distillation_type == 'feature-head'))
        log.info(f'Using SimpleCxrClassifier(classifier_layers=1, ' \
                 f'output_encoder_layer={args.distillation_type == "feature"}, '
                 f'add_projection_head={args.distillation_type == "feature-head"}) and copy classifier from teacher!!!')
        teacher = torch.load(args.teacher, map_location=args.device)
        model.classifier.load_state_dict(teacher.classifier.state_dict())
    elif args.model == 'simple-cxr-autoencoder':
        model = SimpleCxrClassifier(output_encoder_layer=(args.distillation_type == 'feature'),
                                    add_projection_head=(args.distillation_type == 'feature-head'),
                                    load_encoder_path='results/autoencoder/last_model.pth')
        log.info(f'Using SimpleCxrClassifier(output_encoder_layer={args.distillation_type == "feature"}, '
                 f'add_projection_head={args.distillation_type == "feature-head"}, '
                 f'Using SimpleCxrClassifier(load_encoder_path="results/autoencoder/last_model.pth")')
    elif args.model == 'densenet-cxr':
        model = DensenetCxr(add_projection_head=(args.distillation_type == 'feature-head'))
        log.info(f'Using DensenetCxr(add_projection_head={args.distillation_type == "feature-head"})')
    elif args.model == 'iodice':
        model = IodiceCalciumDetector()
        log.info(f'Using IodiceCalciumDetector()')
    elif args.model == 'resnet18':
        model = ResnetCxr(resnet_type='resnet18', output_encoder_layer=(args.distillation_type == 'feature'))
        log.info(f'Using resnet18()')
    elif args.model == 'resnet34':
        model = ResnetCxr(resnet_type='resnet34', output_encoder_layer=(args.distillation_type == 'feature'))
        log.info(f'Using resnet34()')
    elif args.model == 'resnet50':
        model = ResnetCxr(resnet_type='resnet50', output_encoder_layer=(args.distillation_type == 'feature'))
        log.info(f'Using resnet50()')
    else:
        raise Exception(f'Invalid model type {args.model}')
    model = model.to(torch.device(args.device))
    
    freeze_model(args, model)
    return model


def freeze_model(args, model):
    if args.freeze >= 0:
        if args.model.startswith('monai') and args.freeze > 0:
            # freeze only modules of retina encoder
            for name, layer in list(model.retina.feature_extractor.body.named_children())[:args.freeze]:
                log.info(f'Layer [FREEZE] retina.feature_extractor.body.{name} - {layer}')
                for param in layer.parameters():
                    param.requires_grad = False
            # print other not freezed modules of retina encoder
            for name, layer in list(model.retina.feature_extractor.body.named_children())[args.freeze:]:
                log.info(f'layer {name} - {layer}')
            # print all other modules
            for name, layer in list(model.named_children())[1:]:
                log.info(f'layer {name} - {layer}')
        else:
            for name, layer in list(model.named_children())[:args.freeze]:
                log.info(f'Layer [FREEZE] {name} - {layer}')
                for param in layer.parameters():
                    param.requires_grad = False
            for name, layer in list(model.named_children())[args.freeze:]:
                log.info(f'layer {name} - {layer}')
    else:
        raise Exception(f'Invalid freeze parameter {args.freeze}')

                        
def get_teacher_model(args):
    teacher = torch.load(args.teacher, map_location=args.device)
    if args.distillation_type.startswith('feature'):
        if hasattr(teacher, "classifier"):
            teacher.classifier = nn.Identity()
        elif hasattr(teacher, "fc3"):
            teacher.fc3 = nn.Identity()
            teacher.bn2 = nn.Identity()
            teacher.fc2 = nn.Identity()
            teacher.bn1 = nn.Identity()
            teacher.fc1 = nn.Identity()
        else:
            raise Exception(f'Invalid teacher, do not have classifier or fc* layers')
    return teacher


def get_optimizer(args, model):
    if args.model == 'iodice':
        encoder_last_layer = unfreeze_lastlayer_encoder(model)
        params = [model.fc.parameters(), encoder_last_layer.parameters()]
        return torch.optim.AdamW(itertools.chain(*params), betas=(0.9, 0.999),
                                 eps=1e-08, lr=args.lr, weight_decay=args.l2_regularization)
    elif args.model == 'simple-cxr-autoencoder':
        return torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999),
                                 eps=1e-08, lr=args.lr, weight_decay=args.l2_regularization)
    else:
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


def get_train_epoch(args, model, loss, metrics, optimizer):
    if args.distillation:
        if args.distillation_loss == 'kldivloss':
            distillation_loss = nn.KLDivLoss(reduction='batchmean')
            log.info(f'Using distillation loss KLDivLoss(reduction=batchmean)')
        elif args.distillation_loss == 'mseloss':
            distillation_loss = nn.MSELoss(reduction='mean')
            log.info(f'Using distillation loss MSELoss(reduction=mean)')
        elif args.distillation_loss == 'l1loss':
            distillation_loss = nn.L1Loss()
            log.info(f'Using distillation loss L1Loss()')
        else:
            raise Exception(f'Invalid distillation loss {args.distillation_loss}')
        
        train_epoch = TrainDistillationEpoch(
            teacher=get_teacher_model(args),
            student=model,
            loss=loss,
            distillation_loss=distillation_loss,
            metrics=metrics,
            optimizer=optimizer,
            temperature=args.distillation_temperature,
            alpha=args.distillation_alpha,
            distillation_loss_scale=args.distillation_scale,
            device=args.device,
            cache_teacher_results=True,
            feature_based_kd=args.distillation_type.startswith('feature'),
            verbose=True)
    else:
        train_epoch = TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=args.device, verbose=True)
    return train_epoch


def get_valid_epoch(args, model, loss, metrics):
    if args.distillation:
        valid_epoch = ValidDistillationEpoch(model, loss=loss, metrics=metrics, device=args.device, verbose=True)
    else:
        valid_epoch = ValidEpoch(model, loss=loss, metrics=metrics, device=args.device, verbose=True)
    return valid_epoch


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
    
    if args.jump == 0:
        log.info(f'Arguments: {args}')
    
    global_epoch = 0
    
    if args.delay > 0:
        log.info(f'Sleep {args.delay} hours')
        time.sleep(3600 * args.delay)
        log.info(f'Wake up!')

    torch.multiprocessing.set_start_method('spawn')
    
    loss = smp.utils.losses.CrossEntropyLoss()
    metrics = [
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(), BA(),
        smp.utils.metrics.Recall(),
        smp.utils.metrics.Precision(),
        TP(), TN(), FP(), FN()
    ]
    
    # TRAIN
    if not args.test_only:
        dataset = get_dataset(args, dataset_type=DatasetType.TRAIN)
        if not args.no_cache:
            prepare_cache(dataset)
        
        # kFold validation to understand if model learns as expected
        if args.kfold > 0:
            if args.k_epochs == 0:
                k_epochs = args.epochs//args.kfold
            else:
                k_epochs = args.k_epochs

            if args.kfold_not_stratified:
                kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
            else:
                kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=0)
            for fold, (train_ids, valid_ids) in enumerate(
                    kfold.split(dataset, list(map(lambda p: p["cac_score"] > 0, dataset.patients))
                )
            ):
                if args.kfold_limit > 0 and fold >= args.kfold_limit:
                    break
                
                if args.jump == 0:
                    log.info(f'--- FOLD {fold} ---')
                    log.info(f'Training samples: {train_ids}')
                    log.info(f'Validation samples: {valid_ids}')

                model = get_model(args, dataset.samples_shape)
                optimizer = get_optimizer(args, model)
                scheduler = get_optim_scheduler(args, optimizer)
                
                train_subsampler = SubsetRandomSampler(train_ids)
                valid_subsampler = SubsetRandomSampler(valid_ids)

                train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size,
                                          sampler=train_subsampler, drop_last=args.drop_last)
                valid_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size,
                                          sampler=valid_subsampler)

                train_epoch = get_train_epoch(args, model, loss, metrics, optimizer)
                valid_epoch = get_valid_epoch(args, model, loss, metrics)

                best_score = 0
                best_score_metric = 'fscore'

                for i in range(1, k_epochs+1):
                    global_epoch += 1
                    if args.jump > global_epoch:
                        continue
                    elif args.jump == global_epoch:
                        # load last model
                        model = torch.load(os.path.join(results_path, f'last_model_{fold}.pth'), map_location=args.device)
                        freeze_model(args, model)
                        optimizer = get_optimizer(args, model)
                        train_epoch = get_train_epoch(args, model, loss, metrics, optimizer)
                        valid_epoch = get_valid_epoch(args, model, loss, metrics)
                        continue
                        
                    log.info(f'[Fold {fold}] Starting Epoch: {i} --({global_epoch})')
                    train_logs = train_epoch.run(train_loader)
                    valid_logs = valid_epoch.run(valid_loader)

                    if best_score <= valid_logs[best_score_metric]:
                        best_score = valid_logs[best_score_metric]
                        torch.save(model, os.path.join(results_path, f'best_model_{fold}.pth'))
                        log.info(f'[Fold {fold} Epoch {i}] Best {best_score_metric} model saved!')
                    torch.save(model, os.path.join(results_path, f'last_model_{fold}.pth'))

                    for k,v in train_logs.items():
                        log.info(f'[Fold {fold} Epoch {i}] Train stats {k}: {v}')
                    for k,v in valid_logs.items():
                        log.info(f'[Fold {fold} Epoch {i}] Valid stats {k}: {v}')
                        
                    if scheduler is not None:
                        scheduler.step()

                    #time.sleep(1)
        
        # train on whole dataset
        if args.jump > global_epoch:
            log.info(f'--- ALL ---')
            log.info(f'Training on whole train dataset')
        
        model = get_model(args, dataset.samples_shape)
        optimizer = get_optimizer(args, model)
        scheduler = get_optim_scheduler(args, optimizer)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size,
                                  shuffle=True, drop_last=args.drop_last)
        
        train_epoch = get_train_epoch(args, model, loss, metrics, optimizer)
        
        best_score = 0
        best_score_metric = 'fscore'
        
        for i in range(1, args.epochs+1):
            global_epoch += 1
            if args.jump > global_epoch:
                continue
            elif args.jump == global_epoch:
                # load last model
                model = torch.load(os.path.join(results_path, f'last_model.pth'), map_location=args.device)
                freeze_model(args, model)
                optimizer = get_optimizer(args, model)
                train_epoch = get_train_epoch(args, model, loss, metrics, optimizer)
                continue
                        
            log.info(f'[ALL] Starting Epoch: {i} --({global_epoch})')
            train_logs = train_epoch.run(train_loader)

            if best_score < train_logs[best_score_metric]:
                best_score = train_logs[best_score_metric]
                torch.save(model, os.path.join(results_path, f'best_model.pth'))
                log.info(f'[All Epoch {i}] Best {best_score_metric} model saved!')
            torch.save(model, os.path.join(results_path, f'last_model.pth'))

            for k,v in train_logs.items():
                log.info(f'[ALL Epoch {i}] Train stats {k}: {v}')
                
            if scheduler is not None:
                scheduler.step()
                    
            #time.sleep(1)
        
        # release memory
        dataset = None
        model = None

    # TEST
    test_dataset = get_dataset(args, dataset_type=DatasetType.TEST)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    executed_folds = args.kfold if args.kfold_limit == 0 else args.kfold_limit
    model_paths = [f'best_model_{i}.pth' for i in range(0, executed_folds)] \
                + [f'last_model_{i}.pth' for i in range(0, executed_folds)] \
                + ['best_model.pth', 'last_model.pth']
    
    for model_path in model_paths:
        log.info(f'Running {model_path} on TEST set')
        best_model = torch.load(os.path.join(results_path, model_path), map_location=args.device)
        test_epoch = TestEpoch(best_model, loss=loss, metrics=metrics, device=args.device, verbose=True)
        test_logs = test_epoch.run(test_loader)
        for k,v in test_logs.items():
            log.info(f'[TEST {model_path}] Stats {k}: {v}')
