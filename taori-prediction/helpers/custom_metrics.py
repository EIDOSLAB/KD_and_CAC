import re
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def _tp(y_pr, y_gt):
    return torch.sum(y_gt * y_pr, dtype=torch.float)


def _tn(y_pr, y_gt):
    return torch.sum((1-y_gt) * (1-y_pr), dtype=torch.float)


def _fp(y_pr, y_gt):
    return torch.sum(y_pr, dtype=torch.float) - _tp(y_pr, y_gt)


def _fn(y_pr, y_gt):
    return torch.sum(y_gt, dtype=torch.float) - _tp(y_pr, y_gt)


class BA(smp.utils.base.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        tp = _tp(y_pr, y_gt)
        tn = _tn(y_pr, y_gt)
        fp = _fp(y_pr, y_gt)
        fn = _fn(y_pr, y_gt)
        if (tp + fn) == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)
        if (tn + fp) == 0:
            tnr = 0
        else:
            tnr = tn / (tn + fp)
        return (tpr + tnr) / 2


class TP(smp.utils.base.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        return _tp(y_pr, y_gt)
    
    
class TN(smp.utils.base.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        return _tn(y_pr, y_gt)
    
    
class FP(smp.utils.base.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        return _fp(y_pr, y_gt)
    
    
class FN(smp.utils.base.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        return _fn(y_pr, y_gt)
