import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResnetCxr(nn.Module):
    def __init__(self, resnet_type='resnet18', output_encoder_layer=False):
        super().__init__()
        self._output_encoder_layer = output_encoder_layer
        
        self.resnet = self._resnet(resnet_type)
        self.classifier = self._classifier(resnet_type)
    
    def _resnet(self, resnet_type):
        if resnet_type == 'resnet18':
            resnet = torchvision.models.resnet18()
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif resnet_type == 'resnet34':
            resnet = torchvision.models.resnet34()
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif resnet_type == 'resnet50':
            resnet = torchvision.models.resnet50()
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            raise Exception(f'Unknown model {resnet_type}')
        resnet.fc = nn.Identity()
        return resnet
    
    def _classifier(self, resnet_type):
        if resnet_type == 'resnet18':
            classifier = nn.Linear(512, 2)
        elif resnet_type == 'resnet34':
            classifier = nn.Linear(512, 2)
        elif resnet_type == 'resnet50':
            classifier = nn.Linear(2048, 2)
        else:
            raise Exception(f'Unknown model {resnet_type}')
        return classifier
        
    def forward(self, x):
        encoder_out = self.resnet(x)
        out = self.classifier(encoder_out)
        
        if self._output_encoder_layer and self.training:
            # used for feature-based KD
            return out, encoder_out
        else:
            return out