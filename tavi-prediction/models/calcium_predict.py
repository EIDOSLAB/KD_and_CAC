import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet


class CalciumPredictNet(nn.Module):
    def __init__(self, depth=150, width=280, height=220, device: str='cuda:1',
                 net_type: str='resnet10', encoder_layers: int=4, dropout: float=0):
        super().__init__()
        self._encoder_layers = encoder_layers
        self._dropout = dropout
        
        if net_type == 'resnet10':
            MODEL_PATH = '../data/medicalnet-pretrain/resnet_10_23dataset.pth'
            net = resnet.resnet10(sample_input_W=None, sample_input_H=None, sample_input_D=None,
                                  shortcut_type='B', no_cuda=False, num_seg_classes=10)
        elif net_type == 'resnet18':
            MODEL_PATH = '../data/medicalnet-pretrain/resnet_18_23dataset.pth'
            net = resnet.resnet18(sample_input_W=None, sample_input_H=None, sample_input_D=None,
                                  shortcut_type='A', no_cuda=False, num_seg_classes=10)
            resnet_shortcut = 'A'
        elif net_type == 'resnet34':
            MODEL_PATH = '../data/medicalnet-pretrain/resnet_34_23dataset.pth'
            net = resnet.resnet34(sample_input_W=None, sample_input_H=None, sample_input_D=None,
                                  shortcut_type='A', no_cuda=False, num_seg_classes=10)
        elif net_type == 'resnet50':
            MODEL_PATH = '../data/medicalnet-pretrain/resnet_50_23dataset.pth'
            net = resnet.resnet50(sample_input_W=None, sample_input_H=None, sample_input_D=None,
                                  shortcut_type='B', no_cuda=False, num_seg_classes=10)
        else:
            raise Exception(f'Unknown net {net_type}')
        
        net.to(torch.device(device))
        
        pretrain = torch.load(MODEL_PATH, map_location=device)
        net_dict = net.state_dict() 

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        net.load_state_dict(net_dict)
        
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        linear_size = 128
        if encoder_layers > 2:
            self.layer3 = net.layer3
            linear_size = 256
        if encoder_layers > 3:
            self.layer4 = net.layer4
            linear_size = 512
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.adaptavg = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc1 = nn.Linear(linear_size, linear_size // 2)
        self.bn_fc1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.bn_fc2 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, 2)
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        if self._encoder_layers > 2:
            out = self.layer3(out)
        if self._encoder_layers > 3:
            out = self.layer4(out)
        
        if self._dropout > 0:
            out = self.dropout(out)

        out = self.adaptavg(out)
        out = out.view(-1, self.fc1.in_features)
        out = torch.relu(self.bn_fc1(self.fc1(out)))
        out = torch.relu(self.bn_fc2(self.fc2(out)))
        out = self.fc3(out)
        return out
