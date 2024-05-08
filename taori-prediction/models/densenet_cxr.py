import torch
import torch.nn as nn
import sys

from .model_chexpert import HierarchicalResidual

class DensenetCxr(torch.nn.Module):
    def __init__(self, add_projection_head=False):
        super().__init__()
        
        self._add_projection_head = add_projection_head

        size_fc = 1000
        model_chexpert = HierarchicalResidual(encoder = 'efficientnet-b0')

        del model_chexpert.fc1
        del model_chexpert.fc2

        self.encoder = model_chexpert.encoder
        self.fc = nn.Sequential(
            nn.Linear(size_fc, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        if self._add_projection_head:
            self.projection = nn.Sequential(
                nn.Linear(size_fc, 512)
            )

        for param in self.encoder.parameters():
            param.requires_grad = True
            
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.encoder(x)
        encoder_out = torch.flatten(out, 1)
        out = self.fc(encoder_out)
        
        if self._add_projection_head and self.training:
            proj_out = self.projection(encoder_out)
            return out, proj_out
        else:
            return out
