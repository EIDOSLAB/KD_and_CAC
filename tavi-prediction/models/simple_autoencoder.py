import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = self._conv_layer(1, 64, 7, (2,2))
        self.conv2 = self._conv_layer(64, 128)
        self.conv3 = self._conv_layer(128, 256)
        self.conv4 = self._conv_layer(256, 512)
        
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2)
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=2)
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2)
        )
        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=2)
        )
        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=3)
        )
        
    def _conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        
        return out
