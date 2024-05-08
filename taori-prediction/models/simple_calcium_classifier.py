import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCalciumClassifier(nn.Module):
    def __init__(self, classifier_layers=3):
        super().__init__()
        self.conv1 = self._conv_layer(1, 64, 7, (2,2,1))
        self.conv2 = self._conv_layer(64, 128)
        self.conv3 = self._conv_layer(128, 256)
        self.conv4 = self._conv_layer(256, 512)
        self.adaptavg = nn.AdaptiveAvgPool3d(output_size=1)
        self.classifier = self._classifier(classifier_layers)

    def _conv_layer(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(out_channels)
        )

    def _classifier(self, classifier_layers):
        if classifier_layers == 3:
            return nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        elif classifier_layers == 1:
            return nn.Sequential(
                nn.Linear(512, 2)
            )
        else:
            raise Exception('Invalid number of layers for classifiers. Accept 1 or 3')
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.adaptavg(out)
        out = out.view(-1, 512)
        if hasattr(self, "classifier"):
            out = self.classifier(out)
        else:
            out = self.fc1(out)
            out = self.bn1(out)
            out = torch.relu(out)
            out = self.fc2(out)
            out = self.bn2(out)
            out = torch.relu(out)
            out = self.fc3(out)
        return out
