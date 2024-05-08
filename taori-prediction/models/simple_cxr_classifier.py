import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCxrClassifier(nn.Module):
    def __init__(self, conv_layers=4, classifier_layers=3,
                 output_encoder_layer=False,
                 add_projection_head=False,
                 load_encoder_path: str=None):
        super().__init__()
        if conv_layers < 1 or conv_layers > 4:
            raise Exception('Invalid number of convolutional layers. Accept 1 to 4')
        if output_encoder_layer and add_projection_head:
            raise Exception('Only one of output_encoder_layer and add_projection_head can be True')
        
        self.conv_layers = conv_layers
        self._output_encoder_layer = output_encoder_layer
        self._add_projection_head = add_projection_head
        
        self.conv1 = self._conv_layer(1, 64, 7, (2,2))
        if self.conv_layers >= 2:
            self.conv2 = self._conv_layer(64, 128)
        if self.conv_layers >= 3:
            self.conv3 = self._conv_layer(128, 256)
        if self.conv_layers >= 4:
            self.conv4 = self._conv_layer(256, 512)
        self.adaptavg = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = self._classifier(classifier_layers)
        if self._add_projection_head:
            self.projection = self._projection_head()
            
        if load_encoder_path is not None:
            base_model = torch.load(load_encoder_path)
            self.conv1.load_state_dict(base_model.conv1.state_dict())
            self.conv2.load_state_dict(base_model.conv2.state_dict())
            self.conv3.load_state_dict(base_model.conv3.state_dict())
            self.conv4.load_state_dict(base_model.conv4.state_dict())
        
    def _conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def _projection_head(self):
        in_features = 512 // 2**(4-self.conv_layers)
#         return nn.Sequential(
#             nn.Linear(in_features, 128)
#         )
        return nn.Sequential(
            nn.Linear(in_features, in_features//4),
            nn.BatchNorm1d(in_features//4),
            nn.ReLU(),
            nn.Linear(in_features//4, in_features//16)
        )
    
    def _classifier(self, classifier_layers):
        in_features = 512 // 2**(4-self.conv_layers)
        
        if classifier_layers == 3:
            return nn.Sequential(
                nn.Linear(in_features, in_features//4),
                nn.BatchNorm1d(in_features//4),
                nn.ReLU(),
                nn.Linear(in_features//4, in_features//16),
                nn.BatchNorm1d(in_features//16),
                nn.ReLU(),
                nn.Linear(in_features//16, 2)
            )
        elif classifier_layers == 1:
            return nn.Sequential(
                nn.Linear(in_features, 2)
            )
        else:
            raise Exception('Invalid number of layers for classifiers. Accept 1 or 3')

    def forward(self, x):
        out = self.conv1(x)
        if self.conv_layers >= 2:
            out = self.conv2(out)
        if self.conv_layers >= 3:
            out = self.conv3(out)
        if self.conv_layers >= 4:
            out = self.conv4(out)
        out = self.adaptavg(out)
        encoder_out = out.view(-1, self.classifier[0].in_features)
        out = self.classifier(encoder_out)
        
        if self._add_projection_head and self.training:
            proj_out = self.projection(encoder_out)
            return out, proj_out
        elif self._output_encoder_layer and self.training:
            # used for feature-based KD
            return out, encoder_out
        else:
            return out
