import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

#from monai.apps.detection.metrics.coco import COCOMetric
#from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
#from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist
#from monai.data.utils import no_collation
from monai.networks.nets import resnet
#from monai.transforms import ScaleIntensityRanged
#from monai.utils import set_determinism

class MonaiRetina(nn.Module):
    def __init__(self, pretrained_path: str='../data/lung_nodule_ct_detection/model.pt',
                 use_first_retina_out: bool=False):
        super().__init__()
        self._use_first_retina_out = use_first_retina_out
        
        retina = self._get_retina()
        retina.load_state_dict(torch.load(pretrained_path))
        
        retina.feature_extractor.fpn = nn.Identity()
        retina.classification_head = nn.Identity()
        retina.regression_head = nn.Identity()
        
        self.retina = retina
        self.adaptavg = nn.AdaptiveAvgPool3d(output_size=1)
        if self._use_first_retina_out:
            linear_size = 256
        else:
            linear_size = 512
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.fc1.out_features, 32)
        self.bn2 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, 2)
        
        
    def forward(self, x):
        out = self.retina(x)['classification']
        if self._use_first_retina_out:
            out = out[0]
        else:
            out = out[1]
        out = self.adaptavg(out)
        out = out.view(-1, self.fc1.in_features)
        out = self.fc1(out)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out
        
        
    def _get_retina(self):
        returned_layers = [1,2]
        base_anchor_shapes = [[6,8,4],[8,6,5],[10,10,6]]

        num_classes = 1
        spatial_dims = 3
        n_input_channels = 1
        conv1_t_stride = (2, 2, 1)
        conv1_t_size = [7,7,7]

        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[1,2,4],
            base_anchor_shapes=base_anchor_shapes,
        )

        backbone = resnet.ResNet(
            block=resnet.ResNetBottleneck,
            layers=[3, 4, 6, 3],
            block_inplanes=resnet.get_inplanes(),
            n_input_channels=n_input_channels,
            conv1_t_stride=conv1_t_stride,
            conv1_t_size=conv1_t_size,
        )
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=spatial_dims,
            pretrained_backbone=False,
            trainable_backbone_layers=None,
            returned_layers=returned_layers,
        )
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
        return RetinaNet(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
